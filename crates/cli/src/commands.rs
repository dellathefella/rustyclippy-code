//! CLI command implementations.

use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use chrono::Utc;
use deeprust_core::{AgentMode, CacheTurnRecord, Config, SessionStore, SessionSummary};
use deeprust_tools::SubagentRunner;

use crate::agent::{Agent, TurnResult};
use crate::subagent_runner::CliSubagentRunner;
use crate::ui::{fmt_num, Output};

pub async fn run(
    config: Config,
    prompt: String,
    session: Option<String>,
    mode: Option<String>,
    yes: bool,
    json: bool,
    max_steps: Option<usize>,
) -> Result<()> {
    let mut config = config;
    if let Some(mode) = mode {
        config.agent.mode = AgentMode::parse(&mode)?;
    }
    if let Some(steps) = max_steps {
        config.agent.max_steps = steps;
    }
    let output = Output::new(json);
    let cwd = std::env::current_dir().context("resolving current directory")?;
    let runner: Arc<dyn SubagentRunner> =
        Arc::new(CliSubagentRunner::new(config.clone(), cwd.clone(), yes));

    let mut agent = match session {
        Some(id) => Agent::resume(config, id, output.clone(), yes, Some(runner))?,
        None => Agent::new_session(config, cwd, output.clone(), yes, Some(runner))?,
    };

    agent.output.banner(
        &agent.session_id,
        &agent.config.provider.model,
        agent.mode.as_str(),
        &agent.config.provider.base_url,
    );

    let result = agent.run_turn(&prompt).await?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json_result(&agent, &result))?
        );
    } else {
        println!();
        agent.output.turn_summary(&result.usage, result.steps);
    }
    Ok(())
}

pub async fn chat(
    config: Config,
    session: Option<String>,
    mode: Option<String>,
    yes: bool,
) -> Result<()> {
    let mut config = config;
    if let Some(mode) = mode {
        config.agent.mode = AgentMode::parse(&mode)?;
    }
    let output = Output::new(false);
    let cwd = std::env::current_dir().context("resolving current directory")?;
    let runner: Arc<dyn SubagentRunner> =
        Arc::new(CliSubagentRunner::new(config.clone(), cwd.clone(), yes));

    let mut agent = match session {
        Some(id) => Agent::resume(
            config.clone(),
            id,
            output.clone(),
            yes,
            Some(runner.clone()),
        )?,
        None => Agent::new_session(
            config.clone(),
            cwd.clone(),
            output.clone(),
            yes,
            Some(runner.clone()),
        )?,
    };

    agent.output.banner(
        &agent.session_id,
        &agent.config.provider.model,
        agent.mode.as_str(),
        &agent.config.provider.base_url,
    );
    eprintln!("commands: /exit /session /cache /new /help");

    loop {
        print!("› ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        match input {
            "/exit" | "/quit" => break,
            "/help" => {
                eprintln!(
                    "/exit  quit\n/session  show session id\n/cache  show cache report\n/new  start a new session\n/help  this message"
                );
            }
            "/session" => eprintln!("session: {}", agent.session_id),
            "/cache" => print_cache_report(&agent.store, &agent.session_id, false)?,
            "/new" => {
                agent = Agent::new_session(
                    config.clone(),
                    cwd.clone(),
                    output.clone(),
                    yes,
                    Some(runner.clone()),
                )?;
                agent.output.banner(
                    &agent.session_id,
                    &agent.config.provider.model,
                    agent.mode.as_str(),
                    &agent.config.provider.base_url,
                );
            }
            "/mode" => eprintln!(
                "mode changes rewrite the frozen system prompt and tool block, so they would \
                 invalidate the cached prefix. Start a new session with --mode instead."
            ),
            _ => match agent.run_turn(input).await {
                Ok(result) => {
                    println!();
                    agent.output.turn_summary(&result.usage, result.steps);
                }
                Err(error) => agent.output.error(&format!("{error:#}")),
            },
        }
    }
    Ok(())
}

pub fn sessions_list() -> Result<()> {
    let store = SessionStore::default_store();
    let sessions = store.list()?;
    if sessions.is_empty() {
        println!("no sessions yet");
        return Ok(());
    }
    print_session_table(&sessions);
    Ok(())
}

pub fn sessions_show(id: &str) -> Result<()> {
    let store = SessionStore::default_store();
    let (meta, events) = store.replay(id)?;
    println!("session:    {}", meta.id);
    println!("created:    {}", meta.created_at.to_rfc3339());
    println!("cwd:        {}", meta.cwd);
    println!("provider:   {}", meta.provider);
    println!("model:      {}", meta.model);
    println!("mode:       {}", meta.mode);
    println!("system:     {}", short_hash(&meta.system_hash));
    println!("tools:      {}", short_hash(&meta.tools_hash));
    let messages = events
        .iter()
        .filter(|event| matches!(event, deeprust_core::SessionEvent::Message { .. }))
        .count();
    println!("messages:   {messages}");
    println!();
    print_cache_report(&store, id, false)?;
    Ok(())
}

pub fn sessions_delete(id: &str) -> Result<()> {
    let store = SessionStore::default_store();
    store.delete(id)?;
    println!("deleted session {id}");
    Ok(())
}

pub fn cache_report(session: Option<String>, json: bool) -> Result<()> {
    let store = SessionStore::default_store();
    match session {
        Some(id) => print_cache_report(&store, &id, json),
        None => aggregate_report(&store, json),
    }
}

fn print_cache_report(store: &SessionStore, session_id: &str, json: bool) -> Result<()> {
    let records = store.load_cache(session_id)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&records)?);
        return Ok(());
    }
    if records.is_empty() {
        println!("no cache records for session {session_id}");
        return Ok(());
    }

    println!(
        "{:>5} {:>10} {:>10} {:>7} {:>10} {:>8} {:>8}",
        "turn", "prompt", "cached", "hit%", "prefill", "out", "stable"
    );
    for record in &records {
        let hit = if record.prompt_tokens == 0 {
            0.0
        } else {
            record.cached_tokens as f64 / record.prompt_tokens as f64 * 100.0
        };
        println!(
            "{:>5} {:>10} {:>10} {:>6.1}% {:>10} {:>8} {:>8}",
            record.turn,
            fmt_num(record.prompt_tokens),
            fmt_num(record.cached_tokens),
            hit,
            fmt_num(record.prefill_tokens),
            fmt_num(record.completion_tokens),
            if record.stable { "yes" } else { "NO" }
        );
    }
    print_aggregate(&records);
    Ok(())
}

fn aggregate_report(store: &SessionStore, json: bool) -> Result<()> {
    let sessions = store.list()?;
    let mut all: Vec<(String, CacheTurnRecord)> = Vec::new();
    for session in &sessions {
        for record in store.load_cache(&session.id)? {
            all.push((session.id.clone(), record));
        }
    }
    if json {
        let payload: Vec<_> = all
            .iter()
            .map(|(id, record)| {
                serde_json::json!({
                    "session": id,
                    "turn": record.turn,
                    "prompt_tokens": record.prompt_tokens,
                    "cached_tokens": record.cached_tokens,
                    "prefill_tokens": record.prefill_tokens,
                    "completion_tokens": record.completion_tokens,
                    "stable": record.stable,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }
    if all.is_empty() {
        println!("no cache records yet");
        return Ok(());
    }
    println!(
        "{:<24} {:>5} {:>10} {:>10} {:>7} {:>10} {:>8}",
        "session", "turn", "prompt", "cached", "hit%", "prefill", "out"
    );
    for (id, record) in &all {
        let hit = if record.prompt_tokens == 0 {
            0.0
        } else {
            record.cached_tokens as f64 / record.prompt_tokens as f64 * 100.0
        };
        println!(
            "{:<24} {:>5} {:>10} {:>10} {:>6.1}% {:>10} {:>8}",
            short_session(id),
            record.turn,
            fmt_num(record.prompt_tokens),
            fmt_num(record.cached_tokens),
            hit,
            fmt_num(record.prefill_tokens),
            fmt_num(record.completion_tokens)
        );
    }
    let records: Vec<CacheTurnRecord> = all.into_iter().map(|(_, record)| record).collect();
    print_aggregate(&records);
    Ok(())
}

fn print_aggregate(records: &[CacheTurnRecord]) {
    let prompt: usize = records.iter().map(|record| record.prompt_tokens).sum();
    let cached: usize = records.iter().map(|record| record.cached_tokens).sum();
    let prefill: usize = records.iter().map(|record| record.prefill_tokens).sum();
    let output: usize = records.iter().map(|record| record.completion_tokens).sum();
    let unstable = records.iter().filter(|record| !record.stable).count();
    let ratio = if prompt == 0 {
        0.0
    } else {
        cached as f64 / prompt as f64 * 100.0
    };
    println!();
    println!(
        "total: prompt={} cached={} ({:.1}%) prefill={} out={} unstable_turns={}",
        fmt_num(prompt),
        fmt_num(cached),
        ratio,
        fmt_num(prefill),
        fmt_num(output),
        unstable
    );
}

pub fn config_init(path: &std::path::Path, force: bool) -> Result<()> {
    Config::init(path, force)?;
    println!("wrote {}", path.display());
    Ok(())
}

pub fn config_show(config: &Config) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(config)?);
    Ok(())
}

pub async fn doctor(config: &Config) -> Result<()> {
    println!("deeprust doctor");
    println!("  provider:        {}", config.provider.kind.as_str());
    println!("  base_url:        {}", config.provider.base_url);
    println!("  model:           {}", config.provider.model);
    println!(
        "  context:         {} tokens (autocompact at {})",
        fmt_num(config.provider.max_context_tokens),
        fmt_num(
            config
                .cache
                .autocompact_budget(config.provider.max_context_tokens)
        )
    );
    println!(
        "  output:          {} tokens",
        fmt_num(config.provider.max_output_tokens)
    );
    println!(
        "  reasoning:       {}",
        config
            .provider
            .effective_reasoning_effort()
            .unwrap_or("off")
    );
    println!("  cache granularity: {} tokens", config.cache.granularity);
    println!(
        "  tools:           {}",
        deeprust_tools::ToolRegistry::standard(config.agent.mode).len()
    );
    println!();

    let url = format!("{}/models", config.provider.base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(5))
        .timeout(std::time::Duration::from_secs(15))
        .build()?;
    let mut request = client.get(&url);
    if let Some(key) = config.provider.api_key() {
        request = request.header("Authorization", format!("Bearer {key}"));
    }
    match request.send().await {
        Ok(response) => {
            let status = response.status();
            if status.is_success() {
                println!("  endpoint:        OK ({url})");
                if let Ok(value) = response.json::<serde_json::Value>().await {
                    if let Some(models) = value.get("data").and_then(|value| value.as_array()) {
                        for model in models {
                            if let Some(id) = model.get("id").and_then(|value| value.as_str()) {
                                println!("  served model:    {id}");
                            }
                        }
                    }
                }
            } else {
                println!("  endpoint:        HTTP {status} ({url})");
            }
        }
        Err(error) => {
            println!("  endpoint:        unreachable ({url})");
            println!("  error:           {error}");
            println!(
                "  hint:            start the local server or run `deeprust setup` to detect it"
            );
        }
    }
    Ok(())
}

fn json_result(agent: &Agent, result: &TurnResult) -> serde_json::Value {
    serde_json::json!({
        "session": agent.session_id,
        "model": agent.config.provider.model,
        "mode": agent.mode.as_str(),
        "text": result.text,
        "steps": result.steps,
        "usage": {
            "prompt_tokens": result.usage.prompt_tokens,
            "cached_tokens": result.usage.cached_tokens,
            "prefill_tokens": result.usage.prefill_tokens(),
            "completion_tokens": result.usage.completion_tokens,
            "reasoning_tokens": result.usage.reasoning_tokens,
        },
        "cache": result.plan.as_ref().map(|plan| serde_json::json!({
            "prompt_hash": plan.prompt_hash,
            "stable_prefix_messages": plan.stable_prefix_messages,
            "reusable_tokens": plan.reusable_tokens,
            "prefill_tokens": plan.prefill_tokens,
            "cache_hit_ratio": plan.cache_hit_ratio(),
            "stable": plan.stable,
            "cache_cold": plan.cache_cold,
            "granularity": plan.cache_granularity,
        })),
    })
}

fn print_session_table(sessions: &[SessionSummary]) {
    println!(
        "{:<26} {:<20} {:>7} {:>6} {:<8} model",
        "session", "created", "turns", "msgs", "mode"
    );
    for session in sessions {
        println!(
            "{:<26} {:<20} {:>7} {:>6} {:<8} {}",
            session.id,
            session.created_at.format("%Y-%m-%d %H:%M"),
            session.turns,
            session.messages,
            session.mode,
            session.model
        );
    }
}

fn short_hash(hash: &str) -> String {
    hash.chars().take(12).collect()
}

fn short_session(id: &str) -> String {
    id.chars().take(24).collect()
}

/// Interactive setup wizard entry point.
pub async fn setup(path: Option<PathBuf>, yes: bool, force: bool) -> Result<()> {
    crate::wizard::run(path, yes, force).await
}

#[allow(dead_code)]
fn now() -> chrono::DateTime<Utc> {
    Utc::now()
}

#[allow(dead_code)]
fn require_terminal() -> Result<()> {
    if !io::stdin().is_terminal() {
        bail!("this command needs an interactive terminal");
    }
    Ok(())
}

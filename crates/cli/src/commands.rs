//! CLI command implementations.

use std::collections::BTreeMap;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use chrono::Utc;
use rustyclippy_core::{
    AgentMode, CacheTurnRecord, Config, ImageAttachment, SessionStore, SessionSummary,
};
use rustyclippy_tools::SubagentRunner;

use crate::agent::{Agent, TurnResult};
use crate::images;
use crate::subagent_runner::CliSubagentRunner;
use crate::ui::{fmt_bytes, fmt_num, Output};

#[allow(clippy::too_many_arguments)]
pub async fn run(
    config: Config,
    prompt: String,
    session: Option<String>,
    mode: Option<String>,
    yes: bool,
    json: bool,
    max_steps: Option<usize>,
    image_paths: Vec<PathBuf>,
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

    let mut attachments = Vec::with_capacity(image_paths.len());
    for path in &image_paths {
        attachments.push(images::read_image(path)?);
    }

    let mut agent = match session {
        Some(id) => Agent::resume(config, id, output.clone(), yes, Some(runner))?,
        None => Agent::new_session(config, cwd, output.clone(), yes, Some(runner))?,
    };

    agent.output.header(
        env!("CARGO_PKG_VERSION"),
        &agent.session_id,
        &agent.config.provider.model,
        agent.mode.as_str(),
        &agent.config.provider.base_url,
        &agent.tool_ctx.cwd.to_string_lossy(),
    );

    let result = agent.run_turn_with_images(&prompt, attachments).await?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json_result(&agent, &result))?
        );
    } else {
        println!();
        agent
            .output
            .turn_summary(agent.turn, &result.usage, result.steps);
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
    let output = Output::chat();
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

    print_header(&agent);
    agent.output.hint(
        "commands: /exit /session /cache /stats /new /mode build|plan /paste /image <path> /clear-images /help",
    );

    let context = ChatContext {
        cwd: &cwd,
        yes,
        runner: &runner,
    };
    let mut pending_images: Vec<ImageAttachment> = Vec::new();

    loop {
        if !pending_images.is_empty() {
            agent.output.pending_images(pending_images.len());
        }
        print!("{} ", agent.output.prompt_marker());
        let _ = io::stdout().flush();
        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            break;
        }
        let input = line.trim();
        if input.is_empty() && pending_images.is_empty() {
            continue;
        }

        if input.starts_with('/') {
            match handle_command(&mut agent, input, &mut pending_images, &context)? {
                CommandOutcome::Exit => break,
                CommandOutcome::Continue => continue,
            }
        }

        let (prompt, inline_images) = extract_image_paths(input);
        pending_images.extend(inline_images);
        let images = std::mem::take(&mut pending_images);
        match agent.run_turn_with_images(&prompt, images).await {
            Ok(result) => {
                println!();
                agent
                    .output
                    .turn_summary(agent.turn, &result.usage, result.steps);
            }
            Err(error) => agent.output.error(&format!("{error:#}")),
        }
    }
    Ok(())
}

struct ChatContext<'a> {
    cwd: &'a Path,
    yes: bool,
    runner: &'a Arc<dyn SubagentRunner>,
}

enum CommandOutcome {
    Continue,
    Exit,
}

fn print_header(agent: &Agent) {
    agent.output.header(
        env!("CARGO_PKG_VERSION"),
        &agent.session_id,
        &agent.config.provider.model,
        agent.mode.as_str(),
        &agent.config.provider.base_url,
        &agent.tool_ctx.cwd.to_string_lossy(),
    );
}

fn handle_command(
    agent: &mut Agent,
    input: &str,
    pending_images: &mut Vec<ImageAttachment>,
    context: &ChatContext<'_>,
) -> Result<CommandOutcome> {
    let mut parts = input.splitn(2, char::is_whitespace);
    let command = parts.next().unwrap_or("");
    let rest = parts.next().unwrap_or("").trim();

    match command {
        "/exit" | "/quit" => Ok(CommandOutcome::Exit),
        "/help" => {
            agent.output.hint(
                "/exit          quit\n\
                 /session       show session id\n\
                 /cache         per-turn cache report\n\
                 /stats         lifetime token usage for this project\n\
                 /new           start a fresh session\n\
                 /mode [build|plan]  show or switch agent mode\n\
                 /paste         attach the clipboard image to the next prompt\n\
                 /image <path>  attach an image file to the next prompt\n\
                 /images        list pending attachments\n\
                 /clear-images  drop pending attachments\n\
                 /help          this message",
            );
            Ok(CommandOutcome::Continue)
        }
        "/session" => {
            agent.output.hint(&format!("session: {}", agent.session_id));
            Ok(CommandOutcome::Continue)
        }
        "/cache" => {
            print_cache_report(&agent.store, &agent.session_id, false)?;
            Ok(CommandOutcome::Continue)
        }
        "/stats" => {
            print_project_stats(&agent.store, &agent.tool_ctx.cwd, false)?;
            Ok(CommandOutcome::Continue)
        }
        "/new" => {
            let output = agent.output.clone();
            let config = agent.config.clone();
            *agent = Agent::new_session(
                config,
                context.cwd.to_path_buf(),
                output,
                context.yes,
                Some(context.runner.clone()),
            )?;
            pending_images.clear();
            print_header(agent);
            Ok(CommandOutcome::Continue)
        }
        "/mode" => {
            if rest.is_empty() {
                agent.output.hint(&format!("mode: {}", agent.mode.as_str()));
            } else {
                match AgentMode::parse(rest) {
                    Ok(new_mode) => {
                        agent.switch_mode(new_mode)?;
                        print_header(agent);
                    }
                    Err(error) => agent.output.error(&format!("{error:#}")),
                }
            }
            Ok(CommandOutcome::Continue)
        }
        "/paste" => {
            match images::capture_clipboard() {
                Ok(Some(image)) => {
                    agent
                        .output
                        .image_attached(&image.mime, image.approx_bytes());
                    pending_images.push(image);
                }
                Ok(None) => agent
                    .output
                    .warning("clipboard has no image (copy a screenshot or image first)"),
                Err(error) => agent.output.error(&format!("{error:#}")),
            }
            Ok(CommandOutcome::Continue)
        }
        "/image" => {
            if rest.is_empty() {
                agent.output.error("/image needs a file path");
            } else {
                match images::read_image(Path::new(rest)) {
                    Ok(image) => {
                        agent
                            .output
                            .image_attached(&image.mime, image.approx_bytes());
                        pending_images.push(image);
                    }
                    Err(error) => agent.output.error(&format!("{error:#}")),
                }
            }
            Ok(CommandOutcome::Continue)
        }
        "/images" => {
            if pending_images.is_empty() {
                agent.output.hint("no images attached");
            } else {
                for image in pending_images.iter() {
                    agent.output.hint(&format!(
                        "{} · {}",
                        image.mime,
                        fmt_bytes(image.approx_bytes())
                    ));
                }
            }
            Ok(CommandOutcome::Continue)
        }
        "/clear-images" => {
            pending_images.clear();
            agent.output.hint("cleared pending images");
            Ok(CommandOutcome::Continue)
        }
        other => {
            agent
                .output
                .error(&format!("unknown command `{other}` (try /help)"));
            Ok(CommandOutcome::Continue)
        }
    }
}

/// Drag-and-drop / pasted image paths are attached instead of sent as text.
fn extract_image_paths(input: &str) -> (String, Vec<ImageAttachment>) {
    let mut kept: Vec<String> = Vec::new();
    let mut attachments = Vec::new();
    for token in input.split_whitespace() {
        let path = Path::new(token);
        if is_image_path(path) && path.is_file() {
            match images::read_image(path) {
                Ok(image) => attachments.push(image),
                Err(error) => {
                    eprintln!("warning: {error:#}");
                    kept.push(token.to_string());
                }
            }
        } else {
            kept.push(token.to_string());
        }
    }
    (kept.join(" "), attachments)
}

fn is_image_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase())
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "webp" | "gif")
    )
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
        .filter(|event| matches!(event, rustyclippy_core::SessionEvent::Message { .. }))
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

/// Lifetime token usage for every session recorded in one project directory.
pub fn stats(project: Option<PathBuf>, json: bool) -> Result<()> {
    let store = SessionStore::default_store();
    let root = match project {
        Some(path) => path,
        None => std::env::current_dir().context("resolving current directory")?,
    };
    let root = std::fs::canonicalize(&root).unwrap_or(root);
    print_project_stats(&store, &root, json)
}

#[derive(Default)]
struct ProjectTotals {
    sessions: usize,
    turns: usize,
    prompt_tokens: usize,
    cached_tokens: usize,
    prefill_tokens: usize,
    completion_tokens: usize,
    reasoning_tokens: usize,
}

impl ProjectTotals {
    fn add(&mut self, record: &CacheTurnRecord) {
        self.prompt_tokens += record.prompt_tokens;
        self.cached_tokens += record.cached_tokens;
        self.prefill_tokens += record.prefill_tokens;
        self.completion_tokens += record.completion_tokens;
        self.reasoning_tokens += record.reasoning_tokens;
    }

    fn hit_ratio(&self) -> f64 {
        if self.prompt_tokens == 0 {
            0.0
        } else {
            self.cached_tokens as f64 / self.prompt_tokens as f64
        }
    }
}

fn print_project_stats(store: &SessionStore, root: &Path, json: bool) -> Result<()> {
    let sessions = store.list()?;
    let mut totals = ProjectTotals::default();
    let mut models: BTreeMap<String, ProjectTotals> = BTreeMap::new();

    for session in &sessions {
        let cwd =
            std::fs::canonicalize(&session.cwd).unwrap_or_else(|_| PathBuf::from(&session.cwd));
        if cwd != root {
            continue;
        }
        let records = store.load_cache(&session.id)?;
        totals.sessions += 1;
        totals.turns += records.len();
        for record in &records {
            totals.add(record);
        }
        let model = models.entry(session.model.clone()).or_default();
        model.sessions += 1;
        model.turns += records.len();
        for record in &records {
            model.add(record);
        }
    }

    if json {
        let payload = serde_json::json!({
            "project": root.to_string_lossy(),
            "sessions": totals.sessions,
            "turns": totals.turns,
            "prompt_tokens": totals.prompt_tokens,
            "cached_tokens": totals.cached_tokens,
            "prefill_tokens": totals.prefill_tokens,
            "completion_tokens": totals.completion_tokens,
            "reasoning_tokens": totals.reasoning_tokens,
            "cache_hit_ratio": totals.hit_ratio(),
            "models": models.iter().map(|(model, totals)| serde_json::json!({
                "model": model,
                "sessions": totals.sessions,
                "turns": totals.turns,
                "prompt_tokens": totals.prompt_tokens,
                "cached_tokens": totals.cached_tokens,
                "prefill_tokens": totals.prefill_tokens,
                "completion_tokens": totals.completion_tokens,
                "reasoning_tokens": totals.reasoning_tokens,
                "cache_hit_ratio": totals.hit_ratio(),
            })).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    println!("project:  {}", root.display());
    println!("sessions: {}", totals.sessions);
    println!("turns:    {}", totals.turns);
    if totals.sessions == 0 {
        println!();
        println!("no sessions recorded for this project yet");
        return Ok(());
    }
    println!();
    println!(
        "tokens:   prompt={} cached={} ({:.1}%) prefill={} output={} reasoning={}",
        fmt_num(totals.prompt_tokens),
        fmt_num(totals.cached_tokens),
        totals.hit_ratio() * 100.0,
        fmt_num(totals.prefill_tokens),
        fmt_num(totals.completion_tokens),
        fmt_num(totals.reasoning_tokens)
    );
    println!();
    println!("by model:");
    println!(
        "  {:<56} {:>8} {:>6} {:>10} {:>10} {:>7} {:>10} {:>8}",
        "model", "sessions", "turns", "prompt", "cached", "hit%", "prefill", "out"
    );
    for (model, totals) in &models {
        println!(
            "  {:<56} {:>8} {:>6} {:>10} {:>10} {:>6.1}% {:>10} {:>8}",
            model,
            totals.sessions,
            totals.turns,
            fmt_num(totals.prompt_tokens),
            fmt_num(totals.cached_tokens),
            totals.hit_ratio() * 100.0,
            fmt_num(totals.prefill_tokens),
            fmt_num(totals.completion_tokens)
        );
    }
    Ok(())
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
    let reasoning: usize = records.iter().map(|record| record.reasoning_tokens).sum();
    let unstable = records.iter().filter(|record| !record.stable).count();
    let ratio = if prompt == 0 {
        0.0
    } else {
        cached as f64 / prompt as f64 * 100.0
    };
    println!();
    println!(
        "total: prompt={} cached={} ({:.1}%) prefill={} out={} reasoning={} unstable_turns={}",
        fmt_num(prompt),
        fmt_num(cached),
        ratio,
        fmt_num(prefill),
        fmt_num(output),
        fmt_num(reasoning),
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
    println!("rustyclippy-code doctor");
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
        rustyclippy_tools::ToolRegistry::standard(config.agent.mode).len()
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
                "  hint:            start the local server or run `rustyclippy-code setup` to detect it"
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

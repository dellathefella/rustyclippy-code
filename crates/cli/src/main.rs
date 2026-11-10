//! rustyclippy-code: a cache-stable coding agent tuned for DeepSeek V4.1 Flash.

mod agent;
mod commands;
mod history;
mod images;
mod input;
mod subagent_runner;
mod tui;
mod ui;
mod wizard;

use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use rustyclippy_core::Config;

#[derive(Parser)]
#[command(
    name = "rustyclippy-code",
    version,
    about = "Cache-stable coding agent for DeepSeek V4.1 Flash",
    long_about = "A terminal coding agent that keeps the prompt prefix byte-stable so local \
                  inference servers reuse their KV cache instead of re-prefilling the whole \
                  context every turn."
)]
struct Cli {
    /// Path to a config file (default: ./rustyclippy-code.json, then ~/.rustyclippy-code/config.json)
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    /// Override the model id
    #[arg(long, global = true)]
    model: Option<String>,
    /// Override the provider base URL
    #[arg(long, global = true)]
    base_url: Option<String>,
    /// Override reasoning effort (off/low/high/max)
    #[arg(long, global = true)]
    reasoning_effort: Option<String>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run one prompt headlessly
    Run {
        /// Prompt words (joined with spaces)
        #[arg(value_name = "PROMPT", num_args = 1..)]
        prompt: Vec<String>,
        /// Resume an existing session
        #[arg(long)]
        session: Option<String>,
        /// Resume the most recent session recorded for this directory
        #[arg(long = "continue", conflicts_with = "session")]
        continue_session: bool,
        /// Agent mode: build or plan
        #[arg(long)]
        mode: Option<String>,
        /// Auto-approve mutating tools
        #[arg(long)]
        yes: bool,
        /// Emit a JSON result instead of streaming text
        #[arg(long)]
        json: bool,
        /// Override the maximum agent steps
        #[arg(long)]
        max_steps: Option<usize>,
        /// Attach an image file to the prompt (repeatable)
        #[arg(long = "image", value_name = "PATH")]
        images: Vec<PathBuf>,
    },
    /// Interactive REPL
    Chat {
        #[arg(long)]
        session: Option<String>,
        /// Resume the most recent session recorded for this directory
        #[arg(long = "continue", conflicts_with = "session")]
        continue_session: bool,
        #[arg(long)]
        mode: Option<String>,
        #[arg(long)]
        yes: bool,
    },
    /// Full-screen TUI (opencode-style transcript, input box, live totals)
    Tui {
        #[arg(long)]
        session: Option<String>,
        /// Resume the most recent session recorded for this directory
        #[arg(long = "continue", conflicts_with = "session")]
        continue_session: bool,
        #[arg(long)]
        mode: Option<String>,
        #[arg(long)]
        yes: bool,
    },
    /// Lifetime token usage for a project
    Stats {
        /// Project directory (default: current directory)
        #[arg(long)]
        project: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Manage sessions
    Sessions {
        #[command(subcommand)]
        action: SessionsAction,
    },
    /// Inspect cache accounting
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },
    /// Manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Interactive setup wizard (alias: init)
    #[command(alias = "init")]
    Setup {
        #[arg(long)]
        path: Option<PathBuf>,
        /// Accept detected defaults without prompting
        #[arg(long)]
        yes: bool,
        /// Overwrite an existing config
        #[arg(long)]
        force: bool,
    },
    /// Check the endpoint and effective configuration
    Doctor,
}

#[derive(Subcommand)]
enum SessionsAction {
    List,
    Show { id: String },
    Delete { id: String },
}

#[derive(Subcommand)]
enum CacheAction {
    Report {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    Init {
        #[arg(long, default_value = "rustyclippy-code.json")]
        path: PathBuf,
        #[arg(long)]
        force: bool,
    },
    Show,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    if let Err(error) = run().await {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Setup { path, yes, force } => commands::setup(path, yes, force).await,
        command => {
            let mut config = Config::load(cli.config.as_deref())?;
            if let Some(model) = cli.model {
                config.provider.model = model;
            }
            if let Some(base_url) = cli.base_url {
                config.provider.base_url = base_url;
            }
            if let Some(effort) = cli.reasoning_effort {
                config.provider.reasoning_effort = Some(effort);
            }
            config.validate()?;

            match command {
                Command::Run {
                    prompt,
                    session,
                    continue_session,
                    mode,
                    yes,
                    json,
                    max_steps,
                    images,
                } => {
                    let text = prompt.join(" ");
                    if text.trim().is_empty() {
                        bail!("`run` needs a prompt");
                    }
                    commands::run(
                        config,
                        text,
                        session,
                        continue_session,
                        mode,
                        yes,
                        json,
                        max_steps,
                        images,
                    )
                    .await
                }
                Command::Chat {
                    session,
                    continue_session,
                    mode,
                    yes,
                } => commands::chat(config, session, continue_session, mode, yes).await,
                Command::Tui {
                    session,
                    continue_session,
                    mode,
                    yes,
                } => commands::tui(config, session, continue_session, mode, yes).await,
                Command::Stats { project, json } => commands::stats(project, json),
                Command::Sessions { action } => match action {
                    SessionsAction::List => commands::sessions_list(),
                    SessionsAction::Show { id } => commands::sessions_show(&id),
                    SessionsAction::Delete { id } => commands::sessions_delete(&id),
                },
                Command::Cache { action } => match action {
                    CacheAction::Report { session, json } => commands::cache_report(session, json),
                },
                Command::Config { action } => match action {
                    ConfigAction::Init { path, force } => commands::config_init(&path, force),
                    ConfigAction::Show => commands::config_show(&config),
                },
                Command::Doctor => commands::doctor(&config).await,
                Command::Setup { .. } => unreachable!("handled above"),
            }
        }
    }
}

//! Subagent runner: lets the `task` tool spawn a child agent loop.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use rustyclippy_core::{AgentMode, Config};
use rustyclippy_tools::SubagentRunner;

use crate::agent::Agent;
use crate::ui::Output;

pub struct CliSubagentRunner {
    config: Config,
    cwd: PathBuf,
    auto_approve: bool,
}

impl CliSubagentRunner {
    pub fn new(config: Config, cwd: PathBuf, auto_approve: bool) -> Self {
        Self {
            config,
            cwd,
            auto_approve,
        }
    }
}

#[async_trait]
impl SubagentRunner for CliSubagentRunner {
    async fn run(&self, description: &str, prompt: &str, depth: usize) -> Result<String> {
        let mut config = self.config.clone();
        // Subagents always run in build mode: they exist to do work.
        config.agent.mode = AgentMode::Build;
        let output = Output::quiet();
        let runner: Arc<dyn SubagentRunner> = Arc::new(CliSubagentRunner::new(
            config.clone(),
            self.cwd.clone(),
            self.auto_approve,
        ));
        let mut agent = Agent::new_session(
            config,
            self.cwd.clone(),
            output,
            self.auto_approve,
            Some(runner),
        )?;
        agent.tool_ctx.depth = depth;
        agent.tool_ctx.max_depth = self.config.agent.max_depth;
        let result = agent.run_turn(prompt).await?;
        Ok(format!("[subagent: {description}]\n{}", result.text))
    }
}

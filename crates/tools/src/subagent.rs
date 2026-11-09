//! Subagent tasks: the `task` tool delegates a self-contained job to a child
//! agent loop and returns its final answer.

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::registry::{Tool, ToolContext, ToolResult};

/// Implemented by the CLI: spawns a child agent and runs one turn.
#[async_trait]
pub trait SubagentRunner: Send + Sync {
    async fn run(&self, description: &str, prompt: &str, depth: usize) -> Result<String>;
}

pub struct TaskTool;

#[async_trait]
impl Tool for TaskTool {
    fn name(&self) -> &'static str {
        "task"
    }

    fn description(&self) -> &'static str {
        "Delegate a self-contained job to a subagent and return its final answer. \
         Use for independent research or a focused change that would flood the main \
         context. The subagent has the same tools and starts with a fresh transcript."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Short label shown to the user."},
                "prompt": {"type": "string", "description": "Complete, standalone instructions for the subagent."}
            },
            "required": ["description", "prompt"]
        })
    }

    fn mutates(&self) -> bool {
        true
    }

    async fn execute(&self, args: &Value, ctx: &ToolContext) -> Result<ToolResult> {
        let description = args
            .get("description")
            .and_then(|value| value.as_str())
            .unwrap_or("subagent task");
        let prompt = args
            .get("prompt")
            .and_then(|value| value.as_str())
            .context("task requires a `prompt` string")?;

        if ctx.depth >= ctx.max_depth {
            return Ok(ToolResult::failure(format!(
                "subagent depth limit reached ({}); finish this task directly",
                ctx.max_depth
            )));
        }
        let Some(runner) = &ctx.subagent else {
            return Ok(ToolResult::failure(
                "subagents are not available in this context",
            ));
        };

        let answer = runner.run(description, prompt, ctx.depth + 1).await?;
        Ok(ToolResult::success(answer))
    }
}

//! `question` tool: ask the user a question and wait for the answer.

use std::io::{IsTerminal, Write};

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::registry::{Tool, ToolContext, ToolResult};

pub struct QuestionTool;

#[async_trait]
impl Tool for QuestionTool {
    fn name(&self) -> &'static str {
        "question"
    }

    fn description(&self) -> &'static str {
        "Ask the user a question and wait for their answer. Use when a decision \
         cannot be resolved from the workspace. Provide options when the choice \
         is a small set."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to ask."},
                "header": {"type": "string", "description": "Short heading (optional)."},
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Suggested answers (optional)."
                }
            },
            "required": ["question"]
        })
    }

    fn mutates(&self) -> bool {
        false
    }

    async fn execute(&self, args: &Value, _ctx: &ToolContext) -> Result<ToolResult> {
        let question = args
            .get("question")
            .and_then(|value| value.as_str())
            .context("question requires a `question` string")?;
        let header = args.get("header").and_then(|value| value.as_str());
        let options: Vec<String> = args
            .get("options")
            .and_then(|value| value.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(|text| text.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        if !std::io::stdin().is_terminal() {
            return Ok(ToolResult::failure(
                "no interactive answerer available (stdin is not a terminal); \
                 proceed with your best judgment and state the assumption",
            ));
        }

        let mut prompt = String::new();
        if let Some(header) = header {
            prompt.push_str(&format!("[{header}] "));
        }
        prompt.push_str(question);
        if !options.is_empty() {
            prompt.push('\n');
            for (index, option) in options.iter().enumerate() {
                prompt.push_str(&format!("  {}. {option}\n", index + 1));
            }
        }
        prompt.push_str("answer: ");

        eprint!("{prompt}");
        let _ = std::io::stderr().flush();

        let mut line = String::new();
        std::io::stdin()
            .read_line(&mut line)
            .context("reading answer from stdin")?;
        let answer = line.trim();
        if answer.is_empty() {
            return Ok(ToolResult::failure("the user gave no answer"));
        }

        // Map a numeric choice to the option text.
        if let Ok(index) = answer.parse::<usize>() {
            if index >= 1 && index <= options.len() {
                return Ok(ToolResult::success(options[index - 1].clone()));
            }
        }

        Ok(ToolResult::success(answer.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deeprust_core::AgentMode;
    use std::path::PathBuf;

    #[tokio::test]
    async fn non_interactive_stdin_is_reported() {
        let ctx = ToolContext::new(PathBuf::from("/tmp"), AgentMode::Build);
        // In the test harness stdin is not a terminal.
        let result = QuestionTool
            .execute(&json!({"question": "which branch?"}), &ctx)
            .await
            .unwrap();
        assert!(!result.ok);
        assert!(result.content.contains("no interactive answerer"));
    }
}

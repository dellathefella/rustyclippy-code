//! `todo` tool: session-scoped task list.

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::registry::{TodoItem, Tool, ToolContext, ToolResult};

pub struct TodoTool;

#[async_trait]
impl Tool for TodoTool {
    fn name(&self) -> &'static str {
        "todo"
    }

    fn description(&self) -> &'static str {
        "Record or update the task list. Send the complete list every call. \
         Statuses: pending, in_progress, completed."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}
                        },
                        "required": ["content", "status"]
                    }
                }
            },
            "required": ["todos"]
        })
    }

    fn mutates(&self) -> bool {
        false
    }

    async fn execute(&self, args: &Value, ctx: &ToolContext) -> Result<ToolResult> {
        let items = args
            .get("todos")
            .and_then(|value| value.as_array())
            .context("todo requires a `todos` array")?;

        let mut todos = Vec::with_capacity(items.len());
        for item in items {
            let content = item
                .get("content")
                .and_then(|value| value.as_str())
                .context("todo item requires `content`")?
                .to_string();
            let status = item
                .get("status")
                .and_then(|value| value.as_str())
                .unwrap_or("pending")
                .to_string();
            if !matches!(status.as_str(), "pending" | "in_progress" | "completed") {
                return Ok(ToolResult::failure(format!(
                    "invalid status `{status}` (expected pending, in_progress, completed)"
                )));
            }
            todos.push(TodoItem { content, status });
        }

        {
            let mut stored = ctx.todos.lock().await;
            *stored = todos.clone();
        }

        let mut output = String::new();
        for todo in &todos {
            let marker = match todo.status.as_str() {
                "completed" => "[x]",
                "in_progress" => "[>]",
                _ => "[ ]",
            };
            output.push_str(&format!("{marker} {}\n", todo.content));
        }
        Ok(ToolResult::success(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustyclippy_core::AgentMode;
    use std::path::PathBuf;

    fn context() -> ToolContext {
        ToolContext::new(PathBuf::from("/tmp"), AgentMode::Build)
    }

    #[tokio::test]
    async fn stores_and_formats() {
        let ctx = context();
        let result = TodoTool
            .execute(
                &json!({"todos": [
                    {"content": "read code", "status": "completed"},
                    {"content": "write tests", "status": "in_progress"}
                ]}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(result.ok);
        assert!(result.content.contains("[x] read code"));
        assert!(result.content.contains("[>] write tests"));
        assert_eq!(ctx.todos.lock().await.len(), 2);
    }

    #[tokio::test]
    async fn rejects_bad_status() {
        let result = TodoTool
            .execute(
                &json!({"todos": [{"content": "x", "status": "done"}]}),
                &context(),
            )
            .await
            .unwrap();
        assert!(!result.ok);
    }
}

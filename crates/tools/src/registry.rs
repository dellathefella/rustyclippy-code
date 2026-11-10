//! Tool abstraction and registry.
//!
//! Tool schemas are serialized in a fixed, name-sorted order so the tool block
//! of the prompt is byte-identical every turn. Adding or removing a tool
//! mid-session would invalidate the cache, so the registry is fixed at session
//! start.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use rustyclippy_core::{AgentMode, WebConfig};
use serde_json::{json, Value};

use crate::files::{EditTool, ListTool, ReadTool, WriteTool};
use crate::question::QuestionTool;
use crate::search::{GlobTool, GrepTool};
use crate::shell::{BashTool, LoopTool};
use crate::subagent::{SubagentRunner, TaskTool};
use crate::tasks::{TaskManager, TasksTool, WaitTool};
use crate::todo::TodoTool;
use crate::web::{WebFetchTool, WebSearchTool};

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub ok: bool,
    pub content: String,
}

impl ToolResult {
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            ok: true,
            content: content.into(),
        }
    }

    pub fn failure(content: impl Into<String>) -> Self {
        Self {
            ok: false,
            content: content.into(),
        }
    }
}

/// Execution context shared by all tools.
#[derive(Clone)]
pub struct ToolContext {
    pub cwd: PathBuf,
    pub mode: AgentMode,
    pub max_output_chars: usize,
    pub timeout: Duration,
    pub todos: Arc<tokio::sync::Mutex<Vec<TodoItem>>>,
    pub tasks: Arc<TaskManager>,
    pub web: WebConfig,
    /// Runner used by the `task` tool to spawn a subagent.
    pub subagent: Option<Arc<dyn SubagentRunner>>,
    /// Current subagent depth (0 = top-level agent).
    pub depth: usize,
    pub max_depth: usize,
}

impl ToolContext {
    pub fn new(cwd: PathBuf, mode: AgentMode) -> Self {
        Self {
            cwd,
            mode,
            max_output_chars: 12_000,
            timeout: Duration::from_secs(120),
            todos: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            tasks: Arc::new(TaskManager::new()),
            web: WebConfig::default(),
            subagent: None,
            depth: 0,
            max_depth: 3,
        }
    }

    pub fn resolve(&self, path: &str) -> PathBuf {
        let candidate = std::path::Path::new(path);
        if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            self.cwd.join(candidate)
        }
    }

    pub fn truncate(&self, text: &str) -> String {
        truncate(text, self.max_output_chars)
    }
}

impl std::fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolContext")
            .field("cwd", &self.cwd)
            .field("mode", &self.mode)
            .field("depth", &self.depth)
            .field("max_depth", &self.max_depth)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TodoItem {
    pub content: String,
    pub status: String,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn parameters(&self) -> Value;
    /// Tools that mutate the workspace need permission checks.
    fn mutates(&self) -> bool;
    async fn execute(&self, args: &Value, ctx: &ToolContext) -> Result<ToolResult>;
}

/// Fixed, name-sorted tool set for a mode.
pub struct ToolRegistry {
    tools: Vec<Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn standard(mode: AgentMode) -> Self {
        let mut tools: Vec<Arc<dyn Tool>> = vec![
            Arc::new(BashTool),
            Arc::new(GlobTool),
            Arc::new(GrepTool),
            Arc::new(ListTool),
            Arc::new(LoopTool),
            Arc::new(QuestionTool),
            Arc::new(ReadTool),
            Arc::new(TasksTool),
            Arc::new(TodoTool),
            Arc::new(WaitTool),
            Arc::new(WebFetchTool),
            Arc::new(WebSearchTool),
        ];
        if mode == AgentMode::Build {
            tools.push(Arc::new(EditTool));
            tools.push(Arc::new(WriteTool));
            tools.push(Arc::new(TaskTool));
        }
        tools.sort_by_key(|tool| tool.name());
        Self { tools }
    }

    /// Canonical tool schema block (sorted by name).
    pub fn schemas(&self) -> Vec<Value> {
        self.tools
            .iter()
            .map(|tool| {
                json!({
                    "name": tool.name(),
                    "description": tool.description(),
                    "parameters": tool.parameters(),
                })
            })
            .collect()
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.iter().find(|tool| tool.name() == name).cloned()
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.tools.iter().map(|tool| tool.name()).collect()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

/// Truncate `text` to at most `max` characters.
///
/// The budget is counted in characters, not bytes: `max_output_chars` is named
/// in chars, and comparing bytes would cut CJK output to a third of the
/// configured size (each glyph is three bytes in UTF-8).
pub fn truncate(text: &str, max: usize) -> String {
    let chars = text.chars().count();
    if chars <= max {
        return text.to_string();
    }
    let kept: String = text.chars().take(max).collect();
    let omitted = chars - max;
    format!("{kept}…\n[truncated {omitted} chars]")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_counts_chars_not_bytes() {
        // CJK glyphs are three bytes each, so a byte-based budget would cut
        // this to a third of the requested size.
        let cjk = "日".repeat(12_000);
        let out = truncate(&cjk, 12_000);
        assert_eq!(out, cjk, "text at the budget must pass through unchanged");

        let out = truncate(&cjk, 5_000);
        assert!(out.starts_with(&"日".repeat(5_000)));
        assert!(out.contains("[truncated 7000 chars]"));
        // 5000 kept chars plus the ellipsis and the notice.
        assert_eq!(
            out.chars().count(),
            5_000 + "…\n[truncated 7000 chars]".chars().count()
        );
    }

    #[test]
    fn truncate_is_char_boundary_safe() {
        let text = "日本語テキスト";
        let out = truncate(text, 3);
        assert!(out.starts_with("日本語"));
        assert!(out.contains("[truncated 4 chars]"));
    }

    #[test]
    fn schemas_are_sorted_and_stable() {
        let registry = ToolRegistry::standard(AgentMode::Build);
        let names: Vec<String> = registry
            .schemas()
            .iter()
            .map(|schema| schema["name"].as_str().unwrap().to_string())
            .collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
        assert_eq!(registry.schemas(), registry.schemas());
    }

    #[test]
    fn registry_has_the_full_tool_surface() {
        let registry = ToolRegistry::standard(AgentMode::Build);
        for expected in [
            "bash",
            "edit",
            "glob",
            "grep",
            "list",
            "loop",
            "question",
            "read",
            "task",
            "tasks",
            "todo",
            "wait",
            "webfetch",
            "websearch",
            "write",
        ] {
            assert!(registry.get(expected).is_some(), "missing tool {expected}");
        }
    }

    #[test]
    fn plan_mode_has_no_mutating_tools() {
        let registry = ToolRegistry::standard(AgentMode::Plan);
        assert!(registry.get("write").is_none());
        assert!(registry.get("edit").is_none());
        assert!(registry.get("task").is_none());
        assert!(registry.get("bash").is_some());
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        let text = "日本語テキスト";
        let truncated = truncate(text, 4);
        assert!(truncated.starts_with("日"));
        assert!(truncated.contains("truncated"));
    }
}

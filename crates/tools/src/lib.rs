//! Tool implementations for rustyclippy-code.

pub mod files;
pub mod question;
pub mod registry;
pub mod search;
pub mod shell;
pub mod subagent;
pub mod tasks;
pub mod todo;
pub mod web;

pub use files::{EditTool, ListTool, ReadTool, WriteTool};
pub use question::QuestionTool;
pub use registry::{truncate, TodoItem, Tool, ToolContext, ToolRegistry, ToolResult};
pub use search::{GlobTool, GrepTool};
pub use shell::{run_shell, BashTool, LoopTool};
pub use subagent::{SubagentRunner, TaskTool};
pub use tasks::{spawn_background, BackgroundTask, TaskManager, TaskStatus, TasksTool, WaitTool};
pub use todo::TodoTool;
pub use web::{WebFetchTool, WebSearchTool};

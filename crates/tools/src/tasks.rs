//! Background task management: spawn, list, and wait for completion.
//!
//! Background tasks let the agent start a long-running process (dev server,
//! build, test suite) and keep working. `wait` blocks until the task settles;
//! `tasks` lists everything currently tracked.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::registry::{truncate, Tool, ToolContext, ToolResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    Running,
    Completed,
    Failed,
    TimedOut,
}

impl TaskStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            TaskStatus::Running => "running",
            TaskStatus::Completed => "completed",
            TaskStatus::Failed => "failed",
            TaskStatus::TimedOut => "timed_out",
        }
    }

    pub fn is_terminal(self) -> bool {
        self != TaskStatus::Running
    }
}

#[derive(Debug)]
pub struct BackgroundTask {
    pub id: String,
    pub command: String,
    pub status: TaskStatus,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub started_at: Instant,
    pub finished_at: Option<Instant>,
}

impl BackgroundTask {
    pub fn new(id: String, command: String) -> Self {
        Self {
            id,
            command,
            status: TaskStatus::Running,
            stdout: String::new(),
            stderr: String::new(),
            exit_code: None,
            started_at: Instant::now(),
            finished_at: None,
        }
    }

    pub fn elapsed_secs(&self) -> f64 {
        match self.finished_at {
            Some(finished) => finished.duration_since(self.started_at).as_secs_f64(),
            None => self.started_at.elapsed().as_secs_f64(),
        }
    }

    pub fn render(&self) -> String {
        let mut out = format!(
            "task {} [{}] {:.1}s{} {}\n",
            self.id,
            self.status.as_str(),
            self.elapsed_secs(),
            self.exit_code
                .map(|code| format!(" exit={code}"))
                .unwrap_or_default(),
            truncate(&self.command, 120)
        );
        if !self.stdout.trim().is_empty() {
            out.push_str("--- stdout ---\n");
            out.push_str(&truncate(&self.stdout, 4_000));
            out.push('\n');
        }
        if !self.stderr.trim().is_empty() {
            out.push_str("--- stderr ---\n");
            out.push_str(&truncate(&self.stderr, 4_000));
            out.push('\n');
        }
        out
    }
}

#[derive(Clone, Default)]
pub struct TaskManager {
    tasks: Arc<Mutex<HashMap<String, Arc<Mutex<BackgroundTask>>>>>,
    counter: Arc<AtomicUsize>,
}

impl TaskManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn next_id(&self) -> String {
        let number = self.counter.fetch_add(1, Ordering::SeqCst) + 1;
        format!("task-{number}")
    }

    pub async fn insert(&self, task: Arc<Mutex<BackgroundTask>>) {
        let id = task.lock().await.id.clone();
        self.tasks.lock().await.insert(id, task);
    }

    pub async fn get(&self, id: &str) -> Option<Arc<Mutex<BackgroundTask>>> {
        self.tasks.lock().await.get(id).cloned()
    }

    pub async fn list(&self) -> Vec<Arc<Mutex<BackgroundTask>>> {
        let mut tasks: Vec<_> = self.tasks.lock().await.values().cloned().collect();
        tasks.sort_by(|a, b| {
            // Sort by numeric id suffix so task-2 comes after task-1.
            let a_id = a.try_lock().map(|t| t.id.clone()).unwrap_or_default();
            let b_id = b.try_lock().map(|t| t.id.clone()).unwrap_or_default();
            a_id.cmp(&b_id)
        });
        tasks
    }
}

/// Spawn a shell command in the background and register it with the manager.
pub async fn spawn_background(
    manager: &TaskManager,
    command: String,
    workdir: PathBuf,
    max_output_chars: usize,
) -> String {
    let id = manager.next_id();
    let record = Arc::new(Mutex::new(BackgroundTask::new(id.clone(), command.clone())));
    manager.insert(record.clone()).await;

    tokio::spawn(async move {
        let spawned = Command::new("sh")
            .arg("-c")
            .arg(&command)
            .current_dir(&workdir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();

        let mut child = match spawned {
            Ok(child) => child,
            Err(error) => {
                let mut task = record.lock().await;
                task.status = TaskStatus::Failed;
                task.stderr = format!("failed to spawn command: {error}");
                task.finished_at = Some(Instant::now());
                return;
            }
        };

        let stdout_pipe = child.stdout.take();
        let stderr_pipe = child.stderr.take();
        let stdout_task = stdout_pipe.map(|pipe| {
            let record = record.clone();
            tokio::spawn(pump(record, pipe, true, max_output_chars))
        });
        let stderr_task = stderr_pipe.map(|pipe| {
            let record = record.clone();
            tokio::spawn(pump(record, pipe, false, max_output_chars))
        });

        let status = child.wait().await;
        if let Some(task) = stdout_task {
            let _ = task.await;
        }
        if let Some(task) = stderr_task {
            let _ = task.await;
        }

        let mut task = record.lock().await;
        match status {
            Ok(status) => {
                task.exit_code = status.code();
                task.status = if status.success() {
                    TaskStatus::Completed
                } else {
                    TaskStatus::Failed
                };
            }
            Err(error) => {
                task.status = TaskStatus::Failed;
                task.stderr.push_str(&format!("\nwait error: {error}"));
            }
        }
        task.finished_at = Some(Instant::now());
    });

    id
}

async fn pump<R: AsyncRead + Unpin>(
    record: Arc<Mutex<BackgroundTask>>,
    mut reader: R,
    stdout: bool,
    max_output_chars: usize,
) {
    let mut buffer = [0u8; 4096];
    loop {
        match reader.read(&mut buffer).await {
            Ok(0) => break,
            Ok(read) => {
                let chunk = String::from_utf8_lossy(&buffer[..read]).into_owned();
                let mut task = record.lock().await;
                let target = if stdout {
                    &mut task.stdout
                } else {
                    &mut task.stderr
                };
                append_capped(target, &chunk, max_output_chars);
            }
            Err(_) => break,
        }
    }
}

fn append_capped(target: &mut String, chunk: &str, max: usize) {
    if target.len() >= max {
        return;
    }
    let room = max - target.len();
    if chunk.len() <= room {
        target.push_str(chunk);
    } else {
        let mut end = room;
        while end > 0 && !chunk.is_char_boundary(end) {
            end -= 1;
        }
        target.push_str(&chunk[..end]);
    }
}

pub struct TasksTool;

#[async_trait]
impl Tool for TasksTool {
    fn name(&self) -> &'static str {
        "tasks"
    }

    fn description(&self) -> &'static str {
        "List background tasks with status, elapsed time, exit code and output."
    }

    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {}, "required": []})
    }

    fn mutates(&self) -> bool {
        false
    }

    async fn execute(&self, _args: &Value, ctx: &ToolContext) -> Result<ToolResult> {
        let tasks = ctx.tasks.list().await;
        if tasks.is_empty() {
            return Ok(ToolResult::success("no background tasks"));
        }
        let mut out = String::new();
        for task in tasks {
            let task = task.lock().await;
            out.push_str(&task.render());
        }
        Ok(ToolResult::success(ctx.truncate(&out)))
    }
}

pub struct WaitTool;

#[async_trait]
impl Tool for WaitTool {
    fn name(&self) -> &'static str {
        "wait"
    }

    fn description(&self) -> &'static str {
        "Wait for a background task to finish and return its output. \
         Returns immediately if the task already finished."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task id from `bash` with background=true."},
                "timeout_ms": {"type": "integer", "description": "Maximum wait in milliseconds (default 300000)."}
            },
            "required": ["task_id"]
        })
    }

    fn mutates(&self) -> bool {
        false
    }

    async fn execute(&self, args: &Value, ctx: &ToolContext) -> Result<ToolResult> {
        let task_id = args
            .get("task_id")
            .and_then(|value| value.as_str())
            .context("wait requires a `task_id` string")?;
        let timeout_ms = args
            .get("timeout_ms")
            .and_then(|value| value.as_u64())
            .unwrap_or(300_000);
        let deadline = Instant::now() + Duration::from_millis(timeout_ms.max(1));

        let Some(task) = ctx.tasks.get(task_id).await else {
            return Ok(ToolResult::failure(format!("unknown task `{task_id}`")));
        };

        loop {
            let snapshot = task.lock().await;
            if snapshot.status.is_terminal() {
                let rendered = snapshot.render();
                let ok = snapshot.status == TaskStatus::Completed;
                return Ok(ToolResult {
                    ok,
                    content: ctx.truncate(&rendered),
                });
            }
            drop(snapshot);
            if Instant::now() >= deadline {
                let snapshot = task.lock().await;
                return Ok(ToolResult::failure(format!(
                    "task `{task_id}` still running after {timeout_ms}ms\n{}",
                    ctx.truncate(&snapshot.render())
                )));
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustyclippy_core::AgentMode;

    fn context() -> ToolContext {
        ToolContext::new(PathBuf::from("/tmp"), AgentMode::Build)
    }

    #[tokio::test]
    async fn background_task_completes_and_wait_returns_output() {
        let ctx = context();
        let id = spawn_background(
            &ctx.tasks,
            "echo background-hello".to_string(),
            PathBuf::from("/tmp"),
            4_000,
        )
        .await;
        let result = WaitTool
            .execute(&json!({"task_id": id, "timeout_ms": 10_000}), &ctx)
            .await
            .unwrap();
        assert!(result.ok);
        assert!(result.content.contains("background-hello"));
    }

    #[tokio::test]
    async fn tasks_tool_lists_running_work() {
        let ctx = context();
        let _id = spawn_background(
            &ctx.tasks,
            "sleep 2".to_string(),
            PathBuf::from("/tmp"),
            4_000,
        )
        .await;
        let result = TasksTool.execute(&json!({}), &ctx).await.unwrap();
        assert!(result.content.contains("running"));
    }

    #[tokio::test]
    async fn wait_rejects_unknown_task() {
        let result = WaitTool
            .execute(&json!({"task_id": "task-99"}), &context())
            .await
            .unwrap();
        assert!(!result.ok);
    }
}

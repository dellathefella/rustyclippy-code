//! Shell tools: `bash` (foreground or background) and `loop` (poll until a
//! condition matches).

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use regex::Regex;
use serde_json::{json, Value};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;

use crate::registry::{truncate, Tool, ToolContext, ToolResult};
use crate::tasks::spawn_background;

pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &'static str {
        "bash"
    }

    fn description(&self) -> &'static str {
        "Run a shell command. Returns exit code, stdout and stderr. \
         Set background=true for long-running processes; then use `wait` or `tasks`."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute via `sh -c`."
                },
                "workdir": {
                    "type": "string",
                    "description": "Directory to run in (defaults to the workspace root)."
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default 120000)."
                },
                "background": {
                    "type": "boolean",
                    "description": "Run in the background and return a task id."
                }
            },
            "required": ["command"]
        })
    }

    fn mutates(&self) -> bool {
        true
    }

    async fn execute(&self, args: &Value, ctx: &ToolContext) -> Result<ToolResult> {
        let command = args
            .get("command")
            .and_then(|value| value.as_str())
            .context("bash requires a `command` string")?;

        let workdir = args
            .get("workdir")
            .and_then(|value| value.as_str())
            .map(|path| ctx.resolve(path))
            .unwrap_or_else(|| ctx.cwd.clone());

        let timeout_ms = args
            .get("timeout_ms")
            .and_then(|value| value.as_u64())
            .unwrap_or(ctx.timeout.as_millis() as u64);
        let timeout = Duration::from_millis(timeout_ms.max(1));

        let background = args
            .get("background")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);

        if background {
            let id = spawn_background(
                &ctx.tasks,
                command.to_string(),
                workdir,
                ctx.max_output_chars,
            )
            .await;
            return Ok(ToolResult::success(format!(
                "started background task `{id}`; use `wait` with task_id={id} to collect output, \
                 or `tasks` to list all background work"
            )));
        }

        let (ok, content) = run_shell(command, &workdir, timeout, ctx.max_output_chars).await?;
        Ok(ToolResult { ok, content })
    }
}

pub struct LoopTool;

#[async_trait]
impl Tool for LoopTool {
    fn name(&self) -> &'static str {
        "loop"
    }

    fn description(&self) -> &'static str {
        "Run a command repeatedly until its output matches a regex, or until \
         max_iterations/timeout is reached. Use to wait for a server, a build, \
         or a file to appear."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to run each iteration."},
                "until": {"type": "string", "description": "Regex that must match the command output."},
                "interval_ms": {"type": "integer", "description": "Delay between iterations (default 1000)."},
                "max_iterations": {"type": "integer", "description": "Maximum iterations (default 60)."},
                "timeout_ms": {"type": "integer", "description": "Per-iteration command timeout (default 60000)."}
            },
            "required": ["command", "until"]
        })
    }

    fn mutates(&self) -> bool {
        true
    }

    async fn execute(&self, args: &Value, ctx: &ToolContext) -> Result<ToolResult> {
        let command = args
            .get("command")
            .and_then(|value| value.as_str())
            .context("loop requires a `command` string")?;
        let until = args
            .get("until")
            .and_then(|value| value.as_str())
            .context("loop requires an `until` regex")?;
        let interval_ms = args
            .get("interval_ms")
            .and_then(|value| value.as_u64())
            .unwrap_or(1_000);
        let max_iterations = args
            .get("max_iterations")
            .and_then(|value| value.as_u64())
            .unwrap_or(60)
            .max(1) as usize;
        let timeout_ms = args
            .get("timeout_ms")
            .and_then(|value| value.as_u64())
            .unwrap_or(60_000);

        let regex =
            Regex::new(until).with_context(|| format!("invalid `until` regex `{until}`"))?;
        let timeout = Duration::from_millis(timeout_ms.max(1));

        let mut last_output = String::new();
        for iteration in 1..=max_iterations {
            let (ok, content) = run_shell(command, &ctx.cwd, timeout, ctx.max_output_chars).await?;
            if regex.is_match(&content) {
                return Ok(ToolResult::success(format!(
                    "condition matched on iteration {iteration}\n{content}"
                )));
            }
            last_output = content;
            if !ok && iteration == max_iterations {
                break;
            }
            if iteration < max_iterations {
                tokio::time::sleep(Duration::from_millis(interval_ms)).await;
            }
        }

        Ok(ToolResult::failure(format!(
            "condition `{until}` did not match after {max_iterations} iterations\n--- last output ---\n{}",
            truncate(&last_output, 4_000)
        )))
    }
}

/// Run a shell command to completion with a timeout.
pub async fn run_shell(
    command: &str,
    workdir: &Path,
    timeout: Duration,
    max_chars: usize,
) -> Result<(bool, String)> {
    let mut command_builder = Command::new("sh");
    command_builder
        .arg("-c")
        .arg(command)
        .current_dir(workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // Put the shell in its own process group so a timeout can kill the whole
    // tree. Killing only `sh` leaves grandchildren (a `cargo build`, a dev
    // server) running forever.
    #[cfg(unix)]
    command_builder.process_group(0);

    let mut child = command_builder
        .spawn()
        .with_context(|| format!("spawning command in {}", workdir.display()))?;

    let stdout_pipe = child.stdout.take().expect("stdout pipe");
    let stderr_pipe = child.stderr.take().expect("stderr pipe");
    let stdout_task = tokio::spawn(read_all(stdout_pipe));
    let stderr_task = tokio::spawn(read_all(stderr_pipe));

    let status = match tokio::time::timeout(timeout, child.wait()).await {
        Ok(result) => result.context("waiting for command")?,
        Err(_) => {
            kill_tree(&mut child).await;
            let _ = child.wait().await;
            let stdout = stdout_task.await.unwrap_or_default();
            let stderr = stderr_task.await.unwrap_or_default();
            let content = format!(
                "command timed out after {}ms\n--- stdout ---\n{}\n--- stderr ---\n{}",
                timeout.as_millis(),
                truncate(&stdout, max_chars),
                truncate(&stderr, max_chars)
            );
            return Ok((false, content));
        }
    };

    let stdout = stdout_task.await.unwrap_or_default();
    let stderr = stderr_task.await.unwrap_or_default();

    let code = status.code().unwrap_or(-1);
    let stdout_display = if stdout.trim().is_empty() {
        "(empty)".to_string()
    } else {
        truncate(&stdout, max_chars)
    };
    let stderr_display = if stderr.trim().is_empty() {
        "(empty)".to_string()
    } else {
        truncate(&stderr, max_chars)
    };

    let content = format!(
        "exit_code: {code}\n--- stdout ---\n{stdout_display}\n--- stderr ---\n{stderr_display}"
    );
    Ok((status.success(), content))
}

/// Kill a spawned shell and every process in its group.
///
/// `child.kill()` signals only the shell; a command like `cargo build` keeps
/// running as an orphan. On Unix the shell was spawned with `process_group(0)`,
/// so its pid is also the group id and `killpg` reaches the whole tree.
async fn kill_tree(child: &mut tokio::process::Child) {
    #[cfg(unix)]
    {
        if let Some(pid) = child.id() {
            // SIGKILL the group; ignore ESRCH when the tree already exited.
            unsafe {
                libc::killpg(pid as libc::pid_t, libc::SIGKILL);
            }
        }
    }
    let _ = child.kill().await;
}

async fn read_all<R: AsyncRead + Unpin>(mut reader: R) -> String {
    let mut buffer = Vec::new();
    let _ = reader.read_to_end(&mut buffer).await;
    String::from_utf8_lossy(&buffer).into_owned()
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
    async fn runs_and_reports_exit_code() {
        let result = BashTool
            .execute(&json!({"command": "echo hello"}), &context())
            .await
            .unwrap();
        assert!(result.ok);
        assert!(result.content.contains("hello"));
        assert!(result.content.contains("exit_code: 0"));
    }

    #[tokio::test]
    async fn failure_is_reported_not_thrown() {
        let result = BashTool
            .execute(&json!({"command": "exit 3"}), &context())
            .await
            .unwrap();
        assert!(!result.ok);
        assert!(result.content.contains("exit_code: 3"));
    }

    #[tokio::test]
    async fn timeout_kills_the_command() {
        let result = BashTool
            .execute(
                &json!({"command": "sleep 5", "timeout_ms": 100}),
                &context(),
            )
            .await
            .unwrap();
        assert!(!result.ok);
        assert!(result.content.contains("timed out"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_kills_the_whole_process_tree() {
        // The shell backgrounds a grandchild and waits. Killing only `sh`
        // would leave the grandchild running; the group kill must reach it.
        let dir = tempfile::TempDir::new().unwrap();
        let pid_file = dir.path().join("child.pid");
        let command = format!("sleep 30 & echo $! > {}; wait", pid_file.display());

        let started = std::time::Instant::now();
        let result = BashTool
            .execute(&json!({"command": command, "timeout_ms": 200}), &context())
            .await
            .unwrap();
        let elapsed = started.elapsed();
        assert!(!result.ok);
        assert!(result.content.contains("timed out"));

        // Draining the pipes blocks until every writer exits. If the grandchild
        // survived, this call would hang for the full `sleep 30`.
        assert!(
            elapsed < Duration::from_secs(5),
            "timeout took {elapsed:?}; the grandchild kept the pipes open"
        );

        let pid: i32 = std::fs::read_to_string(&pid_file)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let alive = unsafe { libc::kill(pid, 0) } == 0;
        assert!(!alive, "grandchild {pid} survived the timeout");
    }

    #[tokio::test]
    async fn background_returns_task_id() {
        let result = BashTool
            .execute(
                &json!({"command": "echo bg", "background": true}),
                &context(),
            )
            .await
            .unwrap();
        assert!(result.ok);
        assert!(result.content.contains("task-"));
    }

    #[tokio::test]
    async fn loop_stops_when_condition_matches() {
        let result = LoopTool
            .execute(
                &json!({"command": "echo ready", "until": "ready", "max_iterations": 3}),
                &context(),
            )
            .await
            .unwrap();
        assert!(result.ok);
        assert!(result.content.contains("iteration 1"));
    }
}

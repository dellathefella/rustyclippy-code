//! File tools: `read`, `write`, `edit`, `list`.

use std::path::Path;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::registry::{Tool, ToolContext, ToolResult};

pub struct ReadTool;

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &'static str {
        "read"
    }

    fn description(&self) -> &'static str {
        "Read a file with line numbers. Use offset/limit for large files."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path (relative to workspace or absolute)."},
                "offset": {"type": "integer", "description": "1-based first line to read (default 1)."},
                "limit": {"type": "integer", "description": "Maximum lines to read (default 2000)."}
            },
            "required": ["path"]
        })
    }

    fn mutates(&self) -> bool {
        false
    }

    async fn execute(&self, args: &Value, ctx: &ToolContext) -> Result<ToolResult> {
        let path = args
            .get("path")
            .and_then(|value| value.as_str())
            .context("read requires a `path` string")?;
        let resolved = ctx.resolve(path);
        if !resolved.exists() {
            return Ok(ToolResult::failure(format!(
                "file not found: {}",
                resolved.display()
            )));
        }
        if resolved.is_dir() {
            return Ok(ToolResult::failure(format!(
                "{} is a directory; use `list`",
                resolved.display()
            )));
        }

        let text = std::fs::read_to_string(&resolved)
            .with_context(|| format!("reading {}", resolved.display()))?;

        let offset = args
            .get("offset")
            .and_then(|value| value.as_u64())
            .unwrap_or(1)
            .max(1) as usize;
        let limit = args
            .get("limit")
            .and_then(|value| value.as_u64())
            .unwrap_or(2_000) as usize;

        let lines: Vec<&str> = text.lines().collect();
        let start = offset - 1;
        if start >= lines.len() {
            return Ok(ToolResult::failure(format!(
                "offset {offset} is past the end of {} ({} lines)",
                resolved.display(),
                lines.len()
            )));
        }
        let end = (start + limit).min(lines.len());

        let mut output = String::new();
        for (index, line) in lines[start..end].iter().enumerate() {
            output.push_str(&format!("{:>6}\t{}\n", start + index + 1, line));
        }
        if end < lines.len() {
            output.push_str(&format!(
                "\n[showing lines {}-{} of {}; use offset to continue]",
                start + 1,
                end,
                lines.len()
            ));
        }

        Ok(ToolResult::success(ctx.truncate(&output)))
    }
}

pub struct WriteTool;

#[async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &'static str {
        "write"
    }

    fn description(&self) -> &'static str {
        "Create or replace a file with the given content. Parent directories are created."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path."},
                "content": {"type": "string", "description": "Full file content."}
            },
            "required": ["path", "content"]
        })
    }

    fn mutates(&self) -> bool {
        true
    }

    async fn execute(&self, args: &Value, ctx: &ToolContext) -> Result<ToolResult> {
        let path = args
            .get("path")
            .and_then(|value| value.as_str())
            .context("write requires a `path` string")?;
        let content = args
            .get("content")
            .and_then(|value| value.as_str())
            .context("write requires a `content` string")?;
        let resolved = ctx.resolve(path);

        if let Some(parent) = resolved.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        std::fs::write(&resolved, content)
            .with_context(|| format!("writing {}", resolved.display()))?;

        Ok(ToolResult::success(format!(
            "wrote {} bytes to {}",
            content.len(),
            resolved.display()
        )))
    }
}

pub struct EditTool;

#[async_trait]
impl Tool for EditTool {
    fn name(&self) -> &'static str {
        "edit"
    }

    fn description(&self) -> &'static str {
        "Replace an exact `old_string` with `new_string` in a file. \
         By default the match must be unique; set replace_all to replace every occurrence."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path."},
                "old_string": {"type": "string", "description": "Exact text to replace."},
                "new_string": {"type": "string", "description": "Replacement text."},
                "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)."}
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    fn mutates(&self) -> bool {
        true
    }

    async fn execute(&self, args: &Value, ctx: &ToolContext) -> Result<ToolResult> {
        let path = args
            .get("path")
            .and_then(|value| value.as_str())
            .context("edit requires a `path` string")?;
        let old_string = args
            .get("old_string")
            .and_then(|value| value.as_str())
            .context("edit requires an `old_string` string")?;
        let new_string = args
            .get("new_string")
            .and_then(|value| value.as_str())
            .context("edit requires a `new_string` string")?;
        let replace_all = args
            .get("replace_all")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);

        if old_string == new_string {
            return Ok(ToolResult::failure(
                "old_string and new_string are identical",
            ));
        }
        if old_string.is_empty() {
            return Ok(ToolResult::failure("old_string must not be empty"));
        }

        let resolved = ctx.resolve(path);
        let text = std::fs::read_to_string(&resolved)
            .with_context(|| format!("reading {}", resolved.display()))?;

        let occurrences = text.matches(old_string).count();
        if occurrences == 0 {
            return Ok(ToolResult::failure(format!(
                "old_string not found in {}",
                resolved.display()
            )));
        }
        if occurrences > 1 && !replace_all {
            return Ok(ToolResult::failure(format!(
                "old_string matches {occurrences} times in {}; provide more context or set replace_all",
                resolved.display()
            )));
        }

        let updated = if replace_all {
            text.replace(old_string, new_string)
        } else {
            text.replacen(old_string, new_string, 1)
        };

        std::fs::write(&resolved, updated)
            .with_context(|| format!("writing {}", resolved.display()))?;

        Ok(ToolResult::success(format!(
            "replaced {occurrences} occurrence(s) in {}",
            resolved.display()
        )))
    }
}

pub struct ListTool;

#[async_trait]
impl Tool for ListTool {
    fn name(&self) -> &'static str {
        "list"
    }

    fn description(&self) -> &'static str {
        "List directory entries (directories first, then files, sorted by name)."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (default: workspace root)."}
            },
            "required": []
        })
    }

    fn mutates(&self) -> bool {
        false
    }

    async fn execute(&self, args: &Value, ctx: &ToolContext) -> Result<ToolResult> {
        let path = args
            .get("path")
            .and_then(|value| value.as_str())
            .unwrap_or(".");
        let resolved = ctx.resolve(path);
        if !resolved.exists() {
            return Ok(ToolResult::failure(format!(
                "directory not found: {}",
                resolved.display()
            )));
        }
        if !resolved.is_dir() {
            return Ok(ToolResult::failure(format!(
                "{} is not a directory",
                resolved.display()
            )));
        }

        let mut dirs: Vec<String> = Vec::new();
        let mut files: Vec<String> = Vec::new();
        for entry in std::fs::read_dir(&resolved)
            .with_context(|| format!("listing {}", resolved.display()))?
        {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                dirs.push(format!("{name}/"));
            } else if file_type.is_symlink() {
                files.push(format!("{name}@"));
            } else {
                files.push(name);
            }
        }
        dirs.sort();
        files.sort();

        let mut output = String::new();
        for dir in &dirs {
            output.push_str(dir);
            output.push('\n');
        }
        for file in &files {
            output.push_str(file);
            output.push('\n');
        }
        if output.is_empty() {
            output.push_str("(empty directory)\n");
        }

        Ok(ToolResult::success(ctx.truncate(&output)))
    }
}

#[allow(dead_code)]
fn exists(path: &Path) -> bool {
    path.exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustyclippy_core::AgentMode;
    use tempfile::TempDir;

    fn context(dir: &TempDir) -> ToolContext {
        ToolContext::new(dir.path().to_path_buf(), AgentMode::Build)
    }

    #[tokio::test]
    async fn write_then_read() {
        let dir = TempDir::new().unwrap();
        let ctx = context(&dir);
        let write = WriteTool
            .execute(&json!({"path": "a.txt", "content": "one\ntwo\n"}), &ctx)
            .await
            .unwrap();
        assert!(write.ok);
        let read = ReadTool
            .execute(&json!({"path": "a.txt"}), &ctx)
            .await
            .unwrap();
        assert!(read.ok);
        assert!(read.content.contains("1\tone"));
        assert!(read.content.contains("2\ttwo"));
    }

    #[tokio::test]
    async fn edit_requires_unique_match() {
        let dir = TempDir::new().unwrap();
        let ctx = context(&dir);
        WriteTool
            .execute(&json!({"path": "a.txt", "content": "x\nx\n"}), &ctx)
            .await
            .unwrap();
        let result = EditTool
            .execute(
                &json!({"path": "a.txt", "old_string": "x", "new_string": "y"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!result.ok);
        assert!(result.content.contains("matches 2 times"));
    }

    #[tokio::test]
    async fn edit_replace_all() {
        let dir = TempDir::new().unwrap();
        let ctx = context(&dir);
        WriteTool
            .execute(&json!({"path": "a.txt", "content": "x\nx\n"}), &ctx)
            .await
            .unwrap();
        let result = EditTool
            .execute(
                &json!({"path": "a.txt", "old_string": "x", "new_string": "y", "replace_all": true}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(result.ok);
        let text = std::fs::read_to_string(dir.path().join("a.txt")).unwrap();
        assert_eq!(text, "y\ny\n");
    }

    #[tokio::test]
    async fn list_sorts_dirs_first() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("b.txt"), "").unwrap();
        let ctx = context(&dir);
        let result = ListTool.execute(&json!({}), &ctx).await.unwrap();
        assert!(result.content.starts_with("sub/\n"));
        assert!(result.content.contains("b.txt"));
    }
}

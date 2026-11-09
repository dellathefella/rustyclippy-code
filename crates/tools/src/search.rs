//! Search tools: `glob` and `grep`.
//!
//! Both respect `.gitignore` via the `ignore` crate and skip hidden files, so
//! results match what a developer expects from ripgrep.

use std::path::PathBuf;

use anyhow::{Context, Result};
use async_trait::async_trait;
use globset::{Glob, GlobMatcher};
use ignore::WalkBuilder;
use regex::Regex;
use serde_json::{json, Value};

use crate::registry::{Tool, ToolContext, ToolResult};

const MAX_RESULTS: usize = 200;
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;

pub struct GlobTool;

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &'static str {
        "glob"
    }

    fn description(&self) -> &'static str {
        "Find files by glob pattern (e.g. `**/*.rs`, `src/**/*.ts`). Respects .gitignore."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern."},
                "path": {"type": "string", "description": "Root directory (default: workspace root)."}
            },
            "required": ["pattern"]
        })
    }

    fn mutates(&self) -> bool {
        false
    }

    async fn execute(&self, args: &Value, ctx: &ToolContext) -> Result<ToolResult> {
        let pattern = args
            .get("pattern")
            .and_then(|value| value.as_str())
            .context("glob requires a `pattern` string")?;
        let root = args
            .get("path")
            .and_then(|value| value.as_str())
            .map(|path| ctx.resolve(path))
            .unwrap_or_else(|| ctx.cwd.clone());

        let matcher = Glob::new(pattern)
            .with_context(|| format!("invalid glob pattern `{pattern}`"))?
            .compile_matcher();

        let mut results = Vec::new();
        for entry in walk(&root) {
            if !entry
                .file_type()
                .map(|kind| kind.is_file())
                .unwrap_or(false)
            {
                continue;
            }
            let relative = entry.path().strip_prefix(&root).unwrap_or(entry.path());
            if matcher.is_match(relative) {
                results.push(relative.display().to_string());
                if results.len() >= MAX_RESULTS {
                    break;
                }
            }
        }
        results.sort();

        if results.is_empty() {
            return Ok(ToolResult::success(format!(
                "no files matched `{pattern}` under {}",
                root.display()
            )));
        }

        let mut output = results.join("\n");
        if results.len() >= MAX_RESULTS {
            output.push_str(&format!("\n[showing first {MAX_RESULTS} matches]"));
        }
        Ok(ToolResult::success(ctx.truncate(&output)))
    }
}

pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &'static str {
        "grep"
    }

    fn description(&self) -> &'static str {
        "Search file contents with a regex. Returns `path:line:content` matches. Respects .gitignore."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern."},
                "path": {"type": "string", "description": "Root directory or file (default: workspace root)."},
                "include": {"type": "string", "description": "Glob limiting which files are searched (e.g. `*.rs`)."}
            },
            "required": ["pattern"]
        })
    }

    fn mutates(&self) -> bool {
        false
    }

    async fn execute(&self, args: &Value, ctx: &ToolContext) -> Result<ToolResult> {
        let pattern = args
            .get("pattern")
            .and_then(|value| value.as_str())
            .context("grep requires a `pattern` string")?;
        let regex = Regex::new(pattern).with_context(|| format!("invalid regex `{pattern}`"))?;

        let root = args
            .get("path")
            .and_then(|value| value.as_str())
            .map(|path| ctx.resolve(path))
            .unwrap_or_else(|| ctx.cwd.clone());

        let include: Option<GlobMatcher> =
            match args.get("include").and_then(|value| value.as_str()) {
                Some(glob) => Some(
                    Glob::new(glob)
                        .with_context(|| format!("invalid include glob `{glob}`"))?
                        .compile_matcher(),
                ),
                None => None,
            };

        let mut matches = Vec::new();
        let mut files_searched = 0usize;

        if root.is_file() {
            search_file(&root, &root, &regex, &mut matches);
            files_searched = 1;
        } else {
            for entry in walk(&root) {
                if !entry
                    .file_type()
                    .map(|kind| kind.is_file())
                    .unwrap_or(false)
                {
                    continue;
                }
                let path = entry.path();
                let relative = path.strip_prefix(&root).unwrap_or(path);
                if let Some(matcher) = &include {
                    if !matcher.is_match(relative) {
                        continue;
                    }
                }
                files_searched += 1;
                search_file(path, relative, &regex, &mut matches);
                if matches.len() >= MAX_RESULTS {
                    break;
                }
            }
        }

        if matches.is_empty() {
            return Ok(ToolResult::success(format!(
                "no matches for `{pattern}` (searched {files_searched} files under {})",
                root.display()
            )));
        }

        let mut output = matches.join("\n");
        if matches.len() >= MAX_RESULTS {
            output.push_str(&format!("\n[showing first {MAX_RESULTS} matches]"));
        }
        Ok(ToolResult::success(ctx.truncate(&output)))
    }
}

fn walk(root: &PathBuf) -> impl Iterator<Item = ignore::DirEntry> {
    WalkBuilder::new(root)
        .hidden(true)
        .git_ignore(true)
        .git_exclude(true)
        .parents(true)
        .build()
        .filter_map(|entry| entry.ok())
}

fn search_file(
    path: &std::path::Path,
    relative: &std::path::Path,
    regex: &Regex,
    matches: &mut Vec<String>,
) {
    let metadata = match std::fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(_) => return,
    };
    if metadata.len() > MAX_FILE_BYTES {
        return;
    }
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(_) => return,
    };
    // Skip binary files (NUL byte in the first 1 KiB).
    if bytes.iter().take(1024).any(|byte| *byte == 0) {
        return;
    }
    let text = String::from_utf8_lossy(&bytes);
    for (line_number, line) in text.lines().enumerate() {
        if regex.is_match(line) {
            matches.push(format!(
                "{}:{}:{}",
                relative.display(),
                line_number + 1,
                line.trim_end()
            ));
            if matches.len() >= MAX_RESULTS {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deeprust_core::AgentMode;
    use tempfile::TempDir;

    fn context(dir: &TempDir) -> ToolContext {
        ToolContext::new(dir.path().to_path_buf(), AgentMode::Build)
    }

    fn seed(dir: &TempDir) {
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/a.rs"), "fn alpha() {}\nfn beta() {}\n").unwrap();
        std::fs::write(dir.path().join("src/b.ts"), "const alpha = 1;\n").unwrap();
        std::fs::write(dir.path().join("README.md"), "alpha\n").unwrap();
    }

    #[tokio::test]
    async fn glob_finds_matching_files() {
        let dir = TempDir::new().unwrap();
        seed(&dir);
        let result = GlobTool
            .execute(&json!({"pattern": "**/*.rs"}), &context(&dir))
            .await
            .unwrap();
        assert!(result.content.contains("src/a.rs"));
        assert!(!result.content.contains("src/b.ts"));
    }

    #[tokio::test]
    async fn grep_reports_line_numbers() {
        let dir = TempDir::new().unwrap();
        seed(&dir);
        let result = GrepTool
            .execute(
                &json!({"pattern": "alpha", "include": "*.rs"}),
                &context(&dir),
            )
            .await
            .unwrap();
        assert!(result.content.contains("src/a.rs:1:fn alpha() {}"));
        assert!(!result.content.contains("b.ts"));
    }

    #[tokio::test]
    async fn invalid_regex_is_an_error() {
        let dir = TempDir::new().unwrap();
        let result = GrepTool
            .execute(&json!({"pattern": "("}), &context(&dir))
            .await;
        assert!(result.is_err());
    }
}

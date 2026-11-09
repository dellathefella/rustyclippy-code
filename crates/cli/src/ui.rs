//! Terminal output: an opencode-inspired REPL surface.
//!
//! Chrome (header, rules, tool activity, metrics) goes to stderr so headless
//! `run` output on stdout stays clean. In chat mode the assistant stream is
//! rendered under a dim rail so the prompt and the answer never blur together.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};

use rustyclippy_core::{CompactionRecord, RequestPlan};
use rustyclippy_llm::Usage;
use rustyclippy_tools::ToolResult;

const RESET: &str = "\x1b[0m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const BOLD: &str = "\x1b[1m";

#[derive(Debug)]
pub struct Output {
    pub json: bool,
    pub quiet: bool,
    color: bool,
    /// Chat mode: draw rules, labels and the assistant rail.
    interactive: bool,
    /// True when the next printed character starts a fresh output line.
    at_line_start: AtomicBool,
}

impl Clone for Output {
    fn clone(&self) -> Self {
        Self {
            json: self.json,
            quiet: self.quiet,
            color: self.color,
            interactive: self.interactive,
            at_line_start: AtomicBool::new(self.at_line_start.load(Ordering::Relaxed)),
        }
    }
}

impl Output {
    /// Headless output (`run`): raw text on stdout, chrome on stderr.
    pub fn new(json: bool) -> Self {
        let color =
            !json && std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
        Self {
            json,
            quiet: false,
            color,
            interactive: false,
            at_line_start: AtomicBool::new(true),
        }
    }

    /// Interactive REPL output: styled header, rules, and a message rail.
    pub fn chat() -> Self {
        let color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
        Self {
            json: false,
            quiet: false,
            color,
            interactive: color,
            at_line_start: AtomicBool::new(true),
        }
    }

    /// Silent output for subagents.
    pub fn quiet() -> Self {
        Self {
            json: false,
            quiet: true,
            color: false,
            interactive: false,
            at_line_start: AtomicBool::new(true),
        }
    }

    fn silent(&self) -> bool {
        self.json || self.quiet
    }

    fn paint(&self, code: &str, text: &str) -> String {
        if self.color {
            format!("{code}{text}{RESET}")
        } else {
            text.to_string()
        }
    }

    fn width(&self) -> usize {
        std::env::var("COLUMNS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(80)
            .clamp(40, 100)
    }

    fn rule(&self) -> String {
        "─".repeat(self.width())
    }

    /// Session header: identity, model, mode, endpoint, session, cwd.
    pub fn header(
        &self,
        version: &str,
        session: &str,
        model: &str,
        mode: &str,
        base_url: &str,
        cwd: &str,
    ) {
        if self.silent() {
            return;
        }
        eprintln!(
            "{} {}",
            self.paint(BOLD, "▌ rustyclippy-code"),
            self.paint(DIM, version)
        );
        eprintln!("  {} {}", self.paint(DIM, "model   "), model);
        eprintln!("  {} {}", self.paint(DIM, "mode    "), mode);
        eprintln!("  {} {}", self.paint(DIM, "endpoint"), base_url);
        eprintln!("  {} {}", self.paint(DIM, "session "), session);
        eprintln!("  {} {}", self.paint(DIM, "cwd     "), cwd);
        eprintln!("{}", self.paint(DIM, &self.rule()));
        eprintln!();
    }

    /// Dim hint line under the header.
    pub fn hint(&self, text: &str) {
        if self.silent() {
            return;
        }
        eprintln!("{} {}", self.paint(DIM, "·"), self.paint(DIM, text));
    }

    /// Styled input prompt marker.
    pub fn prompt_marker(&self) -> String {
        self.paint(&format!("{BOLD}{CYAN}"), "›")
    }

    /// Separate the user's prompt from the assistant answer.
    pub fn turn_begin(&self, turn: usize) {
        if self.silent() || !self.interactive {
            return;
        }
        println!();
        println!("{}", self.paint(DIM, &self.rule()));
        println!(
            "{} {}",
            self.paint(CYAN, "▌"),
            self.paint(BOLD, &format!("assistant · turn {turn}"))
        );
        self.at_line_start.store(true, Ordering::Relaxed);
    }

    /// Stream model text. In chat mode every line is prefixed with a dim rail.
    pub fn text(&self, chunk: &str) {
        if self.silent() {
            return;
        }
        if !self.interactive {
            print!("{chunk}");
            let _ = std::io::stdout().flush();
            return;
        }
        for ch in chunk.chars() {
            if self.at_line_start.load(Ordering::Relaxed) && ch != '\n' {
                print!("{}", self.paint(DIM, "│ "));
                self.at_line_start.store(false, Ordering::Relaxed);
            }
            print!("{ch}");
            if ch == '\n' {
                self.at_line_start.store(true, Ordering::Relaxed);
            }
        }
        let _ = std::io::stdout().flush();
    }

    /// Finish a streamed assistant message with a newline.
    pub fn line_break(&self) {
        if self.silent() {
            return;
        }
        println!();
        self.at_line_start.store(true, Ordering::Relaxed);
    }

    pub fn tool_call(&self, name: &str, arguments: &serde_json::Value) {
        if self.silent() {
            return;
        }
        let detail = serde_json::to_string(arguments).unwrap_or_default();
        let detail = truncate(&detail, 200);
        eprintln!(
            "  {} {} {}",
            self.paint(CYAN, "●"),
            self.paint(BOLD, name),
            self.paint(DIM, &detail)
        );
    }

    pub fn tool_result(&self, name: &str, result: &ToolResult) {
        if self.silent() {
            return;
        }
        let marker = if result.ok {
            self.paint(GREEN, "└")
        } else {
            self.paint(RED, "└")
        };
        let first_line = result
            .content
            .lines()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("(no output)");
        eprintln!(
            "  {} {} {}",
            marker,
            self.paint(DIM, name),
            truncate(first_line, 160)
        );
    }

    /// Per-request cache accounting.
    pub fn metrics(&self, turn: usize, step: usize, plan: &RequestPlan, usage: &Usage) {
        if self.silent() {
            return;
        }
        let prompt = if usage.prompt_tokens > 0 {
            usage.prompt_tokens
        } else {
            plan.estimated_prompt_tokens
        };
        let cached = usage.cached_tokens;
        let prefill = usage.prefill_tokens();
        let ratio = if prompt == 0 {
            0.0
        } else {
            cached as f64 / prompt as f64
        };
        // Report what the server actually did, not just what the local snapshot
        // predicted: a fresh session can still hit a warm server cache.
        let status = if cached > 0 {
            self.paint(GREEN, "prefix reused ✓")
        } else if plan.cache_cold {
            self.paint(YELLOW, "cache cold")
        } else if plan.stable {
            self.paint(GREEN, "prefix stable ✓")
        } else {
            self.paint(RED, "prefix rewritten ✗")
        };
        let reasoning = if usage.reasoning_tokens > 0 {
            format!(" reason={}", fmt_num(usage.reasoning_tokens))
        } else {
            String::new()
        };
        eprintln!(
            "  {} {} · prompt={} cached={} ({:.1}%) prefill={} out={}{} · {}",
            self.paint(DIM, "·"),
            self.paint(DIM, &format!("turn {turn} step {step}")),
            fmt_num(prompt),
            fmt_num(cached),
            ratio * 100.0,
            fmt_num(prefill),
            fmt_num(usage.completion_tokens),
            reasoning,
            status
        );
    }

    pub fn compaction_start(&self, dropped: usize, before_tokens: usize) {
        if self.silent() {
            return;
        }
        eprintln!(
            "  {} compacting {} messages (~{} tokens) at a cache-aligned boundary",
            self.paint(YELLOW, "⟳"),
            dropped,
            fmt_num(before_tokens)
        );
    }

    pub fn compaction_done(&self, record: &CompactionRecord, usage: &Usage) {
        if self.silent() {
            return;
        }
        eprintln!(
            "  {} compacted {} → {} tokens (summary {} tokens, prefill {} tokens)",
            self.paint(GREEN, "✓"),
            fmt_num(record.before_tokens),
            fmt_num(record.after_tokens),
            fmt_num(record.summary_tokens),
            fmt_num(usage.prefill_tokens())
        );
    }

    /// Footer for a finished turn.
    pub fn turn_summary(&self, turn: usize, usage: &Usage, steps: usize) {
        if self.silent() {
            return;
        }
        let prompt = usage.prompt_tokens;
        let cached = usage.cached_tokens;
        let ratio = if prompt == 0 {
            0.0
        } else {
            cached as f64 / prompt as f64
        };
        eprintln!(
            "{}",
            self.paint(
                DIM,
                &format!(
                    "── turn {turn} · {steps} steps · prompt={} cached={} ({:.1}%) prefill={} out={} ──",
                    fmt_num(prompt),
                    fmt_num(cached),
                    ratio * 100.0,
                    fmt_num(usage.prefill_tokens()),
                    fmt_num(usage.completion_tokens)
                )
            )
        );
    }

    /// Announce an intentional mode switch and its cache consequence.
    pub fn mode_switch(&self, mode: &str) {
        if self.silent() {
            return;
        }
        eprintln!(
            "{} mode → {} · system prompt and tool block rebuilt; the cached prefix is cold \
             (next request re-prefills)",
            self.paint(YELLOW, "⟳"),
            self.paint(BOLD, mode)
        );
    }

    /// Announce a pending image attachment.
    pub fn image_attached(&self, mime: &str, bytes: usize) {
        if self.silent() {
            return;
        }
        eprintln!(
            "{} image attached · {} · {}",
            self.paint(CYAN, "▌"),
            mime,
            fmt_bytes(bytes)
        );
    }

    pub fn pending_images(&self, count: usize) {
        if self.silent() {
            return;
        }
        eprintln!(
            "{} {} image(s) attached — send them with the next prompt",
            self.paint(DIM, "·"),
            count
        );
    }

    pub fn warning(&self, message: &str) {
        if self.silent() {
            return;
        }
        eprintln!("{} {}", self.paint(YELLOW, "warning:"), message);
    }

    pub fn error(&self, message: &str) {
        if self.silent() {
            return;
        }
        eprintln!("{} {}", self.paint(RED, "error:"), message);
    }
}

pub fn fmt_num(value: usize) -> String {
    let digits = value.to_string();
    let mut out = String::new();
    let len = digits.len();
    for (index, ch) in digits.chars().enumerate() {
        if index > 0 && (len - index) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

pub fn fmt_bytes(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1_024 {
        format!("{:.1} KB", bytes as f64 / 1_024.0)
    } else {
        format!("{bytes} B")
    }
}

pub fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out = String::new();
    for ch in text.chars().take(max) {
        out.push(ch);
    }
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_get_thousands_separators() {
        assert_eq!(fmt_num(0), "0");
        assert_eq!(fmt_num(999), "999");
        assert_eq!(fmt_num(1000), "1,000");
        assert_eq!(fmt_num(1234567), "1,234,567");
    }

    #[test]
    fn truncate_is_char_safe() {
        assert_eq!(truncate("日本語", 2), "日本…");
    }

    #[test]
    fn bytes_are_human_readable() {
        assert_eq!(fmt_bytes(512), "512 B");
        assert_eq!(fmt_bytes(2048), "2.0 KB");
        assert_eq!(fmt_bytes(2_097_152), "2.0 MB");
    }
}

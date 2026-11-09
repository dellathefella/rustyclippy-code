//! Terminal output: streaming text, tool activity, and cache metrics.

use std::io::{IsTerminal, Write};

use deeprust_core::{CompactionRecord, RequestPlan};
use deeprust_llm::Usage;
use deeprust_tools::ToolResult;

const RESET: &str = "\x1b[0m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const BOLD: &str = "\x1b[1m";

#[derive(Debug, Clone)]
pub struct Output {
    pub json: bool,
    pub quiet: bool,
    color: bool,
}

impl Output {
    pub fn new(json: bool) -> Self {
        let color =
            !json && std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
        Self {
            json,
            quiet: false,
            color,
        }
    }

    /// Silent output for subagents.
    pub fn quiet() -> Self {
        Self {
            json: false,
            quiet: true,
            color: false,
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

    /// Stream model text to stdout.
    pub fn text(&self, chunk: &str) {
        if self.silent() {
            return;
        }
        print!("{chunk}");
        let _ = std::io::stdout().flush();
    }

    pub fn banner(&self, session: &str, model: &str, mode: &str, base_url: &str) {
        if self.silent() {
            return;
        }
        eprintln!(
            "{} {} {} {}",
            self.paint(BOLD, "deeprust"),
            self.paint(DIM, "·"),
            model,
            self.paint(DIM, &format!("({mode}, {base_url})"))
        );
        eprintln!("{} {}", self.paint(DIM, "session"), session);
        eprintln!();
    }

    pub fn tool_call(&self, name: &str, arguments: &serde_json::Value) {
        if self.silent() {
            return;
        }
        let detail = serde_json::to_string(arguments).unwrap_or_default();
        let detail = truncate(&detail, 200);
        eprintln!(
            "{} {} {}",
            self.paint(CYAN, "→"),
            self.paint(BOLD, name),
            self.paint(DIM, &detail)
        );
    }

    pub fn tool_result(&self, name: &str, result: &ToolResult) {
        if self.silent() {
            return;
        }
        let marker = if result.ok {
            self.paint(GREEN, "←")
        } else {
            self.paint(RED, "←")
        };
        let first_line = result
            .content
            .lines()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("(no output)");
        eprintln!(
            "{marker} {} {}",
            self.paint(DIM, name),
            truncate(first_line, 200)
        );
    }

    /// Per-request cache accounting.
    pub fn metrics(&self, turn: usize, step: usize, plan: &RequestPlan, usage: &Usage) {
        if self.silent() {
            return;
        }
        let prompt = usage.prompt_tokens.max(plan.estimated_prompt_tokens);
        let cached = usage.cached_tokens;
        let prefill = usage.prefill_tokens();
        let ratio = if prompt == 0 {
            0.0
        } else {
            cached as f64 / prompt as f64
        };
        let status = if plan.cache_cold {
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
            "{} prompt={} cached={} ({:.1}%) prefill={} out={}{} | {}",
            self.paint(DIM, &format!("[turn {turn} step {step}]")),
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
            "{} compacting {} messages (~{} tokens) at a cache-aligned boundary",
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
            "{} compacted {} → {} tokens (summary {} tokens, prefill {} tokens)",
            self.paint(GREEN, "✓"),
            fmt_num(record.before_tokens),
            fmt_num(record.after_tokens),
            fmt_num(record.summary_tokens),
            fmt_num(usage.prefill_tokens())
        );
    }

    pub fn turn_summary(&self, usage: &Usage, steps: usize) {
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
            "{} {} steps · prompt={} cached={} ({:.1}%) prefill={} out={}",
            self.paint(DIM, "·"),
            steps,
            fmt_num(prompt),
            fmt_num(cached),
            ratio * 100.0,
            fmt_num(usage.prefill_tokens()),
            fmt_num(usage.completion_tokens)
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
}

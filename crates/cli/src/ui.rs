//! Terminal output: an opencode-inspired REPL surface.
//!
//! Chrome (header, rules, tool activity, metrics) goes to stderr so headless
//! `run` output on stdout stays clean. In chat mode the assistant stream is
//! rendered under a dim rail so the prompt and the answer never blur together.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

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

/// One transcript row produced by the agent, consumed by the TUI renderer.
#[derive(Debug, Clone)]
pub enum FeedEntry {
    /// A prompt typed by the user.
    User(String),
    /// Assistant text, streamed in chunks.
    Text(String),
    /// A tool invocation line.
    ToolCall { name: String, detail: String },
    /// A tool result line.
    ToolResult {
        name: String,
        ok: bool,
        detail: String,
    },
    /// Per-request cache accounting.
    Metrics(String),
    /// Compaction progress.
    Compaction(String),
    /// Warning / error chrome.
    Notice { level: NoticeLevel, text: String },
    /// Turn footer.
    TurnSummary(String),
    /// Mode switch announcement.
    ModeSwitch(String),
    /// Image attachment announcement.
    Image(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoticeLevel {
    Hint,
    Warning,
    Error,
}

/// What the user chose at a permission prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalAnswer {
    Yes,
    No,
    Session,
    Select,
}

/// A batch of tool calls waiting for the user's approval.
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    /// `(tool name, human detail)` for each call in the batch.
    pub calls: Vec<(String, String)>,
}

impl ApprovalRequest {
    pub fn is_batch(&self) -> bool {
        self.calls.len() > 1
    }
}

/// Session identity the TUI draws in its header.
#[derive(Debug, Clone)]
pub struct HeaderInfo {
    pub version: String,
    pub session: String,
    pub model: String,
    pub mode: String,
    pub base_url: String,
    pub cwd: String,
}

/// Running token totals for the footer.
#[derive(Debug, Default, Clone)]
pub struct Totals {
    pub turns: usize,
    pub prompt: usize,
    pub cached: usize,
    pub prefill: usize,
    pub completion: usize,
}

impl Totals {
    pub fn hit_ratio(&self) -> f64 {
        if self.prompt == 0 {
            0.0
        } else {
            self.cached as f64 / self.prompt as f64
        }
    }
}

/// Shared transcript buffer the agent writes into and the TUI reads.
#[derive(Debug, Default)]
pub struct UiFeed {
    entries: Vec<FeedEntry>,
    /// Bumped on every push so the renderer knows to redraw.
    revision: u64,
    /// Set while the agent is blocked waiting for an approval answer.
    approval: Option<ApprovalRequest>,
    /// Channel the renderer uses to deliver the answer.
    approval_tx: Option<tokio::sync::oneshot::Sender<ApprovalAnswer>>,
    /// Live session identity for the header.
    header: Option<HeaderInfo>,
    /// Running token totals for the footer.
    totals: Totals,
    /// True while a turn is in flight.
    busy: bool,
}

impl UiFeed {
    pub fn push(&mut self, entry: FeedEntry) {
        self.entries.push(entry);
        self.revision += 1;
    }

    pub fn entries(&self) -> &[FeedEntry] {
        &self.entries
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.revision += 1;
    }

    pub fn set_header(&mut self, header: HeaderInfo) {
        self.header = Some(header);
        self.revision += 1;
    }

    pub fn header(&self) -> Option<&HeaderInfo> {
        self.header.as_ref()
    }

    pub fn totals(&self) -> &Totals {
        &self.totals
    }

    pub fn add_turn(&mut self, prompt: usize, cached: usize, prefill: usize, completion: usize) {
        self.totals.turns += 1;
        self.totals.prompt += prompt;
        self.totals.cached += cached;
        self.totals.prefill += prefill;
        self.totals.completion += completion;
        self.revision += 1;
    }

    pub fn set_busy(&mut self, busy: bool) {
        self.busy = busy;
        self.revision += 1;
    }

    pub fn busy(&self) -> bool {
        self.busy
    }

    /// Register a pending approval request and the channel that answers it.
    pub fn request_approval(
        &mut self,
        request: ApprovalRequest,
        tx: tokio::sync::oneshot::Sender<ApprovalAnswer>,
    ) {
        self.approval = Some(request);
        self.approval_tx = Some(tx);
        self.revision += 1;
    }

    pub fn pending_approval(&self) -> Option<&ApprovalRequest> {
        self.approval.as_ref()
    }

    /// Deliver the user's answer to the blocked agent. Returns false when no
    /// request is pending.
    pub fn answer_approval(&mut self, answer: ApprovalAnswer) -> bool {
        let Some(tx) = self.approval_tx.take() else {
            return false;
        };
        self.approval = None;
        self.revision += 1;
        tx.send(answer).is_ok()
    }
}

#[derive(Debug)]
pub struct Output {
    pub json: bool,
    pub quiet: bool,
    color: bool,
    /// Chat mode: draw rules, labels and the assistant rail.
    interactive: bool,
    /// True when the next printed character starts a fresh output line.
    at_line_start: AtomicBool,
    /// When set, output is collected for a TUI instead of printed.
    feed: Option<Arc<Mutex<UiFeed>>>,
}

impl Clone for Output {
    fn clone(&self) -> Self {
        Self {
            json: self.json,
            quiet: self.quiet,
            color: self.color,
            interactive: self.interactive,
            at_line_start: AtomicBool::new(self.at_line_start.load(Ordering::Relaxed)),
            feed: self.feed.clone(),
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
            feed: None,
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
            feed: None,
        }
    }

    /// Output routed into a TUI transcript instead of the terminal.
    pub fn tui(feed: Arc<Mutex<UiFeed>>) -> Self {
        Self {
            json: false,
            quiet: false,
            color: false,
            interactive: false,
            at_line_start: AtomicBool::new(true),
            feed: Some(feed),
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
            feed: None,
        }
    }

    fn silent(&self) -> bool {
        self.json || self.quiet
    }

    fn feed(&self) -> Option<&Arc<Mutex<UiFeed>>> {
        self.feed.as_ref()
    }

    fn push(&self, entry: FeedEntry) {
        if let Some(feed) = self.feed() {
            if let Ok(mut feed) = feed.lock() {
                feed.push(entry);
            }
        }
    }

    /// True when output is routed into a TUI transcript.
    pub fn is_tui(&self) -> bool {
        self.feed().is_some()
    }

    /// Clear the transcript pane (TUI only).
    pub fn clear_transcript(&self) {
        if let Some(feed) = self.feed() {
            if let Ok(mut feed) = feed.lock() {
                feed.clear();
            }
        }
    }

    /// Record the prompt the user typed so the transcript shows it.
    pub fn user_prompt(&self, text: &str) {
        if self.silent() {
            return;
        }
        if self.feed().is_some() {
            self.push(FeedEntry::User(text.to_string()));
        }
    }

    /// Ask the user to approve a batch of tool calls.
    ///
    /// In TUI mode this publishes the request on the feed and awaits the
    /// renderer's answer. Returns `None` when output is not routed to a TUI, so
    /// the caller can fall back to stdin.
    pub async fn ask_approval(&self, request: ApprovalRequest) -> Option<ApprovalAnswer> {
        let feed = self.feed()?;
        let (tx, rx) = tokio::sync::oneshot::channel();
        match feed.lock() {
            Ok(mut feed) => feed.request_approval(request, tx),
            Err(_) => return Some(ApprovalAnswer::No),
        }
        Some(rx.await.unwrap_or(ApprovalAnswer::No))
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
        if let Some(feed) = self.feed() {
            if let Ok(mut feed) = feed.lock() {
                feed.set_header(HeaderInfo {
                    version: version.to_string(),
                    session: session.to_string(),
                    model: model.to_string(),
                    mode: mode.to_string(),
                    base_url: base_url.to_string(),
                    cwd: cwd.to_string(),
                });
            }
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
        if self.feed().is_some() {
            self.push(FeedEntry::Notice {
                level: NoticeLevel::Hint,
                text: text.to_string(),
            });
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
        if self.silent() {
            return;
        }
        if let Some(feed) = self.feed() {
            if let Ok(mut feed) = feed.lock() {
                feed.set_busy(true);
            }
            return;
        }
        if !self.interactive {
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
        if self.feed().is_some() {
            self.push(FeedEntry::Text(chunk.to_string()));
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
        if self.feed().is_some() {
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
        if self.feed().is_some() {
            self.push(FeedEntry::ToolCall {
                name: name.to_string(),
                detail,
            });
            return;
        }
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
        let first_line = result
            .content
            .lines()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("(no output)");
        if self.feed().is_some() {
            self.push(FeedEntry::ToolResult {
                name: name.to_string(),
                ok: result.ok,
                detail: truncate(first_line, 160),
            });
            return;
        }
        let marker = if result.ok {
            self.paint(GREEN, "└")
        } else {
            self.paint(RED, "└")
        };
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
        let line = format!(
            "turn {turn} step {step} · prompt={} cached={} ({:.1}%) prefill={} out={}{} · {}",
            fmt_num(prompt),
            fmt_num(cached),
            ratio * 100.0,
            fmt_num(prefill),
            fmt_num(usage.completion_tokens),
            reasoning,
            status
        );
        if self.feed().is_some() {
            self.push(FeedEntry::Metrics(line));
            return;
        }
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
        let line = format!(
            "compacting {} messages (~{} tokens) at a cache-aligned boundary",
            dropped,
            fmt_num(before_tokens)
        );
        if self.feed().is_some() {
            self.push(FeedEntry::Compaction(line));
            return;
        }
        eprintln!("  {} {}", self.paint(YELLOW, "⟳"), line);
    }

    pub fn compaction_done(&self, record: &CompactionRecord, usage: &Usage) {
        if self.silent() {
            return;
        }
        let line = format!(
            "compacted {} → {} tokens (summary {} tokens, prefill {} tokens)",
            fmt_num(record.before_tokens),
            fmt_num(record.after_tokens),
            fmt_num(record.summary_tokens),
            fmt_num(usage.prefill_tokens())
        );
        if self.feed().is_some() {
            self.push(FeedEntry::Compaction(line));
            return;
        }
        eprintln!("  {} {}", self.paint(GREEN, "✓"), line);
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
        let line = format!(
            "turn {turn} · {steps} steps · prompt={} cached={} ({:.1}%) prefill={} out={}",
            fmt_num(prompt),
            fmt_num(cached),
            ratio * 100.0,
            fmt_num(usage.prefill_tokens()),
            fmt_num(usage.completion_tokens)
        );
        if self.feed().is_some() {
            self.push(FeedEntry::TurnSummary(line));
            if let Some(feed) = self.feed() {
                if let Ok(mut feed) = feed.lock() {
                    feed.add_turn(
                        prompt,
                        cached,
                        usage.prefill_tokens(),
                        usage.completion_tokens,
                    );
                    feed.set_busy(false);
                }
            }
            return;
        }
        eprintln!("{}", self.paint(DIM, &format!("── {line} ──")));
    }

    /// Announce an intentional mode switch and its cache consequence.
    pub fn mode_switch(&self, mode: &str) {
        if self.silent() {
            return;
        }
        let line = format!(
            "mode → {mode} · system prompt and tool block rebuilt; the cached prefix is cold \
             (next request re-prefills)"
        );
        if self.feed().is_some() {
            self.push(FeedEntry::ModeSwitch(line));
            return;
        }
        eprintln!("{} {}", self.paint(YELLOW, "⟳"), line);
    }

    /// Announce a pending image attachment.
    pub fn image_attached(&self, mime: &str, bytes: usize) {
        if self.silent() {
            return;
        }
        let line = format!("image attached · {mime} · {}", fmt_bytes(bytes));
        if self.feed().is_some() {
            self.push(FeedEntry::Image(line));
            return;
        }
        eprintln!("{} {}", self.paint(CYAN, "▌"), line);
    }

    pub fn pending_images(&self, count: usize) {
        if self.silent() {
            return;
        }
        let line = format!("{count} image(s) attached — send them with the next prompt");
        if self.feed().is_some() {
            self.push(FeedEntry::Notice {
                level: NoticeLevel::Hint,
                text: line,
            });
            return;
        }
        eprintln!("{} {}", self.paint(DIM, "·"), line);
    }

    pub fn warning(&self, message: &str) {
        if self.silent() {
            return;
        }
        if self.feed().is_some() {
            self.push(FeedEntry::Notice {
                level: NoticeLevel::Warning,
                text: message.to_string(),
            });
            return;
        }
        eprintln!("{} {}", self.paint(YELLOW, "warning:"), message);
    }

    pub fn error(&self, message: &str) {
        if self.silent() {
            return;
        }
        if self.feed().is_some() {
            self.push(FeedEntry::Notice {
                level: NoticeLevel::Error,
                text: message.to_string(),
            });
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

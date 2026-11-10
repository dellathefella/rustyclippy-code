//! Terminal output: an opencode-inspired REPL surface.
//!
//! Chrome (header, rules, tool activity, metrics) goes to stderr so headless
//! `run` output on stdout stays clean. In chat mode the assistant stream is
//! rendered under a dim rail so the prompt and the answer never blur together.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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
/// Hide the caret while a spinner repaints in place.
const HIDE_CURSOR: &str = "\x1b[?25l";
const SHOW_CURSOR: &str = "\x1b[?25h";

/// Braille frames for the thinking spinner.
pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Human-readable one-line summary of a tool call's arguments.
///
/// Raw JSON is unreadable in a transcript (`edit {"new_string":"..."`), so each
/// tool reports the argument that actually identifies the work: the command for
/// `bash`, the path for file tools, the query for search, and so on.
pub fn tool_detail(name: &str, args: &serde_json::Value) -> String {
    let text = |key: &str| {
        args.get(key)
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string()
    };
    let detail = match name {
        "bash" | "loop" => text("command"),
        "read" | "write" | "edit" | "list" => text("path"),
        "glob" | "grep" => {
            let pattern = text("pattern");
            let path = text("path");
            if path.is_empty() {
                pattern
            } else {
                format!("{pattern} in {path}")
            }
        }
        "websearch" => text("query"),
        "webfetch" => text("url"),
        "task" => text("description"),
        "wait" => text("task_id"),
        "question" => text("question"),
        "todo" => {
            let count = args
                .get("todos")
                .and_then(|value| value.as_array())
                .map(|items| items.len())
                .unwrap_or(0);
            format!("{count} item(s)")
        }
        _ => String::new(),
    };
    truncate(&detail, 120)
}

/// One transcript row produced by the agent, consumed by the TUI renderer.
#[derive(Debug, Clone)]
pub enum FeedEntry {
    /// A prompt typed by the user.
    User(String),
    /// Assistant text, streamed in chunks.
    Text(String),
    /// Model reasoning ("thinking"), streamed in chunks. Only emitted when the
    /// user has enabled thinking output.
    Reasoning(String),
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
    /// Estimated prompt tokens for the most recent request.
    pub context_tokens: usize,
    /// Token budget that triggers autocompaction.
    pub compaction_budget: usize,
    /// Decode speed of the most recent request, in tokens/s.
    pub decode_tokens_per_second: Option<f64>,
    /// Reasoning ("thinking") tokens accumulated across turns.
    pub reasoning: usize,
}

impl Totals {
    pub fn hit_ratio(&self) -> f64 {
        if self.prompt == 0 {
            0.0
        } else {
            self.cached as f64 / self.prompt as f64
        }
    }

    /// Fraction of the compaction budget currently in use, clamped to 1.0.
    pub fn compaction_ratio(&self) -> f64 {
        if self.compaction_budget == 0 {
            return 0.0;
        }
        (self.context_tokens as f64 / self.compaction_budget as f64).min(1.0)
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
    /// Set when the user asks to interrupt the running turn.
    interrupt: bool,
    /// True while a reasoning block is streaming, so the transcript can animate
    /// the thinking rail instead of freezing it.
    thinking: bool,
    /// Estimated reasoning tokens streamed so far in the current turn.
    thinking_tokens: usize,
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

    /// Record how full the context is relative to the compaction budget, plus
    /// the decode speed of the request that just finished.
    pub fn set_context(
        &mut self,
        context_tokens: usize,
        compaction_budget: usize,
        decode_tokens_per_second: Option<f64>,
    ) {
        self.totals.context_tokens = context_tokens;
        self.totals.compaction_budget = compaction_budget;
        if decode_tokens_per_second.is_some() {
            self.totals.decode_tokens_per_second = decode_tokens_per_second;
        }
        self.revision += 1;
    }

    pub fn add_turn(
        &mut self,
        prompt: usize,
        cached: usize,
        prefill: usize,
        completion: usize,
        reasoning: usize,
    ) {
        self.totals.turns += 1;
        self.totals.prompt += prompt;
        self.totals.cached += cached;
        self.totals.prefill += prefill;
        self.totals.completion += completion;
        self.totals.reasoning += reasoning;
        self.revision += 1;
    }

    pub fn set_busy(&mut self, busy: bool) {
        self.busy = busy;
        self.revision += 1;
    }

    pub fn busy(&self) -> bool {
        self.busy
    }

    /// Ask the running turn to stop. The agent checks this between steps and
    /// while streaming, so an interrupt lands promptly without killing the
    /// process.
    pub fn request_interrupt(&mut self) {
        self.interrupt = true;
        self.revision += 1;
    }

    pub fn interrupt_requested(&self) -> bool {
        self.interrupt
    }

    /// Update the live thinking state: whether reasoning is streaming and how
    /// many tokens it has produced so far.
    pub fn set_thinking(&mut self, thinking: bool, tokens: usize) {
        self.thinking = thinking;
        self.thinking_tokens = tokens;
        self.revision += 1;
    }

    pub fn thinking(&self) -> bool {
        self.thinking
    }

    pub fn thinking_tokens(&self) -> usize {
        self.thinking_tokens
    }

    /// Clear the interrupt flag once the turn has stopped.
    pub fn clear_interrupt(&mut self) {
        self.interrupt = false;
        self.revision += 1;
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
    /// Render model reasoning ("thinking") in the transcript. On by default;
    /// shared across clones so `/thinking` toggles every surface at once.
    show_reasoning: Arc<AtomicBool>,
    /// Token budget that triggers autocompaction, for the footer gauge.
    compaction_budget: usize,
    /// Spinner frame counter for the animated thinking indicator.
    spinner: AtomicUsize,
    /// True while a reasoning block is streaming, so the spinner can be
    /// repainted in place instead of printing a new line per chunk.
    thinking: AtomicBool,
    /// Reasoning tokens streamed in the current turn, estimated from the text.
    thinking_tokens: AtomicUsize,
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
            show_reasoning: self.show_reasoning.clone(),
            compaction_budget: self.compaction_budget,
            spinner: AtomicUsize::new(self.spinner.load(Ordering::Relaxed)),
            thinking: AtomicBool::new(self.thinking.load(Ordering::Relaxed)),
            thinking_tokens: AtomicUsize::new(self.thinking_tokens.load(Ordering::Relaxed)),
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
            show_reasoning: Arc::new(AtomicBool::new(true)),
            compaction_budget: 0,
            spinner: AtomicUsize::new(0),
            thinking: AtomicBool::new(false),
            thinking_tokens: AtomicUsize::new(0),
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
            show_reasoning: Arc::new(AtomicBool::new(true)),
            compaction_budget: 0,
            spinner: AtomicUsize::new(0),
            thinking: AtomicBool::new(false),
            thinking_tokens: AtomicUsize::new(0),
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
            show_reasoning: Arc::new(AtomicBool::new(true)),
            compaction_budget: 0,
            spinner: AtomicUsize::new(0),
            thinking: AtomicBool::new(false),
            thinking_tokens: AtomicUsize::new(0),
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
            show_reasoning: Arc::new(AtomicBool::new(true)),
            compaction_budget: 0,
            spinner: AtomicUsize::new(0),
            thinking: AtomicBool::new(false),
            thinking_tokens: AtomicUsize::new(0),
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

    /// True when the user asked to interrupt the running turn.
    pub fn interrupt_requested(&self) -> bool {
        self.feed()
            .and_then(|feed| feed.lock().ok().map(|feed| feed.interrupt_requested()))
            .unwrap_or(false)
    }

    /// Clear the interrupt flag after a turn stops.
    pub fn clear_interrupt(&self) {
        if let Some(feed) = self.feed() {
            if let Ok(mut feed) = feed.lock() {
                feed.clear_interrupt();
            }
        }
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
        // Each turn counts its own thinking tokens.
        self.thinking_tokens.store(0, Ordering::Relaxed);
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

    /// Record the compaction budget so the footer can show how full the
    /// context is. Called once when the agent is built.
    pub fn set_compaction_budget(&mut self, budget: usize) {
        self.compaction_budget = budget;
    }

    /// True when model reasoning is rendered in the transcript.
    pub fn show_reasoning(&self) -> bool {
        self.show_reasoning.load(Ordering::Relaxed)
    }

    /// Enable or disable reasoning rendering. Returns the new state.
    pub fn set_show_reasoning(&self, enabled: bool) -> bool {
        self.show_reasoning.store(enabled, Ordering::Relaxed);
        enabled
    }

    /// Stream model reasoning ("thinking"). Rendered dimmed under a `⋯` rail so
    /// it reads as scratch work rather than the answer.
    ///
    /// In chat mode the first chunk paints a spinner that advances on every
    /// chunk, so the block visibly "thinks" instead of appearing as a wall of
    /// text. The spinner is repainted in place with `\r` and cleared once the
    /// reasoning ends.
    pub fn reasoning(&self, chunk: &str) {
        if self.silent() || !self.show_reasoning() {
            return;
        }
        // Count thinking tokens as they stream so the footer and the transcript
        // rail can show progress before the turn finishes.
        let delta = estimate_tokens(chunk);
        let tokens = self.thinking_tokens.fetch_add(delta, Ordering::Relaxed) + delta;
        if self.feed().is_some() {
            if let Some(feed) = self.feed() {
                if let Ok(mut feed) = feed.lock() {
                    feed.set_thinking(true, tokens);
                }
            }
            self.push(FeedEntry::Reasoning(chunk.to_string()));
            return;
        }
        if !self.interactive {
            eprintln!("{chunk}");
            return;
        }
        let frame = self.spinner.fetch_add(1, Ordering::Relaxed) % SPINNER.len();
        let glyph = SPINNER[frame];
        // Hide the caret so the spinner glyph does not flicker against it.
        if !self.thinking.load(Ordering::Relaxed) {
            print!("{HIDE_CURSOR}");
        }
        for ch in chunk.chars() {
            if self.at_line_start.load(Ordering::Relaxed) && ch != '\n' {
                print!("{}", self.paint(DIM, &format!("{glyph} ")));
                self.at_line_start.store(false, Ordering::Relaxed);
            }
            print!("{}", self.paint(DIM, &ch.to_string()));
            if ch == '\n' {
                self.at_line_start.store(true, Ordering::Relaxed);
            }
        }
        self.thinking.store(true, Ordering::Relaxed);
        let _ = std::io::stdout().flush();
    }

    /// Finish a streamed reasoning block: clear the spinner line and reset.
    pub fn reasoning_end(&self) {
        if self.silent() || !self.show_reasoning() {
            return;
        }
        if self.feed().is_some() {
            if let Some(feed) = self.feed() {
                if let Ok(mut feed) = feed.lock() {
                    feed.set_thinking(false, 0);
                }
            }
            return;
        }
        if self.thinking.swap(false, Ordering::Relaxed) {
            // The spinner glyph sits at the start of the current line; erase it
            // so the answer below starts clean, then restore the caret.
            print!("\r\x1b[K{SHOW_CURSOR}");
        }
        println!();
        self.at_line_start.store(true, Ordering::Relaxed);
        let _ = std::io::stdout().flush();
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
        let detail = tool_detail(name, arguments);
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
        let summary = tool_result_summary(result);
        if self.feed().is_some() {
            self.push(FeedEntry::ToolResult {
                name: name.to_string(),
                ok: result.ok,
                detail: summary,
            });
            return;
        }
        let marker = if result.ok {
            self.paint(GREEN, "└")
        } else {
            self.paint(RED, "└")
        };
        eprintln!("  {} {} {}", marker, self.paint(DIM, name), summary);
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
            // Feed the footer's context gauge and decode speed.
            if let Some(feed) = self.feed() {
                if let Ok(mut feed) = feed.lock() {
                    feed.set_context(
                        plan.estimated_prompt_tokens,
                        self.compaction_budget,
                        usage.decode_tokens_per_second,
                    );
                }
            }
            self.push(FeedEntry::Metrics(line));
            return;
        }
        // Chat mode stays quiet: a healthy cache hit is the normal case, and a
        // per-step accounting line for every step buries the conversation. Only
        // surface the states worth noticing; `run` keeps the full line.
        if self.interactive && cached > 0 && plan.stable {
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
                        usage.reasoning_tokens,
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

/// Rough token count for a streamed chunk, matching the estimator's heuristic
/// (CJK counts as one token, other text at ~3.6 chars per token).
fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    let mut cjk = 0usize;
    let mut other = 0usize;
    for ch in text.chars() {
        if is_cjk(ch) {
            cjk += 1;
        } else {
            other += 1;
        }
    }
    cjk + ((other as f64) / 3.6).ceil() as usize
}

fn is_cjk(ch: char) -> bool {
    matches!(ch as u32,
        0x4E00..=0x9FFF
        | 0x3400..=0x4DBF
        | 0x3040..=0x30FF
        | 0xAC00..=0xD7AF
        | 0xF900..=0xFAFF
    )
}

/// Compact one-line summary of a tool result.
///
/// Shell output arrives as `exit_code: N\n--- stdout ---\n...`, which reads as
/// noise in a transcript. Report the exit status and the first meaningful line
/// instead, and collapse empty output to a short marker.
pub fn tool_result_summary(result: &ToolResult) -> String {
    let content = result.content.trim();
    if content.is_empty() {
        return if result.ok {
            "ok".to_string()
        } else {
            "failed".to_string()
        };
    }
    // `bash`/`loop` prefix their output with an exit code and section headers.
    if let Some(rest) = content.strip_prefix("exit_code: ") {
        let mut lines = rest.lines();
        let code = lines.next().unwrap_or("?").trim();
        let body = lines
            .find(|line| {
                let line = line.trim();
                !line.is_empty() && !line.starts_with("---")
            })
            .unwrap_or("");
        let status = if code == "0" {
            format!("exit {code}")
        } else {
            format!("exit {code} ✗")
        };
        if body.is_empty() {
            return status;
        }
        return format!("{status} · {}", truncate(body.trim(), 120));
    }
    let first = content
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("(no output)");
    truncate(first.trim(), 120)
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
    fn tool_detail_is_human_readable() {
        let edit = serde_json::json!({
            "path": "src/main.rs",
            "old_string": "a",
            "new_string": "b"
        });
        assert_eq!(tool_detail("edit", &edit), "src/main.rs");

        let bash = serde_json::json!({ "command": "cargo test" });
        assert_eq!(tool_detail("bash", &bash), "cargo test");

        let grep = serde_json::json!({ "pattern": "fn main", "path": "src" });
        assert_eq!(tool_detail("grep", &grep), "fn main in src");

        let glob = serde_json::json!({ "pattern": "**/*.rs" });
        assert_eq!(tool_detail("glob", &glob), "**/*.rs");

        let todo = serde_json::json!({ "todos": [{"content": "a", "status": "pending"}] });
        assert_eq!(tool_detail("todo", &todo), "1 item(s)");
    }

    #[test]
    fn tool_detail_never_dumps_raw_json() {
        let edit = serde_json::json!({ "path": "a.rs", "new_string": "x".repeat(400) });
        let detail = tool_detail("edit", &edit);
        assert!(!detail.contains("new_string"), "raw JSON leaked: {detail}");
        assert!(detail.chars().count() <= 121);
    }

    #[test]
    fn tool_result_summary_collapses_shell_output() {
        let ok = ToolResult::success("exit_code: 0\n--- stdout ---\nsrc\n--- stderr ---\n");
        assert_eq!(tool_result_summary(&ok), "exit 0 · src");

        let failed = ToolResult::success("exit_code: 3\n--- stdout ---\n--- stderr ---\n");
        assert_eq!(tool_result_summary(&failed), "exit 3 ✗");

        let empty = ToolResult::success("");
        assert_eq!(tool_result_summary(&empty), "ok");
    }

    #[test]
    fn tool_result_summary_keeps_first_line_for_other_tools() {
        let read = ToolResult::success("\n\nfn main() {}\nmore");
        assert_eq!(tool_result_summary(&read), "fn main() {}");
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

//! Full-screen TUI renderer, styled after opencode.
//!
//! Layout: a header bar with session identity, a scrolling transcript pane, a
//! bordered input box, and a footer with live token totals. The agent writes
//! [`FeedEntry`] rows into a [`UiFeed`]; this module owns the terminal and
//! redraws whenever the feed revision changes.

use std::io;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};
use ratatui::Terminal;

use crate::ui::{ApprovalAnswer, FeedEntry, NoticeLevel, UiFeed};

/// Palette tuned to read like opencode: muted chrome, bright accents.
const ACCENT: Color = Color::Rgb(0x7d, 0xd8, 0xff);
const USER: Color = Color::Rgb(0x9d, 0x9d, 0xff);
const TOOL: Color = Color::Rgb(0x8f, 0xd6, 0x8f);
const WARN: Color = Color::Rgb(0xf5, 0xc2, 0x42);
const ERR: Color = Color::Rgb(0xff, 0x6b, 0x6b);
const MUTED: Color = Color::Rgb(0x6b, 0x6b, 0x7b);
const RAIL: Color = Color::Rgb(0x3a, 0x3a, 0x48);

/// Fixed pane heights, shared by the layout and the scroll math.
const HEADER_HEIGHT: u16 = 5;
const INPUT_HEIGHT: u16 = 3;
const FOOTER_HEIGHT: u16 = 2;

/// What the renderer hands back to the caller when the user submits a line.
pub enum TuiOutcome {
    /// A prompt was submitted; drain it with [`Tui::take_submitted`].
    Prompt,
    /// The user asked to switch between build and plan mode.
    ToggleMode,
    /// The user asked to quit.
    Exit,
}

/// Owns the terminal for the lifetime of the TUI session.
pub struct Tui {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
    feed: Arc<Mutex<UiFeed>>,
    input: String,
    cursor: usize,
    /// Transcript scroll offset in lines, counted from the bottom.
    scroll: usize,
    /// Last feed revision drawn, so we only redraw on change.
    last_revision: u64,
    /// Set when the input line or scroll offset changed, forcing a redraw.
    dirty: bool,
    /// Prompts submitted with Enter, in order. The caller drains them one at a
    /// time, so a prompt typed while a turn runs is queued rather than lost.
    submitted: Vec<String>,
    /// Images attached to the next prompt.
    pending_images: Vec<rustyclippy_core::ImageAttachment>,
    /// Spinner frame, advanced on every draw while a turn is in flight.
    spinner: usize,
    /// Prompt history for Up/Down recall.
    history: crate::history::History,
    /// Set when Shift+Tab asked for a mode switch; the caller drains it.
    toggle_mode: bool,
    quit: bool,
}

impl Tui {
    /// Enter the alternate screen and take over the terminal.
    pub fn enter(feed: Arc<Mutex<UiFeed>>, history: crate::history::History) -> Result<Self> {
        enable_raw_mode().context("enabling raw terminal mode")?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableBracketedPaste)
            .context("entering alternate screen")?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend).context("creating terminal")?;
        Ok(Self {
            terminal,
            feed,
            input: String::new(),
            cursor: 0,
            scroll: 0,
            last_revision: u64::MAX,
            dirty: true,
            submitted: Vec::new(),
            pending_images: Vec::new(),
            spinner: 0,
            history,
            toggle_mode: false,
            quit: false,
        })
    }

    /// Restore the terminal. Safe to call twice.
    pub fn leave(&mut self) -> Result<()> {
        disable_raw_mode().ok();
        execute!(
            self.terminal.backend_mut(),
            DisableBracketedPaste,
            LeaveAlternateScreen
        )
        .ok();
        self.terminal.show_cursor().ok();
        Ok(())
    }

    /// Draw one frame if the feed changed. `pending_images` is the count the
    /// caller holds for the next prompt (the TUI hands captured images over on
    /// submit, so the loop owns the authoritative list).
    pub fn draw(&mut self, pending_images: usize) -> Result<()> {
        let revision = self.feed.lock().map(|feed| feed.revision()).unwrap_or(0);
        let busy = self.feed.lock().map(|feed| feed.busy()).unwrap_or(false);
        // While a turn runs, advance the spinner and repaint even when the feed
        // is unchanged, so the header visibly animates instead of freezing.
        if busy {
            self.spinner = self.spinner.wrapping_add(1);
        }
        if revision == self.last_revision && !self.dirty && !busy {
            return Ok(());
        }
        self.last_revision = revision;
        self.dirty = false;
        let feed = self.feed.clone();
        let input = self.input.clone();
        let cursor = self.cursor;
        let scroll = self.scroll;
        let spinner = self.spinner;
        let queued = self.submitted.clone();
        let state = FrameState {
            feed: &feed,
            input: &input,
            cursor,
            scroll,
            pending_images,
            spinner,
            queued: &queued,
        };
        self.terminal
            .draw(|frame| render(frame, &state))
            .context("drawing frame")?;
        Ok(())
    }

    /// Block for one key event and apply it.
    pub fn handle_event(&mut self) -> Result<Option<TuiOutcome>> {
        let event = event::read().context("reading terminal event")?;
        match event {
            Event::Key(key) if key.kind == KeyEventKind::Press => self.handle_key(key),
            Event::Paste(text) => {
                // An empty bracketed paste means the clipboard held an image,
                // not text: attach it instead of inserting nothing.
                if text.trim().is_empty() {
                    self.attach_clipboard_image();
                } else {
                    self.input
                        .insert_str(self.cursor, &text.replace(['\r', '\n'], " "));
                    self.cursor += text.len();
                }
                self.dirty = true;
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> Result<Option<TuiOutcome>> {
        // An approval prompt owns the keyboard while it is pending.
        if self.pending_approval() {
            return self.handle_approval_key(key);
        }
        match key.code {
            KeyCode::Enter => {
                let line = std::mem::take(&mut self.input);
                self.cursor = 0;
                self.dirty = true;
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    return Ok(None);
                }
                self.history.push(&trimmed);
                self.history.save();
                self.submitted.push(trimmed);
                Ok(Some(TuiOutcome::Prompt))
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.quit = true;
                Ok(Some(TuiOutcome::Exit))
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if self.input.is_empty() {
                    self.quit = true;
                    Ok(Some(TuiOutcome::Exit))
                } else {
                    Ok(None)
                }
            }
            KeyCode::Char('v') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.attach_clipboard_image();
                self.dirty = true;
                Ok(None)
            }
            // Shift+Tab cycles build ↔ plan. Refused mid-turn: switching rebuilds
            // the system prompt and tool block, which would corrupt a running
            // request.
            KeyCode::BackTab => {
                if self.feed.lock().map(|feed| feed.busy()).unwrap_or(false) {
                    if let Ok(mut feed) = self.feed.lock() {
                        feed.push(FeedEntry::Notice {
                            level: NoticeLevel::Warning,
                            text: "cannot switch mode while a turn is running".to_string(),
                        });
                    }
                    self.dirty = true;
                    return Ok(None);
                }
                Ok(Some(TuiOutcome::ToggleMode))
            }
            KeyCode::Char(ch) => {
                self.input.insert(self.cursor, ch);
                self.cursor += ch.len_utf8();
                self.dirty = true;
                Ok(None)
            }
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    let previous = self.input[..self.cursor]
                        .char_indices()
                        .last()
                        .map(|(index, _)| index)
                        .unwrap_or(0);
                    self.input.remove(previous);
                    self.cursor = previous;
                    self.dirty = true;
                }
                Ok(None)
            }
            KeyCode::Delete => {
                if self.cursor < self.input.len() {
                    self.input.remove(self.cursor);
                    self.dirty = true;
                }
                Ok(None)
            }
            KeyCode::Left => {
                if self.cursor > 0 {
                    self.cursor = self.input[..self.cursor]
                        .char_indices()
                        .last()
                        .map(|(index, _)| index)
                        .unwrap_or(0);
                    self.dirty = true;
                }
                Ok(None)
            }
            KeyCode::Right => {
                if self.cursor < self.input.len() {
                    self.cursor += self.input[self.cursor..]
                        .chars()
                        .next()
                        .map(char::len_utf8)
                        .unwrap_or(1);
                    self.dirty = true;
                }
                Ok(None)
            }
            KeyCode::Home => {
                // Jump to the oldest lines when the prompt box is empty; move
                // the caret to the start of the line while composing.
                if self.input.is_empty() {
                    self.scroll = self.max_scroll();
                } else {
                    self.cursor = 0;
                }
                self.dirty = true;
                Ok(None)
            }
            KeyCode::End => {
                // Jump back to the newest lines when the prompt box is empty;
                // move the caret to the end of the line while composing.
                if self.input.is_empty() {
                    self.scroll = 0;
                } else {
                    self.cursor = self.input.len();
                }
                self.dirty = true;
                Ok(None)
            }
            KeyCode::Up => {
                self.recall_previous();
                Ok(None)
            }
            KeyCode::Down => {
                self.recall_next();
                Ok(None)
            }
            KeyCode::PageUp => {
                self.scroll = self.scroll.saturating_add(10).min(self.max_scroll());
                self.dirty = true;
                Ok(None)
            }
            KeyCode::PageDown => {
                self.scroll = self.scroll.saturating_sub(10);
                self.dirty = true;
                Ok(None)
            }
            KeyCode::Esc => {
                // While a turn runs, Esc interrupts it so the user can reprompt.
                // Otherwise it snaps the transcript back to the newest lines.
                if self.feed.lock().map(|feed| feed.busy()).unwrap_or(false) {
                    if let Ok(mut feed) = self.feed.lock() {
                        feed.request_interrupt();
                    }
                    // An interrupt abandons the queued prompts too: the user is
                    // stopping to reprompt, not to run the backlog.
                    self.clear_queued();
                } else {
                    self.scroll = 0;
                }
                self.dirty = true;
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    /// Capture the clipboard image and attach it to the next prompt.
    fn attach_clipboard_image(&mut self) {
        match crate::images::capture_clipboard() {
            Ok(Some(image)) => {
                let mime = image.mime.clone();
                self.pending_images.push(image);
                if let Ok(mut feed) = self.feed.lock() {
                    feed.push(FeedEntry::Image(format!(
                        "image attached · {} · {} pending",
                        mime,
                        self.pending_images.len()
                    )));
                }
            }
            Ok(None) => {
                if let Ok(mut feed) = self.feed.lock() {
                    feed.push(FeedEntry::Notice {
                        level: NoticeLevel::Warning,
                        text: "clipboard has no image (copy a screenshot or image first)"
                            .to_string(),
                    });
                }
            }
            Err(error) => {
                if let Ok(mut feed) = self.feed.lock() {
                    feed.push(FeedEntry::Notice {
                        level: NoticeLevel::Error,
                        text: format!("clipboard capture failed: {error:#}"),
                    });
                }
            }
        }
    }

    /// Drain images attached to the prompt being submitted.
    pub fn take_images(&mut self) -> Vec<rustyclippy_core::ImageAttachment> {
        std::mem::take(&mut self.pending_images)
    }

    /// Largest useful scroll offset: the rows above the visible window.
    ///
    /// `scroll` counts from the bottom, so this is the offset that shows the
    /// oldest lines. Computed from the live terminal size and the transcript
    /// contents so Home lands exactly at the top.
    fn max_scroll(&self) -> usize {
        let Ok(size) = self.terminal.size() else {
            return 0;
        };
        let entries = self
            .feed
            .lock()
            .map(|feed| feed.entries().to_vec())
            .unwrap_or_default();
        let lines = transcript_lines(&entries);
        // The transcript pane sits between a 5-row header, a 3-row prompt box
        // and a 1-row footer, and loses two columns/rows to its border.
        let width = size.width.saturating_sub(2) as usize;
        let visible = size
            .height
            .saturating_sub(HEADER_HEIGHT + INPUT_HEIGHT + FOOTER_HEIGHT + 2)
            as usize;
        wrapped_height(&lines, width).saturating_sub(visible)
    }

    /// Up: recall the previous prompt while composing, otherwise scroll.
    fn recall_previous(&mut self) {
        if up_recalls_history(&self.input, self.history.browsing()) {
            if let Some(line) = self.history.previous(&self.input) {
                self.input = line;
                self.cursor = self.input.len();
                self.dirty = true;
            }
        } else {
            self.scroll = self.scroll.saturating_add(1).min(self.max_scroll());
            self.dirty = true;
        }
    }

    /// Down: walk forward through history, restoring the draft at the end.
    fn recall_next(&mut self) {
        if self.history.browsing() {
            if let Some(line) = self.history.next() {
                self.input = line;
                self.cursor = self.input.len();
                self.dirty = true;
            }
        } else {
            self.scroll = self.scroll.saturating_sub(1);
            self.dirty = true;
        }
    }

    fn pending_approval(&self) -> bool {
        self.feed
            .lock()
            .map(|feed| feed.pending_approval().is_some())
            .unwrap_or(false)
    }

    fn handle_approval_key(&mut self, key: KeyEvent) -> Result<Option<TuiOutcome>> {
        let answer = match key.code {
            KeyCode::Enter | KeyCode::Char('y') | KeyCode::Char('Y') => Some(ApprovalAnswer::Yes),
            KeyCode::Char('n') | KeyCode::Char('N') => Some(ApprovalAnswer::No),
            KeyCode::Char('a') | KeyCode::Char('A') => Some(ApprovalAnswer::Session),
            KeyCode::Char('s') | KeyCode::Char('S') => Some(ApprovalAnswer::Select),
            KeyCode::Esc => Some(ApprovalAnswer::No),
            _ => None,
        };
        if let Some(answer) = answer {
            if let Ok(mut feed) = self.feed.lock() {
                feed.answer_approval(answer);
            }
        }
        Ok(None)
    }

    /// Drain the oldest submitted prompt, if any.
    pub fn take_submitted(&mut self) -> Option<String> {
        if self.submitted.is_empty() {
            return None;
        }
        Some(self.submitted.remove(0))
    }

    /// Drop every queued prompt, used when the user interrupts a turn.
    pub fn clear_queued(&mut self) {
        self.submitted.clear();
        self.dirty = true;
    }

    /// Drain a pending mode-switch request, if any.
    pub fn take_toggle_mode(&mut self) -> bool {
        std::mem::take(&mut self.toggle_mode)
    }

    pub fn quit(&self) -> bool {
        self.quit
    }

    /// Wait up to `timeout` for input; returns true when an event was handled.
    pub fn poll(&mut self, timeout: Duration) -> Result<bool> {
        if event::poll(timeout).context("polling terminal")? {
            if let Some(TuiOutcome::ToggleMode) = self.handle_event()? {
                self.toggle_mode = true;
            }
            return Ok(true);
        }
        Ok(false)
    }
}

/// Up recalls history while the user is composing or already browsing; with an
/// empty input box it scrolls the transcript instead.
fn up_recalls_history(input: &str, browsing: bool) -> bool {
    !input.is_empty() || browsing
}

/// Everything the renderer needs for one frame, so the signature stays small.
struct FrameState<'a> {
    feed: &'a Arc<Mutex<UiFeed>>,
    input: &'a str,
    cursor: usize,
    scroll: usize,
    pending_images: usize,
    spinner: usize,
    queued: &'a [String],
}

fn render(frame: &mut ratatui::Frame, state: &FrameState<'_>) {
    let FrameState {
        feed,
        input,
        cursor,
        scroll,
        pending_images,
        spinner,
        queued,
    } = state;
    let cursor = *cursor;
    let scroll = *scroll;
    let pending_images = *pending_images;
    let spinner = *spinner;
    let area = frame.area();

    let (header, entries, approval, busy, totals) = match feed.lock() {
        Ok(feed) => (
            feed.header().cloned(),
            feed.entries().to_vec(),
            feed.pending_approval().cloned(),
            feed.busy(),
            feed.totals().clone(),
        ),
        Err(_) => (None, Vec::new(), None, false, Default::default()),
    };

    // The input pane grows to fit an approval prompt (one line per call plus
    // the hint), so the keys are never clipped.
    let input_height = match &approval {
        Some(request) => {
            let body = if request.is_batch() {
                request.calls.len() + 2
            } else {
                2
            };
            (body as u16).saturating_add(2)
        }
        None => INPUT_HEIGHT,
    };

    // Queued prompts get their own pane directly above the prompt box, so they
    // stack at the bottom of the screen instead of crowding the input line.
    let queue_height = queued.len() as u16;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(HEADER_HEIGHT),
            Constraint::Min(3),
            Constraint::Length(queue_height),
            Constraint::Length(input_height),
            Constraint::Length(FOOTER_HEIGHT),
        ])
        .split(area);

    render_header(frame, chunks[0], header.as_ref(), busy, spinner);
    render_transcript(frame, chunks[1], &entries, scroll);
    render_queue(frame, chunks[2], queued);
    render_input(
        frame,
        chunks[3],
        input,
        cursor,
        approval.as_ref(),
        pending_images,
    );
    render_footer(frame, chunks[4], &totals);
}

/// Queued prompts, stacked above the prompt box.
fn render_queue(frame: &mut ratatui::Frame, area: Rect, queued: &[String]) {
    if queued.is_empty() || area.height == 0 {
        return;
    }
    let mut lines: Vec<Line<'static>> = Vec::new();
    for (index, prompt) in queued.iter().enumerate() {
        lines.push(Line::from(vec![
            Span::styled("⏳ ", Style::default().fg(MUTED)),
            Span::styled(
                format!("{} ", index + 1),
                Style::default().fg(MUTED).add_modifier(Modifier::DIM),
            ),
            Span::styled(prompt.clone(), Style::default().fg(MUTED)),
        ]));
    }
    frame.render_widget(Paragraph::new(Text::from(lines)), area);
}

fn render_header(
    frame: &mut ratatui::Frame,
    area: Rect,
    header: Option<&crate::ui::HeaderInfo>,
    busy: bool,
    spinner: usize,
) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(RAIL));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let Some(header) = header else {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                " rustyclippy-code",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ))),
            inner,
        );
        return;
    };

    let status = if busy {
        let glyph = crate::ui::SPINNER[spinner % crate::ui::SPINNER.len()];
        Span::styled(format!("{glyph} working"), Style::default().fg(WARN))
    } else {
        Span::styled("● idle", Style::default().fg(MUTED))
    };
    let title = Line::from(vec![
        Span::styled(
            " rustyclippy-code",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!(" {}", header.version), Style::default().fg(MUTED)),
        Span::raw("  "),
        status,
    ]);
    let detail = Line::from(vec![
        Span::styled(" model ", Style::default().fg(MUTED)),
        Span::styled(header.model.clone(), Style::default().fg(Color::White)),
        Span::styled("  mode ", Style::default().fg(MUTED)),
        Span::styled(header.mode.clone(), Style::default().fg(USER)),
        Span::styled("  session ", Style::default().fg(MUTED)),
        Span::styled(short(&header.session), Style::default().fg(Color::White)),
    ]);
    let cwd = Line::from(vec![
        Span::styled(" cwd ", Style::default().fg(MUTED)),
        Span::styled(header.cwd.clone(), Style::default().fg(MUTED)),
        Span::styled("  endpoint ", Style::default().fg(MUTED)),
        Span::styled(header.base_url.clone(), Style::default().fg(MUTED)),
    ]);
    frame.render_widget(Paragraph::new(Text::from(vec![title, detail, cwd])), inner);
}

fn render_transcript(frame: &mut ratatui::Frame, area: Rect, entries: &[FeedEntry], scroll: usize) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(RAIL))
        .title(Span::styled(" transcript ", Style::default().fg(MUTED)));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let lines = transcript_lines(entries);

    // A chat transcript grows downward, so pin the view to the newest lines and
    // let `scroll` walk backwards into history.
    let total = wrapped_height(&lines, inner.width as usize);
    let visible = inner.height as usize;
    let bottom = total.saturating_sub(visible);
    let offset = bottom.saturating_sub(scroll);
    let paragraph = Paragraph::new(Text::from(lines))
        .wrap(Wrap { trim: false })
        .scroll((offset.min(u16::MAX as usize) as u16, 0));
    frame.render_widget(paragraph, inner);
}

/// Number of terminal rows the lines occupy once wrapped to `width`.
///
/// Mirrors ratatui's word wrapping closely enough for scroll math: greedy
/// word packing, with over-long words split across rows.
fn wrapped_height(lines: &[Line<'_>], width: usize) -> usize {
    if width == 0 {
        return 0;
    }
    let mut rows = 0usize;
    for line in lines {
        let text = line.to_string();
        if text.is_empty() {
            rows += 1;
            continue;
        }
        let mut current = 0usize;
        let mut line_rows = 1usize;
        for word in text.split(' ') {
            let word_width = word.chars().count();
            if word_width > width {
                // Long word: split across as many rows as needed.
                let extra = word_width.div_ceil(width);
                line_rows += extra.saturating_sub(1);
                current = word_width % width;
                continue;
            }
            if current == 0 {
                current = word_width;
            } else if current + 1 + word_width <= width {
                current += 1 + word_width;
            } else {
                line_rows += 1;
                current = word_width;
            }
        }
        rows += line_rows;
    }
    rows
}

/// Turn feed entries into styled transcript lines.
fn transcript_lines(entries: &[FeedEntry]) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    for entry in entries {
        match entry {
            FeedEntry::User(text) => {
                lines.push(Line::from(vec![
                    Span::styled("▌ ", Style::default().fg(USER)),
                    Span::styled(
                        "you",
                        Style::default().fg(USER).add_modifier(Modifier::BOLD),
                    ),
                ]));
                for line in text.lines() {
                    lines.push(Line::from(vec![
                        Span::styled("│ ", Style::default().fg(RAIL)),
                        Span::raw(line.to_string()),
                    ]));
                }
                lines.push(Line::default());
            }
            FeedEntry::Text(chunk) => {
                // Streamed chunks arrive piecemeal; append to the last rail line
                // so a message reads as one block instead of one line per chunk.
                for piece in chunk.split_inclusive('\n') {
                    let body = piece.trim_end_matches('\n');
                    if body.is_empty() {
                        lines.push(Line::from(Span::styled("│", Style::default().fg(RAIL))));
                    } else {
                        lines.push(Line::from(vec![
                            Span::styled("│ ", Style::default().fg(RAIL)),
                            Span::raw(body.to_string()),
                        ]));
                    }
                }
            }
            FeedEntry::Reasoning(chunk) => {
                // Dimmed scratch work under a `⋯` rail, distinct from the answer.
                for piece in chunk.split_inclusive('\n') {
                    let body = piece.trim_end_matches('\n');
                    if body.is_empty() {
                        lines.push(Line::from(Span::styled("⋯", Style::default().fg(MUTED))));
                    } else {
                        lines.push(Line::from(vec![
                            Span::styled("⋯ ", Style::default().fg(MUTED)),
                            Span::styled(body.to_string(), Style::default().fg(MUTED)),
                        ]));
                    }
                }
            }
            FeedEntry::ToolCall { name, detail } => {
                lines.push(Line::from(vec![
                    Span::styled("  ● ", Style::default().fg(TOOL)),
                    Span::styled(
                        name.clone(),
                        Style::default().fg(TOOL).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(format!(" {detail}"), Style::default().fg(MUTED)),
                ]));
            }
            FeedEntry::ToolResult { name, ok, detail } => {
                let color = if *ok { TOOL } else { ERR };
                lines.push(Line::from(vec![
                    Span::styled("  └ ", Style::default().fg(color)),
                    Span::styled(name.clone(), Style::default().fg(MUTED)),
                    Span::styled(format!(" {detail}"), Style::default().fg(MUTED)),
                ]));
            }
            FeedEntry::Metrics(text) => {
                lines.push(Line::from(vec![
                    Span::styled("  · ", Style::default().fg(MUTED)),
                    Span::styled(text.clone(), Style::default().fg(MUTED)),
                ]));
            }
            FeedEntry::Compaction(text) => {
                lines.push(Line::from(vec![
                    Span::styled("  ⟳ ", Style::default().fg(WARN)),
                    Span::styled(text.clone(), Style::default().fg(WARN)),
                ]));
            }
            FeedEntry::Notice { level, text } => {
                let (marker, color) = match level {
                    NoticeLevel::Hint => ("·", MUTED),
                    NoticeLevel::Warning => ("!", WARN),
                    NoticeLevel::Error => ("✗", ERR),
                };
                for line in text.lines() {
                    lines.push(Line::from(vec![
                        Span::styled(format!("  {marker} "), Style::default().fg(color)),
                        Span::styled(line.to_string(), Style::default().fg(color)),
                    ]));
                }
            }
            FeedEntry::TurnSummary(text) => {
                lines.push(Line::from(Span::styled(
                    format!("── {text} ──"),
                    Style::default().fg(MUTED),
                )));
                lines.push(Line::default());
            }
            FeedEntry::ModeSwitch(text) => {
                lines.push(Line::from(vec![
                    Span::styled("⟳ ", Style::default().fg(WARN)),
                    Span::styled(text.clone(), Style::default().fg(WARN)),
                ]));
            }
            FeedEntry::Image(text) => {
                lines.push(Line::from(vec![
                    Span::styled("▌ ", Style::default().fg(ACCENT)),
                    Span::styled(text.clone(), Style::default().fg(ACCENT)),
                ]));
            }
        }
    }
    lines
}

fn render_input(
    frame: &mut ratatui::Frame,
    area: Rect,
    input: &str,
    cursor: usize,
    approval: Option<&crate::ui::ApprovalRequest>,
    pending_images: usize,
) {
    if let Some(request) = approval {
        render_approval(frame, area, request);
        return;
    }

    let title = if pending_images > 0 {
        format!(" prompt · {pending_images} image(s) ")
    } else {
        " prompt ".to_string()
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(ACCENT))
        .title(Span::styled(title, Style::default().fg(ACCENT)));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let line = Line::from(vec![
        Span::styled("› ", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        Span::raw(input.to_string()),
    ]);
    frame.render_widget(Paragraph::new(line), inner);

    // Place the real cursor so the terminal shows the caret.
    let prefix = 2u16;
    let offset = input[..cursor].chars().count() as u16;
    let x = inner.x.saturating_add(prefix).saturating_add(offset);
    let y = inner.y;
    if x < inner.x.saturating_add(inner.width) {
        frame.set_cursor_position((x, y));
    }
}

fn render_approval(frame: &mut ratatui::Frame, area: Rect, request: &crate::ui::ApprovalRequest) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(WARN))
        .title(Span::styled(" approval needed ", Style::default().fg(WARN)));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines = Vec::new();
    if request.is_batch() {
        lines.push(Line::from(Span::styled(
            format!("{} tools need approval:", request.calls.len()),
            Style::default().fg(WARN),
        )));
        for (index, (name, detail)) in request.calls.iter().enumerate() {
            lines.push(Line::from(vec![
                Span::styled(format!("  {}. ", index + 1), Style::default().fg(MUTED)),
                Span::styled(name.clone(), Style::default().fg(TOOL)),
                Span::styled(format!(" {detail}"), Style::default().fg(MUTED)),
            ]));
        }
    } else if let Some((name, detail)) = request.calls.first() {
        lines.push(Line::from(vec![
            Span::styled("allow ", Style::default().fg(WARN)),
            Span::styled(name.clone(), Style::default().fg(TOOL)),
            Span::styled(format!(" {detail}"), Style::default().fg(MUTED)),
        ]));
    }
    let hint = if request.is_batch() {
        "[Y] allow all · [n] deny · [a] allow for session · [s] choose each"
    } else {
        "[Y] allow · [n] deny · [a] allow for session"
    };
    lines.push(Line::from(Span::styled(hint, Style::default().fg(MUTED))));
    frame.render_widget(Paragraph::new(Text::from(lines)), inner);
}

fn render_footer(frame: &mut ratatui::Frame, area: Rect, totals: &crate::ui::Totals) {
    // Row 1: token accounting and decode speed.
    let mut stats = vec![
        Span::styled(" turns ", Style::default().fg(MUTED)),
        Span::styled(totals.turns.to_string(), Style::default().fg(Color::White)),
        Span::styled("  prompt ", Style::default().fg(MUTED)),
        Span::styled(
            crate::ui::fmt_num(totals.prompt),
            Style::default().fg(Color::White),
        ),
        Span::styled("  cached ", Style::default().fg(MUTED)),
        Span::styled(
            format!("{:.1}%", totals.hit_ratio() * 100.0),
            Style::default().fg(TOOL),
        ),
        Span::styled("  out ", Style::default().fg(MUTED)),
        Span::styled(
            crate::ui::fmt_num(totals.completion),
            Style::default().fg(Color::White),
        ),
    ];
    if let Some(rate) = totals.decode_tokens_per_second {
        stats.push(Span::styled("  ", Style::default().fg(MUTED)));
        stats.push(Span::styled(
            format!("{rate:.1} tok/s"),
            Style::default().fg(ACCENT),
        ));
    }

    // Row 2: how close the next request is to triggering compaction, plus keys.
    let mut gauge = Vec::new();
    if totals.compaction_budget > 0 {
        let ratio = totals.compaction_ratio();
        let color = if ratio >= 0.9 {
            ERR
        } else if ratio >= 0.75 {
            WARN
        } else {
            TOOL
        };
        gauge.push(Span::styled(" ctx ", Style::default().fg(MUTED)));
        gauge.push(Span::styled(
            format!(
                "{} / {} ({:.0}%)",
                crate::ui::fmt_num(totals.context_tokens),
                crate::ui::fmt_num(totals.compaction_budget),
                ratio * 100.0
            ),
            Style::default().fg(color),
        ));
        gauge.push(Span::styled("  ", Style::default().fg(MUTED)));
    }
    gauge.push(Span::styled(
        "⇥ mode · esc stop · ^C quit",
        Style::default().fg(MUTED),
    ));

    frame.render_widget(
        Paragraph::new(Text::from(vec![Line::from(stats), Line::from(gauge)])),
        area,
    );
}

fn short(id: &str) -> String {
    if id.chars().count() <= 12 {
        return id.to_string();
    }
    id.chars().take(12).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_renders_user_and_text() {
        let entries = vec![
            FeedEntry::User("hello".into()),
            FeedEntry::Text("world\n".into()),
        ];
        let lines = transcript_lines(&entries);
        assert!(lines.iter().any(|line| line.to_string().contains("you")));
        assert!(lines.iter().any(|line| line.to_string().contains("world")));
    }

    #[test]
    fn transcript_marks_failed_tools() {
        let entries = vec![FeedEntry::ToolResult {
            name: "bash".into(),
            ok: false,
            detail: "boom".into(),
        }];
        let lines = transcript_lines(&entries);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].to_string().contains("boom"));
    }

    #[test]
    fn short_truncates_long_ids() {
        assert_eq!(short("abc"), "abc");
        assert_eq!(short("0123456789abcdef").chars().count(), 12);
    }

    /// Render a frame into an offscreen buffer and return it as text rows.
    fn render_to_rows_scrolled(
        feed: &Arc<Mutex<UiFeed>>,
        input: &str,
        cursor: usize,
        scroll: usize,
    ) -> Vec<String> {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                render(
                    frame,
                    &FrameState {
                        feed,
                        input,
                        cursor,
                        scroll,
                        pending_images: 0,
                        spinner: 0,
                        queued: &[],
                    },
                )
            })
            .unwrap();
        let buffer = terminal.backend().buffer();
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect()
    }

    fn render_to_rows(feed: &Arc<Mutex<UiFeed>>, input: &str, cursor: usize) -> Vec<String> {
        render_to_rows_queued(feed, input, cursor, &[])
    }

    fn render_to_rows_queued(
        feed: &Arc<Mutex<UiFeed>>,
        input: &str,
        cursor: usize,
        queued: &[String],
    ) -> Vec<String> {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                render(
                    frame,
                    &FrameState {
                        feed,
                        input,
                        cursor,
                        scroll: 0,
                        pending_images: 0,
                        spinner: 0,
                        queued,
                    },
                )
            })
            .unwrap();
        let buffer = terminal.backend().buffer();
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect()
    }

    #[test]
    fn transcript_renders_reasoning_dimmed() {
        let entries = vec![FeedEntry::Reasoning("weighing options\n".into())];
        let lines = transcript_lines(&entries);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].to_string().contains("weighing options"));
        assert!(lines[0].to_string().starts_with("⋯"));
    }

    #[test]
    fn reasoning_is_on_by_default_and_toggleable() {
        let feed = Arc::new(Mutex::new(UiFeed::default()));
        let output = crate::ui::Output::tui(feed.clone());
        assert!(output.show_reasoning());

        output.reasoning("secret thoughts");
        let entries = feed.lock().unwrap().entries().to_vec();
        assert!(matches!(entries.as_slice(), [FeedEntry::Reasoning(_)]));

        assert!(!output.set_show_reasoning(false));
        output.reasoning("more thoughts");
        assert_eq!(feed.lock().unwrap().entries().len(), 1);
    }

    #[test]
    fn up_recalls_history_only_while_composing() {
        assert!(up_recalls_history("typing", false));
        assert!(up_recalls_history("", true));
        assert!(!up_recalls_history("", false));
    }

    #[test]
    fn header_spinner_animates_while_busy() {
        let feed = Arc::new(Mutex::new(UiFeed::default()));
        {
            let mut feed = feed.lock().unwrap();
            feed.set_header(crate::ui::HeaderInfo {
                version: "0.1.0".into(),
                session: "sess".into(),
                model: "m".into(),
                mode: "build".into(),
                base_url: "http://x".into(),
                cwd: "/tmp".into(),
            });
            feed.set_busy(true);
        }
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let mut frames = Vec::new();
        for spinner in 0..3 {
            let backend = TestBackend::new(80, 24);
            let mut terminal = Terminal::new(backend).unwrap();
            terminal
                .draw(|frame| {
                    render(
                        frame,
                        &FrameState {
                            feed: &feed,
                            input: "",
                            cursor: 0,
                            scroll: 0,
                            pending_images: 0,
                            spinner,
                            queued: &[],
                        },
                    )
                })
                .unwrap();
            let buffer = terminal.backend().buffer();
            frames.push(
                (0..buffer.area.height)
                    .map(|y| {
                        (0..buffer.area.width)
                            .map(|x| buffer[(x, y)].symbol().to_string())
                            .collect::<String>()
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }
        assert!(frames[0].contains("working"), "busy status missing:\n{}", frames[0]);
        assert_ne!(frames[0], frames[1], "spinner frame did not advance");
        assert_ne!(frames[1], frames[2], "spinner frame did not advance");
    }

    #[test]
    fn prompt_title_shows_pending_images() {
        let feed = Arc::new(Mutex::new(UiFeed::default()));
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                render(
                    frame,
                    &FrameState {
                        feed: &feed,
                        input: "",
                        cursor: 0,
                        scroll: 0,
                        pending_images: 2,
                        spinner: 0,
                        queued: &[],
                    },
                )
            })
            .unwrap();
        let buffer = terminal.backend().buffer();
        let screen: String = (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            screen.contains("2 image(s)"),
            "pending images missing:\n{screen}"
        );
    }

    #[test]
    fn frame_shows_header_transcript_and_input() {
        let feed = Arc::new(Mutex::new(UiFeed::default()));
        {
            let mut feed = feed.lock().unwrap();
            feed.set_header(crate::ui::HeaderInfo {
                version: "0.1.0".into(),
                session: "sess-0123456789abcdef".into(),
                model: "deepseek-v4".into(),
                mode: "build".into(),
                base_url: "http://localhost:8080".into(),
                cwd: "/tmp/project".into(),
            });
            feed.push(FeedEntry::User("explain the cache".into()));
            feed.push(FeedEntry::Text("The prefix stays stable.\n".into()));
            feed.push(FeedEntry::ToolCall {
                name: "bash".into(),
                detail: "ls -la".into(),
            });
            feed.push(FeedEntry::ToolResult {
                name: "bash".into(),
                ok: true,
                detail: "src".into(),
            });
            feed.add_turn(1200, 900, 300, 40);
        }
        let rows = render_to_rows(&feed, "next question", 13);
        let screen = rows.join("\n");

        assert!(screen.contains("rustyclippy-code"), "header missing:\n{screen}");
        assert!(screen.contains("deepseek-v4"), "model missing:\n{screen}");
        assert!(screen.contains("build"), "mode missing:\n{screen}");
        assert!(screen.contains("explain the cache"), "user prompt missing:\n{screen}");
        assert!(screen.contains("prefix stays stable"), "assistant text missing:\n{screen}");
        assert!(screen.contains("ls -la"), "tool call missing:\n{screen}");
        assert!(screen.contains("next question"), "input missing:\n{screen}");
        assert!(screen.contains("1,200"), "totals missing:\n{screen}");
        assert!(screen.contains("75.0%"), "cache ratio missing:\n{screen}");
    }

    #[test]
    fn frame_shows_approval_prompt() {
        let feed = Arc::new(Mutex::new(UiFeed::default()));
        {
            let mut feed = feed.lock().unwrap();
            let (tx, _rx) = tokio::sync::oneshot::channel();
            feed.request_approval(
                crate::ui::ApprovalRequest {
                    calls: vec![("edit".into(), "src/main.rs".into())],
                },
                tx,
            );
        }
        let rows = render_to_rows(&feed, "", 0);
        let screen = rows.join("\n");
        assert!(screen.contains("approval needed"), "approval box missing:\n{screen}");
        assert!(screen.contains("src/main.rs"), "call detail missing:\n{screen}");
        assert!(screen.contains("[Y] allow"), "key hints missing:\n{screen}");
    }

    #[test]
    fn frame_shows_batch_approval() {
        let feed = Arc::new(Mutex::new(UiFeed::default()));
        {
            let mut feed = feed.lock().unwrap();
            let (tx, _rx) = tokio::sync::oneshot::channel();
            feed.request_approval(
                crate::ui::ApprovalRequest {
                    calls: vec![
                        ("edit".into(), "a.rs".into()),
                        ("bash".into(), "cargo build".into()),
                    ],
                },
                tx,
            );
        }
        let screen = render_to_rows(&feed, "", 0).join("\n");
        assert!(screen.contains("2 tools need approval"), "batch header missing:\n{screen}");
        assert!(screen.contains("[s] choose each"), "batch hints missing:\n{screen}");
    }

    #[test]
    fn answering_approval_clears_the_request() {
        let mut feed = UiFeed::default();
        let (tx, rx) = tokio::sync::oneshot::channel();
        feed.request_approval(
            crate::ui::ApprovalRequest {
                calls: vec![("edit".into(), "a.rs".into())],
            },
            tx,
        );
        assert!(feed.pending_approval().is_some());
        assert!(feed.answer_approval(ApprovalAnswer::Session));
        assert!(feed.pending_approval().is_none());
        assert_eq!(rx.blocking_recv().unwrap(), ApprovalAnswer::Session);
    }

    #[test]
    fn wrapped_height_counts_greedy_rows() {
        let lines = vec![
            Line::from("short"),
            Line::from(""),
            Line::from("a b c d e f g h i j"),
        ];
        // "short" = 1 row, "" = 1 row, 19 chars at width 10 = 2 rows.
        assert_eq!(wrapped_height(&lines, 10), 4);
        assert_eq!(wrapped_height(&lines, 0), 0);
    }

    #[test]
    fn wrapped_height_splits_long_words() {
        let lines = vec![Line::from("x".repeat(25))];
        assert_eq!(wrapped_height(&lines, 10), 3);
    }

    #[test]
    fn newest_lines_stay_visible_when_transcript_overflows() {
        let feed = Arc::new(Mutex::new(UiFeed::default()));
        {
            let mut feed = feed.lock().unwrap();
            for index in 0..40 {
                feed.push(FeedEntry::Notice {
                    level: NoticeLevel::Hint,
                    text: format!("row {index}"),
                });
            }
        }
        let screen = render_to_rows(&feed, "", 0).join("\n");
        // The transcript pane is ~14 rows tall, so early rows scroll off and the
        // last row must still be on screen.
        assert!(screen.contains("row 39"), "newest row missing:\n{screen}");
        assert!(!screen.contains("row 0\n"), "oldest row should be scrolled off:\n{screen}");
    }

    #[test]
    fn scrolling_to_the_top_reveals_the_oldest_row() {
        let feed = Arc::new(Mutex::new(UiFeed::default()));
        {
            let mut feed = feed.lock().unwrap();
            for index in 0..40 {
                feed.push(FeedEntry::Notice {
                    level: NoticeLevel::Hint,
                    text: format!("row {index}"),
                });
            }
        }
        // 24 rows tall: header 5 + prompt 3 + footer 1 leaves 15 for the pane,
        // minus its two border rows = 13 visible transcript rows.
        let visible = 24 - HEADER_HEIGHT - INPUT_HEIGHT - FOOTER_HEIGHT - 2;
        let max_scroll = 40usize.saturating_sub(visible as usize);
        assert!(max_scroll > 0, "fixture must overflow the pane");

        let top = render_to_rows_scrolled(&feed, "", 0, max_scroll).join("\n");
        assert!(top.contains("row 0"), "oldest row missing at top:\n{top}");
        assert!(
            !top.contains("row 39"),
            "newest row should be off screen:\n{top}"
        );

        let bottom = render_to_rows_scrolled(&feed, "", 0, 0).join("\n");
        assert!(
            bottom.contains("row 39"),
            "newest row missing at bottom:\n{bottom}"
        );
    }

    #[test]
    fn busy_state_flips_with_turn() {
        let mut feed = UiFeed::default();
        assert!(!feed.busy());
        feed.set_busy(true);
        assert!(feed.busy());
        feed.set_busy(false);
        assert!(!feed.busy());
    }

    #[test]
    fn queued_prompts_stack_above_the_input() {
        let feed = Arc::new(Mutex::new(UiFeed::default()));
        let queued = vec![
            "first follow-up".to_string(),
            "second follow-up".to_string(),
        ];
        let rows = render_to_rows_queued(&feed, "typing", 6, &queued);
        let screen = rows.join("\n");
        assert!(
            screen.contains("first follow-up"),
            "queued prompt missing:\n{screen}"
        );
        assert!(
            screen.contains("second follow-up"),
            "queued prompt missing:\n{screen}"
        );
        assert!(screen.contains("typing"), "live input missing:\n{screen}");

        // The queue pane sits above the prompt box, so both queued lines must
        // appear on rows before the input line.
        let input_row = rows
            .iter()
            .position(|row| row.contains("typing"))
            .expect("input row");
        let first_row = rows
            .iter()
            .position(|row| row.contains("first follow-up"))
            .expect("first queued row");
        let second_row = rows
            .iter()
            .position(|row| row.contains("second follow-up"))
            .expect("second queued row");
        assert!(first_row < input_row, "queue must sit above the input");
        assert!(second_row < input_row, "queue must sit above the input");
        assert!(first_row < second_row, "queue must keep submission order");
    }

    #[test]
    fn submitted_prompts_queue_in_order() {
        let mut queue: Vec<String> = Vec::new();
        queue.push("one".to_string());
        queue.push("two".to_string());
        // Draining takes the oldest first, so a prompt typed mid-turn is not
        // lost behind a later one.
        assert_eq!(queue.remove(0), "one");
        assert_eq!(queue.remove(0), "two");
        assert!(queue.is_empty());
    }

    #[test]
    fn footer_shows_context_gauge_and_decode_speed() {
        let feed = Arc::new(Mutex::new(UiFeed::default()));
        {
            let mut feed = feed.lock().unwrap();
            feed.set_context(85_000, 250_000, Some(42.5));
        }
        let screen = render_to_rows(&feed, "", 0).join("\n");
        assert!(screen.contains("ctx"), "context gauge missing:\n{screen}");
        assert!(
            screen.contains("85,000"),
            "context tokens missing:\n{screen}"
        );
        assert!(screen.contains("250,000"), "budget missing:\n{screen}");
        assert!(screen.contains("34%"), "ratio missing:\n{screen}");
        assert!(
            screen.contains("42.5 tok/s"),
            "decode speed missing:\n{screen}"
        );
    }

    #[test]
    fn footer_omits_gauge_and_speed_when_unknown() {
        let feed = Arc::new(Mutex::new(UiFeed::default()));
        let screen = render_to_rows(&feed, "", 0).join("\n");
        assert!(!screen.contains("ctx"), "gauge should be hidden:\n{screen}");
        assert!(!screen.contains("tok/s"), "speed should be hidden:\n{screen}");
    }

    #[test]
    fn compaction_ratio_clamps_and_colors_by_pressure() {
        let mut totals = crate::ui::Totals::default();
        assert_eq!(totals.compaction_ratio(), 0.0);
        totals.compaction_budget = 100;
        totals.context_tokens = 50;
        assert!((totals.compaction_ratio() - 0.5).abs() < 1e-9);
        totals.context_tokens = 250;
        assert_eq!(totals.compaction_ratio(), 1.0, "ratio must clamp to 1.0");
    }

    #[test]
    fn footer_advertises_mode_and_interrupt_keys() {
        let feed = Arc::new(Mutex::new(UiFeed::default()));
        let screen = render_to_rows(&feed, "", 0).join("\n");
        assert!(screen.contains("mode"), "mode hint missing:\n{screen}");
        assert!(
            screen.contains("esc stop"),
            "interrupt hint missing:\n{screen}"
        );
    }

    #[test]
    fn interrupt_flag_round_trips() {
        let mut feed = UiFeed::default();
        assert!(!feed.interrupt_requested());
        feed.request_interrupt();
        assert!(feed.interrupt_requested());
        feed.clear_interrupt();
        assert!(!feed.interrupt_requested());
    }

    #[test]
    fn interrupt_is_visible_through_output() {
        let feed = Arc::new(Mutex::new(UiFeed::default()));
        let output = crate::ui::Output::tui(feed.clone());
        assert!(!output.interrupt_requested());
        feed.lock().unwrap().request_interrupt();
        assert!(output.interrupt_requested());
        output.clear_interrupt();
        assert!(!output.interrupt_requested());
    }
}

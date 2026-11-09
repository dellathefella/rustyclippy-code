//! Prefix-anchored compaction.
//!
//! Compaction is the only operation that rewrites history, and naive
//! implementations throw away the entire cached prefix. This module chooses a
//! cut point that **preserves the pinned head** (environment + original task),
//! so the KV cache for the head survives and only the summary + recent tail
//! need prefill.
//!
//! Layout before compaction:
//!
//! ```text
//! [ head (pinned) ][ middle (summarized) ][ tail (kept verbatim) ]
//! ```
//!
//! Layout after compaction:
//!
//! ```text
//! [ head (byte-identical) ][ summary ][ tail ]
//! ```
//!
//! The head bytes never change, which is what makes reuse possible.

use serde::{Deserialize, Serialize};

use crate::message::{Message, Role};
use crate::token::TokenEstimator;

/// Marker that identifies a compaction summary message.
pub const SUMMARY_TAG: &str = "<compaction-summary>";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionRecord {
    /// Number of pinned head messages retained byte-for-byte.
    pub head_len: usize,
    /// First retained tail message in the pre-compaction transcript.
    pub tail_start: usize,
    /// Number of messages folded into the summary.
    pub dropped: usize,
    pub summary_tokens: usize,
    pub before_tokens: usize,
    pub after_tokens: usize,
    pub at_turn: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionPlan {
    pub head_len: usize,
    /// First retained tail message.
    pub tail_start: usize,
    pub dropped: usize,
}

/// Decide whether and where to compact.
///
/// Returns `None` when the transcript is short enough, when the pinned head
/// already reaches the tail window, or when the cut would not free enough
/// tokens to be worth a rewrite.
pub fn plan_compaction(
    messages: &[Message],
    head_len: usize,
    keep_tail: usize,
    estimator: &TokenEstimator,
    min_gain_tokens: usize,
) -> Option<CompactionPlan> {
    let head_len = head_len.min(messages.len());
    if messages.len() <= head_len + keep_tail {
        return None;
    }

    // Start at the tail window, then advance past tool results so the tail
    // never begins with an orphan `tool` message (which would be an invalid
    // wire sequence).
    let mut tail_start = messages.len() - keep_tail;
    while tail_start < messages.len() && messages[tail_start].role == Role::Tool {
        tail_start += 1;
    }

    if tail_start <= head_len {
        return None;
    }

    let dropped = tail_start - head_len;
    let dropped_tokens: usize = messages[head_len..tail_start]
        .iter()
        .map(|message| estimator.estimate_message(message))
        .sum();

    if dropped_tokens < min_gain_tokens {
        return None;
    }

    Some(CompactionPlan {
        head_len,
        tail_start,
        dropped,
    })
}

/// Build the summary message inserted after the pinned head.
pub fn summary_message(summary: &str) -> Message {
    Message::user(format!(
        "{SUMMARY_TAG}\nEarlier conversation was compacted to save context. \
         Summary of what was dropped:\n\n{summary}\n\n{SUMMARY_TAG}"
    ))
}

/// The instruction sent to the model to produce the summary.
pub fn summary_instruction() -> String {
    "Summarize the conversation above so work can continue without the full history.\n\
     Preserve, in this order:\n\
     1. The user's goal and any constraints they stated.\n\
     2. Decisions made and why.\n\
     3. Files created or modified, with paths.\n\
     4. Commands run and their relevant results.\n\
     5. Errors encountered and how they were resolved.\n\
     6. Open questions and the next concrete step.\n\
     Be concise and factual. Do not invent details. Output only the summary."
        .to_string()
}

/// True when a message is a compaction summary.
pub fn is_summary(message: &Message) -> bool {
    message.role == Role::User && message.content.starts_with(SUMMARY_TAG)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conversation() -> Vec<Message> {
        vec![
            Message::user("environment"),
            Message::user("the task"),
            Message::assistant("step 1"),
            Message::assistant("step 2"),
            Message::assistant("step 3"),
            Message::assistant("step 4"),
            Message::assistant("step 5"),
        ]
    }

    #[test]
    fn short_transcripts_are_not_compacted() {
        let estimator = TokenEstimator::default();
        let messages = vec![Message::user("a"), Message::user("b")];
        assert!(plan_compaction(&messages, 1, 4, &estimator, 1).is_none());
    }

    #[test]
    fn compaction_keeps_head_and_tail() {
        let estimator = TokenEstimator::default();
        let messages = conversation();
        let plan = plan_compaction(&messages, 2, 3, &estimator, 1).expect("plan");
        assert_eq!(plan.head_len, 2);
        assert_eq!(plan.tail_start, 4);
        assert_eq!(plan.dropped, 2);
        // head + summary + tail
        assert_eq!(plan.head_len + 1 + (messages.len() - plan.tail_start), 6);
    }

    #[test]
    fn tail_never_starts_with_tool_result() {
        let estimator = TokenEstimator::default();
        let mut messages = conversation();
        messages.push(Message::tool("output", "call_1"));
        messages.push(Message::tool("output2", "call_2"));
        let plan = plan_compaction(&messages, 2, 2, &estimator, 1).expect("plan");
        assert!(plan.tail_start <= messages.len());
        if plan.tail_start < messages.len() {
            assert_ne!(messages[plan.tail_start].role, Role::Tool);
        }
    }

    #[test]
    fn summary_message_is_tagged() {
        let message = summary_message("did things");
        assert!(is_summary(&message));
        assert!(message.content.contains("did things"));
    }
}

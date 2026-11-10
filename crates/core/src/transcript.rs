//! Append-only transcript with prefix-anchored compaction.
//!
//! The API deliberately exposes no way to mutate an existing message. Callers
//! can only [`append`](Transcript::append) or [`compact`](Transcript::compact),
//! and compaction preserves the pinned head byte-for-byte.

use crate::compaction::{CompactionPlan, CompactionRecord};
use crate::message::{Message, Role};
use crate::token::TokenEstimator;

#[derive(Debug, Clone)]
pub struct Transcript {
    messages: Vec<Message>,
    head_len: usize,
    compactions: Vec<CompactionRecord>,
}

impl Transcript {
    pub fn new(head_len: usize) -> Self {
        Self {
            messages: Vec::new(),
            head_len,
            compactions: Vec::new(),
        }
    }

    pub fn from_messages(messages: Vec<Message>, head_len: usize) -> Self {
        Self {
            messages,
            head_len,
            compactions: Vec::new(),
        }
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    pub fn head_len(&self) -> usize {
        self.head_len
    }

    pub fn compactions(&self) -> &[CompactionRecord] {
        &self.compactions
    }

    pub fn last_compaction(&self) -> Option<&CompactionRecord> {
        self.compactions.last()
    }

    /// Drop messages appended after `len`, used to roll back an interrupted
    /// turn. The pinned head and compaction summaries are never removed: an
    /// interrupt can only discard messages the current turn added.
    pub fn truncate(&mut self, len: usize) {
        let floor = self.head_len.min(self.messages.len());
        let keep = len.max(floor).min(self.messages.len());
        self.messages.truncate(keep);
    }

    /// Append one message. This is the only growth operation.
    pub fn append(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Apply a compaction plan: keep the pinned head, insert the summary, keep
    /// the tail verbatim.
    pub fn compact(
        &mut self,
        plan: &CompactionPlan,
        summary: Message,
        at_turn: usize,
        estimator: &TokenEstimator,
    ) -> CompactionRecord {
        let head_len = plan.head_len.min(self.messages.len());
        let tail_start = plan.tail_start.min(self.messages.len());

        let before_tokens: usize = self.messages[head_len..]
            .iter()
            .map(|message| estimator.estimate_message(message))
            .sum();
        let summary_tokens = estimator.estimate_message(&summary);

        let mut rebuilt: Vec<Message> = self.messages[..head_len].to_vec();
        rebuilt.push(summary);
        rebuilt.extend_from_slice(&self.messages[tail_start..]);

        let after_tokens: usize = rebuilt[head_len..]
            .iter()
            .map(|message| estimator.estimate_message(message))
            .sum();

        let record = CompactionRecord {
            head_len,
            tail_start,
            dropped: tail_start.saturating_sub(head_len),
            summary_tokens,
            before_tokens,
            after_tokens,
            at_turn,
        };

        self.messages = rebuilt;
        self.compactions.push(record.clone());
        record
    }

    /// Re-apply a recorded compaction during session replay.
    ///
    /// Uses the stored `head_len` / `tail_start` so the rebuilt transcript is
    /// identical to the one that existed when the compaction happened.
    pub fn apply_compaction(&mut self, record: &CompactionRecord, summary: Message) {
        let head_len = record.head_len.min(self.messages.len());
        let tail_start = record.tail_start.min(self.messages.len());
        let mut rebuilt: Vec<Message> = self.messages[..head_len].to_vec();
        rebuilt.push(summary);
        rebuilt.extend_from_slice(&self.messages[tail_start..]);
        self.messages = rebuilt;
        self.compactions.push(record.clone());
    }

    /// Total estimated tokens of the transcript (excluding system/tools).
    pub fn estimated_tokens(&self, estimator: &TokenEstimator) -> usize {
        self.messages
            .iter()
            .map(|message| estimator.estimate_message(message))
            .sum()
    }

    /// True when the transcript starts with the pinned environment message.
    pub fn has_environment(&self) -> bool {
        self.messages
            .first()
            .map(|message| {
                message.role == Role::System && message.content.starts_with("<environment>")
            })
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::plan_compaction;

    fn build_transcript() -> Transcript {
        let mut transcript = Transcript::new(2);
        transcript.append(Message::user("<environment>\ncwd: /tmp\n</environment>"));
        transcript.append(Message::user("the task"));
        for index in 0..6 {
            transcript.append(Message::assistant(format!("step {index}")));
        }
        transcript
    }

    #[test]
    fn truncate_never_removes_the_pinned_head() {
        let mut transcript = build_transcript();
        assert_eq!(transcript.len(), 8);
        transcript.truncate(4);
        assert_eq!(transcript.len(), 4);
        transcript.truncate(0);
        assert_eq!(transcript.len(), 2, "head must survive a rollback");
        transcript.truncate(99);
        assert_eq!(transcript.len(), 2, "truncate cannot grow the transcript");
    }

    #[test]
    fn append_only_growth() {
        let mut transcript = Transcript::new(1);
        transcript.append(Message::user("a"));
        transcript.append(Message::assistant("b"));
        assert_eq!(transcript.len(), 2);
    }

    #[test]
    fn compaction_preserves_head_bytes() {
        let mut transcript = build_transcript();
        let head_before: Vec<String> = transcript.messages()[..2]
            .iter()
            .map(|message| message.wire_hash())
            .collect();

        let estimator = TokenEstimator::default();
        let plan = plan_compaction(transcript.messages(), 2, 3, &estimator, 1).expect("plan");
        let record = transcript.compact(&plan, Message::user("summary"), 1, &estimator);

        let head_after: Vec<String> = transcript.messages()[..2]
            .iter()
            .map(|message| message.wire_hash())
            .collect();

        assert_eq!(head_before, head_after);
        assert_eq!(record.head_len, 2);
        assert!(transcript.len() < 8);
        assert_eq!(transcript.compactions().len(), 1);
    }

    #[test]
    fn compaction_keeps_tail_verbatim() {
        let mut transcript = build_transcript();
        let tail_hashes: Vec<String> = transcript.messages()[5..]
            .iter()
            .map(|message| message.wire_hash())
            .collect();
        let estimator = TokenEstimator::default();
        let plan = plan_compaction(transcript.messages(), 2, 3, &estimator, 1).expect("plan");
        transcript.compact(&plan, Message::user("summary"), 1, &estimator);
        let after_tail: Vec<String> = transcript.messages()[transcript.len() - tail_hashes.len()..]
            .iter()
            .map(|message| message.wire_hash())
            .collect();
        assert_eq!(tail_hashes, after_tail);
    }
}

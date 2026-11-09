//! Prefix-cache planning engine.
//!
//! ## Why this exists
//!
//! Local inference servers (MLX, llama.cpp, vLLM) reuse KV cache only when the
//! incoming prompt starts with *byte-identical* tokens. DeepSeek's hosted API
//! behaves the same way: automatic prefix caching in 64-token blocks.
//!
//! Most agent harnesses accidentally invalidate that cache every turn by
//! rewriting the system prompt (timestamps, cwd), reordering tool schemas,
//! trimming history, or re-rendering messages. The result is a full re-prefill
//! of the entire context — the single most expensive thing an agent can do on
//! a local model.
//!
//! This engine makes prefix stability a first-class, *measured* invariant:
//!
//! 1. The system prompt and tool schema bytes are **frozen** at session start.
//! 2. Messages are append-only (enforced by [`crate::transcript::Transcript`]).
//! 3. [`CacheEngine::plan`] hashes each message and finds the longest common
//!    prefix with the last **committed** request.
//! 4. [`CacheEngine::commit`] records what was actually sent. Planning is pure,
//!    so a request can be planned repeatedly without inflating reuse numbers.
//! 5. Unexpected divergence is surfaced as a hard guard so the agent can refuse
//!    to silently burn a full prefill.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::canonical::{canonical_bytes, hash_bytes, hash_pair, hash_str};
use crate::compaction::CompactionRecord;
use crate::message::Message;
use crate::token::{TokenEstimator, DEEPSEEK_CACHE_GRANULARITY};

/// A frozen prompt segment (system prompt or tool schema block).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenSegment {
    pub hash: String,
    pub bytes: usize,
    pub tokens: usize,
}

/// Per-segment accounting for one request.
#[derive(Debug, Clone, Serialize)]
pub struct SegmentInfo {
    pub name: String,
    pub hash: String,
    pub bytes: usize,
    pub tokens: usize,
    /// True when this segment is byte-identical to the previous request.
    pub cached: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BreakpointKind {
    /// End of the tool-schema block.
    Tools,
    /// End of the byte-stable message prefix.
    StablePrefix,
    /// End of the transcript (cache everything sent this turn).
    TranscriptEnd,
}

/// A cache breakpoint, expressed as a position in the wire prompt.
#[derive(Debug, Clone, Serialize)]
pub struct Breakpoint {
    pub kind: BreakpointKind,
    /// Index of the message after which the breakpoint sits. `None` means the
    /// breakpoint covers the system + tools block only.
    pub after_message: Option<usize>,
    pub hash: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DivergenceKind {
    SystemChanged,
    ToolsChanged,
    HistoryRewritten,
}

#[derive(Debug, Clone, Serialize)]
pub struct Divergence {
    pub kind: DivergenceKind,
    pub detail: String,
    /// True when the rewrite is an intentional prefix-anchored compaction.
    pub expected: bool,
}

/// The result of planning one request: what will be reused, what must be
/// prefilled, and whether the prefix is stable.
#[derive(Debug, Clone, Serialize)]
pub struct RequestPlan {
    /// Hash of the full wire prompt.
    pub prompt_hash: String,
    pub segments: Vec<SegmentInfo>,
    /// Number of leading messages that are byte-identical to the last request.
    pub stable_prefix_messages: usize,
    /// Theoretical reuse: all tokens up to the stable prefix boundary.
    pub reuse_tokens: usize,
    /// Reuse actually achievable given the cache granularity.
    pub reusable_tokens: usize,
    /// Tokens the server must prefill this turn.
    pub prefill_tokens: usize,
    pub estimated_prompt_tokens: usize,
    /// True when no unexpected prefix rewrite occurred.
    pub stable: bool,
    pub divergence: Option<Divergence>,
    pub cache_granularity: usize,
    /// True on the first request of a session (nothing to reuse).
    pub cache_cold: bool,
    pub breakpoints: Vec<Breakpoint>,
}

impl RequestPlan {
    /// Fraction of the prompt served from cache (0.0 when cold).
    pub fn cache_hit_ratio(&self) -> f64 {
        if self.estimated_prompt_tokens == 0 {
            return 0.0;
        }
        self.reusable_tokens as f64 / self.estimated_prompt_tokens as f64
    }
}

/// Persisted snapshot of the last committed request, used to compute reuse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanSnapshot {
    pub system_hash: String,
    pub tools_hash: String,
    pub message_hashes: Vec<String>,
    pub message_tokens: Vec<usize>,
    pub prompt_hash: String,
    pub prompt_tokens: usize,
}

/// Plans requests against a frozen system prompt + tool schema set.
#[derive(Debug, Clone)]
pub struct CacheEngine {
    system: FrozenSegment,
    tools: FrozenSegment,
    last: Option<PlanSnapshot>,
    granularity: usize,
    pub estimator: TokenEstimator,
}

impl CacheEngine {
    /// Create an engine with frozen segments.
    ///
    /// `system` and `tools` are serialized canonically; their bytes are what
    /// the provider will send, so hashing them here is authoritative.
    pub fn new(system: &str, tools: &[Value], granularity: usize) -> Self {
        let tools_bytes = canonical_bytes(&Value::Array(tools.to_vec()));
        let estimator = TokenEstimator::default();
        let system_segment = FrozenSegment {
            hash: hash_str(system),
            bytes: system.len(),
            tokens: estimator.tokens(system),
        };
        let tools_segment = FrozenSegment {
            hash: hash_bytes(&tools_bytes),
            bytes: tools_bytes.len(),
            tokens: estimator.estimate_tools(tools),
        };
        Self {
            system: system_segment,
            tools: tools_segment,
            last: None,
            granularity: granularity.max(1),
            estimator,
        }
    }

    /// Restore an engine with a persisted snapshot (warm resume).
    pub fn from_snapshot(
        system: &str,
        tools: &[Value],
        granularity: usize,
        snapshot: Option<PlanSnapshot>,
    ) -> Self {
        let mut engine = Self::new(system, tools, granularity);
        engine.last = snapshot;
        engine
    }

    pub fn system_hash(&self) -> &str {
        &self.system.hash
    }

    pub fn tools_hash(&self) -> &str {
        &self.tools.hash
    }

    pub fn snapshot(&self) -> Option<&PlanSnapshot> {
        self.last.as_ref()
    }

    pub fn set_snapshot(&mut self, snapshot: Option<PlanSnapshot>) {
        self.last = snapshot;
    }

    pub fn granularity(&self) -> usize {
        self.granularity
    }

    /// Plan a request over the current transcript. **Pure**: does not mutate the
    /// committed snapshot, so calling it twice cannot inflate reuse.
    ///
    /// `compaction` is the record of a compaction performed *before* this
    /// request; it marks the resulting history rewrite as intentional.
    pub fn plan(&self, messages: &[Message], compaction: Option<&CompactionRecord>) -> RequestPlan {
        let message_hashes: Vec<String> = messages.iter().map(|m| m.wire_hash()).collect();
        let message_tokens: Vec<usize> = messages
            .iter()
            .map(|m| self.estimator.estimate_message(m))
            .collect();

        let mut prompt_hash = hash_pair(&self.system.hash, &self.tools.hash);
        for hash in &message_hashes {
            prompt_hash = hash_pair(&prompt_hash, hash);
        }

        let total_message_tokens: usize = message_tokens.iter().sum();
        let estimated_prompt_tokens = self.system.tokens + self.tools.tokens + total_message_tokens;

        let cache_cold = self.last.is_none();
        let (stable_prefix_messages, divergence) = match &self.last {
            None => (0usize, None),
            Some(last) => {
                let lcp = longest_common_prefix(&message_hashes, &last.message_hashes);
                let mut divergence = None;
                if self.system.hash != last.system_hash {
                    divergence = Some(Divergence {
                        kind: DivergenceKind::SystemChanged,
                        detail: format!(
                            "system prompt bytes changed ({} -> {}); the entire prompt must be re-prefilled",
                            short(&last.system_hash),
                            short(&self.system.hash)
                        ),
                        expected: false,
                    });
                } else if self.tools.hash != last.tools_hash {
                    divergence = Some(Divergence {
                        kind: DivergenceKind::ToolsChanged,
                        detail: format!(
                            "tool schema block changed ({} -> {}); the entire prompt must be re-prefilled",
                            short(&last.tools_hash),
                            short(&self.tools.hash)
                        ),
                        expected: false,
                    });
                } else if lcp < last.message_hashes.len() {
                    let expected = compaction
                        .map(|record| lcp >= record.head_len)
                        .unwrap_or(false);
                    divergence = Some(Divergence {
                        kind: DivergenceKind::HistoryRewritten,
                        detail: format!(
                            "history diverged at message {} ({} messages were sent last turn)",
                            lcp,
                            last.message_hashes.len()
                        ),
                        expected,
                    });
                }
                (lcp, divergence)
            }
        };

        // A cold cache has nothing to reuse, even though the system/tools block
        // would be a stable prefix once the server has seen it.
        let reuse_tokens = if cache_cold {
            0
        } else {
            self.system.tokens
                + self.tools.tokens
                + message_tokens[..stable_prefix_messages]
                    .iter()
                    .sum::<usize>()
        };
        let reusable_tokens = floor_to_granularity(reuse_tokens, self.granularity);
        let prefill_tokens = estimated_prompt_tokens.saturating_sub(reusable_tokens);

        let stable = divergence.as_ref().map(|d| d.expected).unwrap_or(true);

        let mut segments = Vec::with_capacity(2 + messages.len());
        segments.push(SegmentInfo {
            name: "system".to_string(),
            hash: self.system.hash.clone(),
            bytes: self.system.bytes,
            tokens: self.system.tokens,
            cached: !cache_cold,
        });
        segments.push(SegmentInfo {
            name: "tools".to_string(),
            hash: self.tools.hash.clone(),
            bytes: self.tools.bytes,
            tokens: self.tools.tokens,
            cached: !cache_cold,
        });
        for (index, message) in messages.iter().enumerate() {
            segments.push(SegmentInfo {
                name: format!("message[{index}]:{}", message.role.as_str()),
                hash: message_hashes[index].clone(),
                bytes: message.content.len(),
                tokens: message_tokens[index],
                cached: index < stable_prefix_messages,
            });
        }

        let mut breakpoints = Vec::with_capacity(3);
        breakpoints.push(Breakpoint {
            kind: BreakpointKind::Tools,
            after_message: None,
            hash: hash_pair(&self.system.hash, &self.tools.hash),
        });
        if stable_prefix_messages > 0 {
            breakpoints.push(Breakpoint {
                kind: BreakpointKind::StablePrefix,
                after_message: Some(stable_prefix_messages - 1),
                hash: hash_pair(
                    &hash_pair(&self.system.hash, &self.tools.hash),
                    &message_hashes[stable_prefix_messages - 1],
                ),
            });
        }
        if !messages.is_empty() {
            breakpoints.push(Breakpoint {
                kind: BreakpointKind::TranscriptEnd,
                after_message: Some(messages.len() - 1),
                hash: prompt_hash.clone(),
            });
        }

        RequestPlan {
            prompt_hash,
            segments,
            stable_prefix_messages,
            reuse_tokens,
            reusable_tokens,
            prefill_tokens,
            estimated_prompt_tokens,
            stable,
            divergence,
            cache_granularity: self.granularity,
            cache_cold,
            breakpoints,
        }
    }

    /// Record what was actually sent. Call after a request completes (or is
    /// dispatched), never before.
    pub fn commit(&mut self, messages: &[Message]) {
        let message_hashes: Vec<String> = messages.iter().map(|m| m.wire_hash()).collect();
        let message_tokens: Vec<usize> = messages
            .iter()
            .map(|m| self.estimator.estimate_message(m))
            .collect();
        let mut prompt_hash = hash_pair(&self.system.hash, &self.tools.hash);
        for hash in &message_hashes {
            prompt_hash = hash_pair(&prompt_hash, hash);
        }
        let prompt_tokens =
            self.system.tokens + self.tools.tokens + message_tokens.iter().sum::<usize>();
        self.last = Some(PlanSnapshot {
            system_hash: self.system.hash.clone(),
            tools_hash: self.tools.hash.clone(),
            message_hashes,
            message_tokens,
            prompt_hash,
            prompt_tokens,
        });
    }

    /// Feed real usage back into the estimator so compaction thresholds track
    /// the provider's tokenizer.
    pub fn observe_usage(&mut self, estimated_prompt_tokens: usize, actual_prompt_tokens: usize) {
        self.estimator
            .observe(estimated_prompt_tokens, actual_prompt_tokens);
    }
}

fn floor_to_granularity(tokens: usize, granularity: usize) -> usize {
    if granularity <= 1 {
        return tokens;
    }
    (tokens / granularity) * granularity
}

fn longest_common_prefix(left: &[String], right: &[String]) -> usize {
    let mut index = 0;
    while index < left.len() && index < right.len() && left[index] == right[index] {
        index += 1;
    }
    index
}

fn short(hash: &str) -> &str {
    &hash[..hash.len().min(12)]
}

/// Convenience constructor for the DeepSeek cache granularity.
pub fn deepseek_engine(system: &str, tools: &[Value]) -> CacheEngine {
    CacheEngine::new(system, tools, DEEPSEEK_CACHE_GRANULARITY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn engine() -> CacheEngine {
        CacheEngine::new("system prompt", &[json!({"name": "bash"})], 64)
    }

    #[test]
    fn first_turn_is_cold() {
        let mut engine = engine();
        let messages = vec![Message::user("hello")];
        let plan = engine.plan(&messages, None);
        assert!(plan.cache_cold);
        assert_eq!(plan.stable_prefix_messages, 0);
        assert_eq!(plan.reusable_tokens, 0);
        assert_eq!(plan.prefill_tokens, plan.estimated_prompt_tokens);
        assert!(plan.stable);
        engine.commit(&messages);
    }

    #[test]
    fn append_only_turn_reuses_prefix() {
        // Granularity 1 so small test prompts show real reuse.
        let mut engine = CacheEngine::new("system prompt", &[json!({"name": "bash"})], 1);
        let mut messages = vec![Message::user("hello")];
        let first = engine.plan(&messages, None);
        engine.commit(&messages);

        messages.push(Message::assistant("hi"));
        let second = engine.plan(&messages, None);

        assert_eq!(second.stable_prefix_messages, 1);
        assert!(second.stable);
        assert!(second.reusable_tokens > 0);
        assert_eq!(second.reuse_tokens, first.estimated_prompt_tokens);
        assert!(second.prefill_tokens < second.estimated_prompt_tokens);
        assert!(second.cache_hit_ratio() > 0.0);
    }

    #[test]
    fn planning_twice_does_not_inflate_reuse() {
        let engine = engine();
        let messages = vec![Message::user("hello")];
        let first = engine.plan(&messages, None);
        let second = engine.plan(&messages, None);
        assert_eq!(first.reusable_tokens, second.reusable_tokens);
        assert_eq!(second.reusable_tokens, 0);
    }

    #[test]
    fn rewriting_history_is_detected() {
        let mut engine = engine();
        let messages = vec![Message::user("hello"), Message::assistant("hi")];
        engine.commit(&messages);

        let rewritten = vec![Message::user("hello"), Message::assistant("different")];
        let plan = engine.plan(&rewritten, None);

        assert_eq!(plan.stable_prefix_messages, 1);
        assert!(!plan.stable);
        let divergence = plan.divergence.expect("divergence");
        assert_eq!(divergence.kind, DivergenceKind::HistoryRewritten);
        assert!(!divergence.expected);
    }

    #[test]
    fn compaction_divergence_is_expected() {
        let mut engine = engine();
        let messages = vec![
            Message::user("task"),
            Message::assistant("step 1"),
            Message::assistant("step 2"),
        ];
        engine.commit(&messages);

        let compacted = vec![Message::user("task"), Message::user("summary")];
        let record = CompactionRecord {
            head_len: 1,
            tail_start: 3,
            dropped: 2,
            summary_tokens: 10,
            before_tokens: 100,
            after_tokens: 40,
            at_turn: 2,
        };
        let plan = engine.plan(&compacted, Some(&record));
        assert_eq!(plan.stable_prefix_messages, 1);
        assert!(plan.stable);
        assert!(plan.divergence.expect("divergence").expected);
    }

    #[test]
    fn reuse_is_floored_to_granularity() {
        let mut engine = CacheEngine::new("s", &[], 64);
        let messages = vec![Message::user("x")];
        engine.commit(&messages);
        let plan = engine.plan(&messages, None);
        assert_eq!(plan.reusable_tokens % 64, 0);
    }

    #[test]
    fn breakpoints_cover_tools_stable_prefix_and_end() {
        let mut engine = engine();
        let messages = vec![Message::user("hello")];
        engine.commit(&messages);
        let plan = engine.plan(&messages, None);
        let kinds: Vec<_> = plan.breakpoints.iter().map(|b| b.kind).collect();
        assert!(kinds.contains(&BreakpointKind::Tools));
        assert!(kinds.contains(&BreakpointKind::StablePrefix));
        assert!(kinds.contains(&BreakpointKind::TranscriptEnd));
    }

    #[test]
    fn snapshot_survives_clone() {
        let mut engine = engine();
        engine.commit(&[Message::user("hi")]);
        let restored = CacheEngine::from_snapshot(
            "system prompt",
            &[json!({"name": "bash"})],
            64,
            engine.snapshot().cloned(),
        );
        let plan = restored.plan(&[Message::user("hi")], None);
        assert_eq!(plan.stable_prefix_messages, 1);
    }
}

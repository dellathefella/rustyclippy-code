//! Token estimation tuned for the DeepSeek tokenizer.
//!
//! We cannot ship the exact tokenizer without a heavy dependency, so the
//! estimator uses a calibrated heuristic:
//!
//! * CJK characters count as roughly one token each (DeepSeek's tokenizer is
//!   efficient for Chinese, so this is a safe upper bound).
//! * Non-CJK text counts at `chars_per_token` characters per token (default
//!   3.6, which is close for English prose and Rust/TypeScript source).
//! * Each message adds a small structural overhead for role markers.
//!
//! The estimate is then scaled by an EMA-calibrated factor derived from real
//! `usage` numbers reported by the provider, so compaction decisions track the
//! actual tokenizer instead of drifting.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::canonical::canonical_string;
use crate::message::{Message, Role};

/// DeepSeek context cache granularity, in tokens. Cache reuse is only possible
/// on whole 64-token blocks, so planning floors reuse estimates to this.
pub const DEEPSEEK_CACHE_GRANULARITY: usize = 64;

/// Conservative per-image token cost. Vision encoders bill by patch count, not
/// by base64 length; counting the encoded bytes would wildly overestimate the
/// prompt and trigger unnecessary compaction.
pub const IMAGE_TOKENS: usize = 1_024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEstimator {
    /// Characters per token for non-CJK text.
    pub chars_per_token: f64,
    /// Calibration factor applied to the raw estimate (1.0 = uncalibrated).
    pub scale: f64,
}

impl Default for TokenEstimator {
    fn default() -> Self {
        Self {
            chars_per_token: 3.6,
            scale: 1.0,
        }
    }
}

impl TokenEstimator {
    pub fn new(chars_per_token: f64) -> Self {
        Self {
            chars_per_token,
            scale: 1.0,
        }
    }

    /// Raw heuristic token count for a string.
    pub fn estimate_text(&self, text: &str) -> usize {
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
        let other_tokens = ((other as f64) / self.chars_per_token).ceil() as usize;
        cjk + other_tokens
    }

    /// Calibrated token count for a string.
    pub fn tokens(&self, text: &str) -> usize {
        scale(self.estimate_text(text), self.scale)
    }

    /// Token cost of one message, including structural overhead.
    pub fn estimate_message(&self, message: &Message) -> usize {
        let mut total = self.estimate_text(&message.content);
        total += message.images.len() * IMAGE_TOKENS;
        for call in &message.tool_calls {
            total += self.estimate_text(&call.name);
            total += self.estimate_text(&canonical_string(&call.arguments));
            total += 8; // id + wrapper overhead
        }
        if message.tool_call_id.is_some() {
            total += 4;
        }
        // Role marker / separators.
        total += 4;
        scale(total, self.scale)
    }

    /// Token cost of a tool schema list (serialized JSON).
    pub fn estimate_tools(&self, tools: &[Value]) -> usize {
        let bytes = canonical_string(&Value::Array(tools.to_vec()));
        scale(self.estimate_text(&bytes), self.scale)
    }

    /// Token cost of a raw JSON value.
    pub fn estimate_value(&self, value: &Value) -> usize {
        scale(self.estimate_text(&canonical_string(value)), self.scale)
    }

    /// Fold an observed `(estimated, actual)` pair into the calibration factor.
    ///
    /// Uses exponential smoothing so a single noisy measurement cannot swing
    /// compaction decisions. The factor is clamped to a sane range.
    pub fn observe(&mut self, estimated: usize, actual: usize) {
        if estimated == 0 || actual == 0 {
            return;
        }
        let observed = actual as f64 / estimated as f64;
        let observed = observed.clamp(0.5, 2.5);
        // 30% weight on the new observation.
        self.scale = (self.scale * 0.7 + observed * 0.3).clamp(0.5, 2.5);
    }
}

fn scale(raw: usize, factor: f64) -> usize {
    ((raw as f64) * factor).ceil() as usize
}

/// CJK ranges plus full-width punctuation.
fn is_cjk(ch: char) -> bool {
    matches!(
        ch as u32,
        0x1100..=0x11FF   // Hangul Jamo
        | 0x2E80..=0x2EFF // CJK Radicals
        | 0x3000..=0x303F // CJK Symbols and Punctuation
        | 0x3040..=0x30FF // Hiragana / Katakana
        | 0x3400..=0x4DBF // CJK Unified Ideographs Extension A
        | 0x4E00..=0x9FFF // CJK Unified Ideographs
        | 0xF900..=0xFAFF // CJK Compatibility Ideographs
        | 0xFF00..=0xFFEF // Halfwidth and Fullwidth Forms
        | 0xAC00..=0xD7AF // Hangul Syllables
    )
}

/// Rough role overhead used when estimating a whole transcript.
pub fn role_overhead(role: Role) -> usize {
    match role {
        Role::System => 6,
        Role::User => 4,
        Role::Assistant => 4,
        Role::Tool => 6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_and_code_estimates_are_sane() {
        let estimator = TokenEstimator::default();
        let text = "fn main() { println!(\"hello world\"); }";
        let tokens = estimator.estimate_text(text);
        assert!((8..=20).contains(&tokens), "got {tokens}");
    }

    #[test]
    fn cjk_counts_one_per_char() {
        let estimator = TokenEstimator::default();
        assert_eq!(estimator.estimate_text("你好世界"), 4);
    }

    #[test]
    fn calibration_moves_toward_observation() {
        let mut estimator = TokenEstimator::default();
        estimator.observe(100, 150);
        assert!(estimator.scale > 1.0);
        assert!(estimator.scale < 1.5);
    }

    #[test]
    fn calibration_is_clamped() {
        let mut estimator = TokenEstimator::default();
        for _ in 0..50 {
            estimator.observe(100, 10_000);
        }
        assert!(estimator.scale <= 2.5);
    }

    #[test]
    fn images_count_a_fixed_cost_not_base64_length() {
        let estimator = TokenEstimator::default();
        let plain = Message::user("describe");
        let with_image = Message::user_with_images(
            "describe",
            vec![crate::image::ImageAttachment::from_bytes(
                "image/png",
                &[0u8; 4096],
            )],
        );
        let delta = estimator.estimate_message(&with_image) - estimator.estimate_message(&plain);
        assert_eq!(delta, IMAGE_TOKENS);
    }
}

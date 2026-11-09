//! Usage parsing across provider dialects.
//!
//! DeepSeek reports cache hits explicitly:
//!   `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`
//! OpenAI reports `prompt_tokens_details.cached_tokens`.
//! llama.cpp reports `timings.cache_n`.

use serde_json::Value;

use crate::provider::Usage;

pub fn parse_usage(value: &Value) -> Usage {
    let prompt_tokens = number(value.get("prompt_tokens"));
    let completion_tokens = number(value.get("completion_tokens"));

    let mut cached_tokens = number(value.get("prompt_cache_hit_tokens"));

    if let Some(details) = value.get("prompt_tokens_details") {
        cached_tokens = cached_tokens.max(number(details.get("cached_tokens")));
    }
    if let Some(timings) = value.get("timings") {
        cached_tokens = cached_tokens.max(number(timings.get("cache_n")));
        cached_tokens = cached_tokens.max(number(timings.get("prompt_cache_hit_tokens")));
    }

    let reasoning_tokens = value
        .get("completion_tokens_details")
        .map(|details| number(details.get("reasoning_tokens")))
        .unwrap_or(0);

    Usage {
        prompt_tokens,
        completion_tokens,
        cached_tokens: cached_tokens.min(prompt_tokens),
        reasoning_tokens,
    }
}

fn number(value: Option<&Value>) -> usize {
    value
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deepseek_usage() {
        let value = json!({
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_cache_hit_tokens": 960,
            "prompt_cache_miss_tokens": 40
        });
        let usage = parse_usage(&value);
        assert_eq!(usage.prompt_tokens, 1000);
        assert_eq!(usage.cached_tokens, 960);
        assert_eq!(usage.prefill_tokens(), 40);
    }

    #[test]
    fn openai_usage() {
        let value = json!({
            "prompt_tokens": 500,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 448}
        });
        let usage = parse_usage(&value);
        assert_eq!(usage.cached_tokens, 448);
    }

    #[test]
    fn llama_cpp_timings() {
        let value = json!({
            "prompt_tokens": 800,
            "completion_tokens": 30,
            "timings": {"cache_n": 768}
        });
        let usage = parse_usage(&value);
        assert_eq!(usage.cached_tokens, 768);
    }

    #[test]
    fn cached_never_exceeds_prompt() {
        let value = json!({"prompt_tokens": 10, "prompt_cache_hit_tokens": 99});
        assert_eq!(parse_usage(&value).cached_tokens, 10);
    }
}

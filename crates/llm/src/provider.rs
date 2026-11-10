//! Provider abstraction and request/stream types.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use futures_util::stream::BoxStream;
use rustyclippy_core::{Breakpoint, Message, ProviderConfig, ProviderKind};
use serde_json::Value;

use crate::openai::OpenAiProvider;

/// A fully materialized request. `system` and `tools` are the frozen segments
/// produced by the cache engine, so their bytes are identical every turn.
#[derive(Debug, Clone)]
pub struct LlmRequest {
    pub model: String,
    pub system: String,
    pub tools: Vec<Value>,
    pub messages: Vec<Message>,
    pub max_tokens: usize,
    pub temperature: f64,
    /// Ask the server to reuse its prompt cache (llama.cpp extension).
    pub cache_prompt: bool,
    /// Cache breakpoints computed by the engine (diagnostics; used by providers
    /// that support explicit cache markers).
    pub breakpoints: Vec<Breakpoint>,
    /// Disable tool calls for this request (used for compaction summaries).
    pub tool_choice_none: bool,
    /// Reasoning effort (`low`/`medium`/`high`/`max`). `None` omits the field.
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// Tokens served from the provider's prefix cache.
    pub cached_tokens: usize,
    pub reasoning_tokens: usize,
    /// Decode speed reported by the server (llama.cpp `timings`), in tokens/s.
    /// `None` when the provider does not report it.
    pub decode_tokens_per_second: Option<f64>,
}

impl Usage {
    /// Tokens the server actually had to prefill.
    pub fn prefill_tokens(&self) -> usize {
        self.prompt_tokens.saturating_sub(self.cached_tokens)
    }

    pub fn cache_hit_ratio(&self) -> f64 {
        if self.prompt_tokens == 0 {
            return 0.0;
        }
        self.cached_tokens as f64 / self.prompt_tokens as f64
    }
}

/// Incremental stream event.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Text(String),
    Reasoning(String),
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        args_fragment: Option<String>,
    },
    Usage(Usage),
    Done,
}

#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    async fn stream(&self, request: LlmRequest) -> Result<BoxStream<'static, Result<StreamEvent>>>;
}

/// Build a provider from configuration.
pub fn build_provider(config: &ProviderConfig) -> Result<Arc<dyn Provider>> {
    match config.kind {
        ProviderKind::Deepseek | ProviderKind::OpenAiCompat => {
            Ok(Arc::new(OpenAiProvider::new(config)?))
        }
    }
}

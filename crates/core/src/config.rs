//! Configuration.
//!
//! Defaults are tuned for **DeepSeek V4.1 Flash served locally** (MLX /
//! llama.cpp OpenAI-compatible endpoint). The DeepSeek hosted API is available
//! as a preset.
//!
//! Config is loaded from, in order: an explicit `--config` path, `./rustyclippy-code.json`,
//! `$RUSTYCLIPPY_HOME/config.json`, then built-in defaults. Environment variables
//! override individual fields.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::session::default_home;
use crate::token::DEEPSEEK_CACHE_GRANULARITY;

/// DeepSeek V4.1 Flash context window (1M tokens).
pub const DEEPSEEK_V41_CONTEXT_TOKENS: usize = 1_048_576;

/// DeepSeek V4.1 Flash output ceiling (384K tokens).
///
/// The architecture allows 384K, but a local server budgets far less: MLX
/// reports `agent_budget.output = 65536` for this model. Requesting more than
/// the server allows is rejected or silently clamped, so the default tracks the
/// server's budget rather than the architectural maximum.
pub const DEEPSEEK_V41_OUTPUT_TOKENS: usize = 65_536;

/// Autocompaction threshold: the context window minus the output budget, so a
/// full-length answer still fits after the prompt.
pub const DEEPSEEK_V41_AUTOCOMPACT_TOKENS: usize =
    DEEPSEEK_V41_CONTEXT_TOKENS - DEEPSEEK_V41_OUTPUT_TOKENS;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    /// DeepSeek hosted API (OpenAI-compatible, automatic prefix caching).
    Deepseek,
    /// Any OpenAI-compatible endpoint: MLX server, llama.cpp, vLLM, LM Studio, Ollama.
    #[serde(rename = "openai_compat")]
    OpenAiCompat,
}

impl ProviderKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ProviderKind::Deepseek => "deepseek",
            ProviderKind::OpenAiCompat => "openai_compat",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    pub base_url: String,
    /// Environment variable holding the API key (empty for local servers).
    pub api_key_env: String,
    pub model: String,
    pub max_context_tokens: usize,
    pub max_output_tokens: usize,
    pub temperature: f64,
    /// llama.cpp extension: reuse the server's prompt cache.
    pub cache_prompt: bool,
    /// Replay `reasoning_content` into later requests. Off by default: it bloats
    /// the prefix and DeepSeek reasoner models reject it.
    pub include_reasoning_in_history: bool,
    /// Reasoning effort: `off`, `low`, `high`, or `max`. `None`/`off` omits the
    /// parameter. Matches the DSH harness's `reasoningEffort` knob.
    pub reasoning_effort: Option<String>,
    pub timeout_secs: u64,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            kind: ProviderKind::OpenAiCompat,
            base_url: "http://127.0.0.1:8080/v1".to_string(),
            api_key_env: "DEEPSEEK_API_KEY".to_string(),
            model: "DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-3.0bpw".to_string(),
            // DeepSeek V4.1 Flash: 1M context, 64k output budget on a local
            // server. Autocompaction fires at context minus output.
            max_context_tokens: DEEPSEEK_V41_CONTEXT_TOKENS,
            max_output_tokens: DEEPSEEK_V41_OUTPUT_TOKENS,
            temperature: 0.2,
            cache_prompt: true,
            include_reasoning_in_history: false,
            reasoning_effort: Some("high".to_string()),
            timeout_secs: 600,
        }
    }
}

impl ProviderConfig {
    /// DeepSeek hosted API preset.
    pub fn deepseek_api(model: Option<String>) -> Self {
        Self {
            kind: ProviderKind::Deepseek,
            base_url: "https://api.deepseek.com/v1".to_string(),
            api_key_env: "DEEPSEEK_API_KEY".to_string(),
            model: model.unwrap_or_else(|| "deepseek-v4.1-flash".to_string()),
            max_context_tokens: DEEPSEEK_V41_CONTEXT_TOKENS,
            max_output_tokens: DEEPSEEK_V41_OUTPUT_TOKENS,
            temperature: 0.2,
            cache_prompt: false, // hosted API caches automatically
            include_reasoning_in_history: false,
            reasoning_effort: None,
            timeout_secs: 300,
        }
    }

    /// Local MLX preset.
    pub fn local_mlx(base_url: Option<String>, model: Option<String>) -> Self {
        Self {
            kind: ProviderKind::OpenAiCompat,
            base_url: base_url.unwrap_or_else(|| "http://127.0.0.1:8080/v1".to_string()),
            model: model
                .unwrap_or_else(|| "DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-3.0bpw".to_string()),
            cache_prompt: true,
            ..Self::default()
        }
    }

    /// Normalized reasoning effort, or `None` when reasoning is off.
    pub fn effective_reasoning_effort(&self) -> Option<&str> {
        match self.reasoning_effort.as_deref() {
            None => None,
            Some(value) if value.trim().is_empty() || value.eq_ignore_ascii_case("off") => None,
            Some(value) => Some(value.trim()),
        }
    }

    /// Resolve the API key from the configured environment variable.
    pub fn api_key(&self) -> Option<String> {
        if self.api_key_env.trim().is_empty() {
            return None;
        }
        std::env::var(&self.api_key_env)
            .ok()
            .filter(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentMode {
    /// Can edit files and run commands.
    Build,
    /// Read-only analysis; produces a plan.
    Plan,
}

impl AgentMode {
    pub fn as_str(self) -> &'static str {
        match self {
            AgentMode::Build => "build",
            AgentMode::Plan => "plan",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_lowercase().as_str() {
            "build" => Ok(AgentMode::Build),
            "plan" => Ok(AgentMode::Plan),
            other => bail!("unknown mode `{other}` (expected build or plan)"),
        }
    }

    /// The other mode, for cycling with Shift+Tab.
    pub fn toggle(self) -> Self {
        match self {
            AgentMode::Build => AgentMode::Plan,
            AgentMode::Plan => AgentMode::Build,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentConfig {
    pub mode: AgentMode,
    pub max_steps: usize,
    /// Maximum subagent recursion depth for the `task` tool.
    pub max_depth: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            mode: AgentMode::Build,
            max_steps: 60,
            max_depth: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CacheConfig {
    /// Compact when the projected prompt crosses this absolute token count.
    /// DeepSeek V4.1 Flash defaults to 250k, leaving headroom for output and
    /// avoiding a hard-context failure.
    pub autocompact_at_tokens: usize,
    /// Fallback fraction used when `autocompact_at_tokens` is 0.
    pub compaction_threshold: f64,
    /// Messages pinned at the head (environment + original task).
    pub head_messages: usize,
    /// Recent messages kept verbatim after compaction.
    pub tail_messages: usize,
    /// Skip compaction when it would free fewer tokens than this.
    pub min_compaction_gain_tokens: usize,
    /// Tool output is truncated at append time, never rewritten later.
    pub tool_output_max_chars: usize,
    /// Cache block granularity (DeepSeek uses 64).
    pub granularity: usize,
    /// Assume the server still holds the KV cache when resuming a session.
    pub assume_warm_resume: bool,
    /// Max tokens for the compaction summary.
    pub summary_max_tokens: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            autocompact_at_tokens: DEEPSEEK_V41_AUTOCOMPACT_TOKENS,
            compaction_threshold: 0.85,
            head_messages: 2,
            tail_messages: 8,
            min_compaction_gain_tokens: 2_048,
            tool_output_max_chars: 12_000,
            granularity: DEEPSEEK_CACHE_GRANULARITY,
            assume_warm_resume: false,
            summary_max_tokens: 1_024,
        }
    }
}

impl ProviderConfig {
    /// Clamp the output budget to what the server actually allows.
    ///
    /// A local server reports its own ceiling (MLX exposes `agent_budget.output`).
    /// Requesting more than that is rejected or silently truncated, so the
    /// configured value is lowered to fit. Returns the value that was applied.
    pub fn clamp_output_to(&mut self, server_output: usize) -> usize {
        if server_output > 0 && self.max_output_tokens > server_output {
            self.max_output_tokens = server_output;
        }
        self.max_output_tokens
    }

    /// Clamp the context window to what the server reports.
    pub fn clamp_context_to(&mut self, server_context: usize) -> usize {
        if server_context > 0 && self.max_context_tokens > server_context {
            self.max_context_tokens = server_context;
        }
        self.max_context_tokens
    }
}

impl CacheConfig {
    /// Token budget that triggers autocompaction.
    pub fn autocompact_budget(&self, max_context_tokens: usize) -> usize {
        if self.autocompact_at_tokens > 0 {
            self.autocompact_at_tokens.min(max_context_tokens)
        } else {
            (max_context_tokens as f64 * self.compaction_threshold) as usize
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionMode {
    Ask,
    Auto,
    Readonly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PermissionConfig {
    pub bash: PermissionMode,
    pub edit: PermissionMode,
    pub write: PermissionMode,
    pub task: PermissionMode,
}

impl Default for PermissionConfig {
    fn default() -> Self {
        Self {
            bash: PermissionMode::Ask,
            edit: PermissionMode::Ask,
            write: PermissionMode::Ask,
            task: PermissionMode::Ask,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WebProvider {
    /// DuckDuckGo HTML endpoint; no API key required.
    #[serde(rename = "duckduckgo")]
    DuckDuckGo,
    Brave,
    Tavily,
    Searxng,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WebConfig {
    pub provider: WebProvider,
    /// Environment variable holding the search API key (Brave/Tavily).
    pub api_key_env: String,
    /// SearXNG base URL when `provider = searxng`.
    pub searxng_url: Option<String>,
    pub max_results: usize,
    pub fetch_max_chars: usize,
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            provider: WebProvider::DuckDuckGo,
            api_key_env: "BRAVE_API_KEY".to_string(),
            searxng_url: None,
            max_results: 8,
            fetch_max_chars: 20_000,
        }
    }
}

impl WebConfig {
    pub fn api_key(&self) -> Option<String> {
        if self.api_key_env.trim().is_empty() {
            return None;
        }
        std::env::var(&self.api_key_env)
            .ok()
            .filter(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    pub provider: ProviderConfig,
    pub agent: AgentConfig,
    pub cache: CacheConfig,
    pub permissions: PermissionConfig,
    pub web: WebConfig,
}

fn env_value(name: &str, legacy: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .or_else(|| std::env::var(legacy).ok())
        .filter(|value| !value.trim().is_empty())
}

impl Config {
    /// Load config from the first available source.
    pub fn load(explicit: Option<&Path>) -> Result<Self> {
        let mut config = if let Some(path) = explicit {
            Self::from_file(path)?
        } else if Path::new("rustyclippy-code.json").exists() {
            Self::from_file(Path::new("rustyclippy-code.json"))?
        } else if Path::new("deeprust.json").exists() {
            // Legacy project config from before the rename.
            Self::from_file(Path::new("deeprust.json"))?
        } else {
            let home_config = default_home().join("config.json");
            if home_config.exists() {
                Self::from_file(&home_config)?
            } else {
                Self::default()
            }
        };
        config.apply_env();
        config.validate()?;
        Ok(config)
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing config {}", path.display()))
    }

    pub fn apply_env(&mut self) {
        if let Some(value) = env_value("RUSTYCLIPPY_BASE_URL", "DEEPRUST_BASE_URL") {
            if !value.trim().is_empty() {
                self.provider.base_url = value;
            }
        }
        if let Some(value) = env_value("RUSTYCLIPPY_MODEL", "DEEPRUST_MODEL") {
            if !value.trim().is_empty() {
                self.provider.model = value;
            }
        }
        if let Some(value) = env_value("RUSTYCLIPPY_PROVIDER", "DEEPRUST_PROVIDER") {
            match value.trim().to_lowercase().as_str() {
                "deepseek" => self.provider.kind = ProviderKind::Deepseek,
                "openai_compat" | "openai" | "local" => {
                    self.provider.kind = ProviderKind::OpenAiCompat
                }
                _ => {}
            }
        }
        if let Some(value) = env_value("RUSTYCLIPPY_MAX_CONTEXT", "DEEPRUST_MAX_CONTEXT") {
            if let Ok(tokens) = value.parse::<usize>() {
                self.provider.max_context_tokens = tokens;
            }
        }
        if let Some(value) = env_value("RUSTYCLIPPY_AUTOCOMPACT", "DEEPRUST_AUTOCOMPACT") {
            if let Ok(tokens) = value.parse::<usize>() {
                self.cache.autocompact_at_tokens = tokens;
            }
        }
        if let Some(value) = env_value("RUSTYCLIPPY_REASONING_EFFORT", "DEEPRUST_REASONING_EFFORT")
        {
            self.provider.reasoning_effort = Some(value);
        }
        if let Some(value) = env_value("RUSTYCLIPPY_MODE", "DEEPRUST_MODE") {
            if let Ok(mode) = AgentMode::parse(&value) {
                self.agent.mode = mode;
            }
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.provider.base_url.trim().is_empty() {
            bail!("provider.base_url is empty");
        }
        if self.provider.model.trim().is_empty() {
            bail!("provider.model is empty");
        }
        if self.provider.max_context_tokens == 0 {
            bail!("provider.max_context_tokens must be > 0");
        }
        if self.provider.max_output_tokens >= self.provider.max_context_tokens {
            bail!(
                "provider.max_output_tokens ({}) must be smaller than max_context_tokens ({})",
                self.provider.max_output_tokens,
                self.provider.max_context_tokens
            );
        }
        if self.cache.autocompact_at_tokens > self.provider.max_context_tokens {
            bail!(
                "cache.autocompact_at_tokens ({}) must not exceed max_context_tokens ({})",
                self.cache.autocompact_at_tokens,
                self.provider.max_context_tokens
            );
        }
        if !(0.0..=1.0).contains(&self.cache.compaction_threshold) {
            bail!("cache.compaction_threshold must be within 0.0..=1.0");
        }
        if self.cache.granularity == 0 {
            bail!("cache.granularity must be > 0");
        }
        if let Some(effort) = self.provider.effective_reasoning_effort() {
            let allowed = ["low", "medium", "high", "max"];
            if !allowed.contains(&effort) {
                bail!(
                    "provider.reasoning_effort must be one of off/low/medium/high/max, got `{effort}`"
                );
            }
        }
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(path, text).with_context(|| format!("writing config {}", path.display()))?;
        Ok(())
    }

    /// Write a default config, refusing to overwrite unless `force`.
    pub fn init(path: &Path, force: bool) -> Result<()> {
        if path.exists() && !force {
            bail!(
                "{} already exists (use --force to overwrite)",
                path.display()
            );
        }
        Self::default().save(path)
    }

    pub fn config_path(&self) -> PathBuf {
        default_home().join("config.json")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        Config::default().validate().unwrap();
    }

    #[test]
    fn output_must_be_smaller_than_context() {
        let mut config = Config::default();
        config.provider.max_output_tokens = config.provider.max_context_tokens;
        assert!(config.validate().is_err());
    }

    #[test]
    fn autocompact_cannot_exceed_context() {
        let mut config = Config::default();
        config.cache.autocompact_at_tokens = config.provider.max_context_tokens + 1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn presets_are_valid() {
        let deepseek = Config {
            provider: ProviderConfig::deepseek_api(None),
            ..Config::default()
        };
        deepseek.validate().unwrap();
        let local = Config {
            provider: ProviderConfig::local_mlx(None, None),
            ..Config::default()
        };
        local.validate().unwrap();
    }

    #[test]
    fn mode_parsing() {
        assert_eq!(AgentMode::parse("BUILD").unwrap(), AgentMode::Build);
        assert_eq!(AgentMode::parse("plan").unwrap(), AgentMode::Plan);
        assert!(AgentMode::parse("nope").is_err());
    }

    #[test]
    fn mode_toggle_cycles_between_build_and_plan() {
        assert_eq!(AgentMode::Build.toggle(), AgentMode::Plan);
        assert_eq!(AgentMode::Plan.toggle(), AgentMode::Build);
        assert_eq!(AgentMode::Build.toggle().toggle(), AgentMode::Build);
    }

    #[test]
    fn reasoning_effort_normalization() {
        let mut config = Config::default();
        assert_eq!(config.provider.effective_reasoning_effort(), Some("high"));
        config.provider.reasoning_effort = Some("off".to_string());
        assert_eq!(config.provider.effective_reasoning_effort(), None);
        config.provider.reasoning_effort = Some("max".to_string());
        config.validate().unwrap();
        config.provider.reasoning_effort = Some("extreme".to_string());
        assert!(config.validate().is_err());
    }

    #[test]
    fn autocompact_budget_prefers_absolute() {
        let cache = CacheConfig::default();
        assert_eq!(
            cache.autocompact_budget(DEEPSEEK_V41_CONTEXT_TOKENS),
            DEEPSEEK_V41_AUTOCOMPACT_TOKENS
        );
        let mut fallback = cache.clone();
        fallback.autocompact_at_tokens = 0;
        assert_eq!(fallback.autocompact_budget(100_000), 85_000);
    }

    #[test]
    fn defaults_match_the_deepseek_v41_window() {
        let config = Config::default();
        assert_eq!(config.provider.max_context_tokens, 1_048_576);
        assert_eq!(config.provider.max_output_tokens, 65_536);
        // Autocompaction leaves room for a full-length answer.
        assert_eq!(
            config.cache.autocompact_at_tokens,
            config.provider.max_context_tokens - config.provider.max_output_tokens
        );
        config.validate().unwrap();
    }

    #[test]
    fn output_is_clamped_to_the_server_budget() {
        let mut provider = ProviderConfig::default();
        assert_eq!(provider.clamp_output_to(65_536), 65_536, "equal is kept");
        assert_eq!(provider.clamp_output_to(32_768), 32_768, "lowered to fit");
        assert_eq!(provider.clamp_output_to(0), 32_768, "zero means unknown");
    }

    #[test]
    fn context_is_clamped_to_the_server_window() {
        let mut provider = ProviderConfig::default();
        assert_eq!(provider.clamp_context_to(262_144), 262_144);
        assert_eq!(provider.clamp_context_to(0), 262_144, "zero means unknown");
    }
}

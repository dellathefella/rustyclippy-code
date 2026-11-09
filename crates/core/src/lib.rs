//! Core domain types for rustyclippy-code: cache-stable context engine, transcript,
//! session persistence, configuration, and token estimation.

pub mod cache;
pub mod canonical;
pub mod compaction;
pub mod config;
pub mod image;
pub mod message;
pub mod session;
pub mod system_prompt;
pub mod token;
pub mod transcript;

pub use cache::{
    Breakpoint, BreakpointKind, CacheEngine, Divergence, DivergenceKind, FrozenSegment,
    PlanSnapshot, RequestPlan, SegmentInfo,
};
pub use canonical::{
    canonical_bytes, canonical_string, canonicalize, hash_bytes, hash_str, hash_value,
};
pub use compaction::{CompactionPlan, CompactionRecord, SUMMARY_TAG};
pub use config::{
    AgentMode, CacheConfig, Config, PermissionConfig, PermissionMode, ProviderConfig, ProviderKind,
    WebConfig, WebProvider,
};
pub use image::{base64_decode, base64_encode, detect_mime, ImageAttachment};
pub use message::{Message, Role, ToolCall};
pub use session::{CacheTurnRecord, SessionEvent, SessionMeta, SessionStore, SessionSummary};
pub use token::{TokenEstimator, DEEPSEEK_CACHE_GRANULARITY};
pub use transcript::Transcript;

//! Append-only session persistence.
//!
//! Every session is a JSONL event log. Replaying it reconstructs the exact
//! transcript, which means resuming a session never silently rewrites history.
//! A separate `cache.jsonl` records per-turn cache accounting for reporting.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::compaction::CompactionRecord;
use crate::message::Message;

/// Default home directory for sessions and config.
pub fn default_home() -> PathBuf {
    if let Ok(home) = std::env::var("DEEPRUST_HOME") {
        if !home.trim().is_empty() {
            return PathBuf::from(home);
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".deeprust")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: String,
    pub cwd: String,
    pub provider: String,
    pub model: String,
    pub mode: String,
    pub system_hash: String,
    pub tools_hash: String,
    /// Frozen system prompt text. Stored so resuming replays the exact bytes.
    #[serde(default)]
    pub system_prompt: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEvent {
    Created {
        meta: SessionMeta,
    },
    Message {
        message: Message,
    },
    Compaction {
        record: CompactionRecord,
        summary: Message,
    },
    Usage {
        turn: usize,
        prompt_tokens: usize,
        cached_tokens: usize,
        prefill_tokens: usize,
        completion_tokens: usize,
        estimated_prompt_tokens: usize,
        stable: bool,
    },
    TurnEnd {
        turn: usize,
        text: String,
    },
}

/// Per-turn cache accounting, written to `cache.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheTurnRecord {
    pub turn: usize,
    pub at: DateTime<Utc>,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub prefill_tokens: usize,
    pub completion_tokens: usize,
    pub estimated_prompt_tokens: usize,
    pub stable_prefix_messages: usize,
    pub prompt_hash: String,
    pub stable: bool,
    pub cache_cold: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionSummary {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub cwd: String,
    pub model: String,
    pub mode: String,
    pub messages: usize,
    pub turns: usize,
}

#[derive(Debug, Clone)]
pub struct SessionStore {
    root: PathBuf,
}

impl SessionStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn default_store() -> Self {
        Self::new(default_home().join("sessions"))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn session_dir(&self, id: &str) -> PathBuf {
        self.root.join(id)
    }

    pub fn log_path(&self, id: &str) -> PathBuf {
        self.session_dir(id).join("session.jsonl")
    }

    pub fn cache_log_path(&self, id: &str) -> PathBuf {
        self.session_dir(id).join("cache.jsonl")
    }

    /// Create a new session id (timestamp + short random suffix).
    pub fn new_session_id() -> String {
        let now = Utc::now();
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        format!("{}-{}", now.format("%Y%m%d-%H%M%S"), &suffix[..6])
    }

    pub fn create(&self, meta: SessionMeta) -> Result<()> {
        let dir = self.session_dir(&meta.id);
        fs::create_dir_all(&dir)
            .with_context(|| format!("creating session dir {}", dir.display()))?;
        let id = meta.id.clone();
        self.append(&id, &SessionEvent::Created { meta })
    }

    pub fn append(&self, id: &str, event: &SessionEvent) -> Result<()> {
        let path = self.log_path(id);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("opening session log {}", path.display()))?;
        let line = serde_json::to_string(event)?;
        writeln!(file, "{line}")?;
        Ok(())
    }

    pub fn append_cache(&self, id: &str, record: &CacheTurnRecord) -> Result<()> {
        let path = self.cache_log_path(id);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("opening cache log {}", path.display()))?;
        let line = serde_json::to_string(record)?;
        writeln!(file, "{line}")?;
        Ok(())
    }

    pub fn load(&self, id: &str) -> Result<Vec<SessionEvent>> {
        let path = self.log_path(id);
        let file = File::open(&path)
            .with_context(|| format!("session {} not found at {}", id, path.display()))?;
        let reader = BufReader::new(file);
        let mut events = Vec::new();
        for (line_no, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let event: SessionEvent = serde_json::from_str(&line)
                .with_context(|| format!("parsing {} line {}", path.display(), line_no + 1))?;
            events.push(event);
        }
        Ok(events)
    }

    pub fn load_cache(&self, id: &str) -> Result<Vec<CacheTurnRecord>> {
        let path = self.cache_log_path(id);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            records.push(serde_json::from_str(&line)?);
        }
        Ok(records)
    }

    pub fn list(&self) -> Result<Vec<SessionSummary>> {
        if !self.root.exists() {
            return Ok(Vec::new());
        }
        let mut summaries = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let id = entry.file_name().to_string_lossy().to_string();
            let events = match self.load(&id) {
                Ok(events) => events,
                Err(_) => continue,
            };
            let meta = events.iter().find_map(|event| match event {
                SessionEvent::Created { meta } => Some(meta.clone()),
                _ => None,
            });
            let Some(meta) = meta else { continue };
            let messages = events
                .iter()
                .filter(|event| matches!(event, SessionEvent::Message { .. }))
                .count();
            let turns = events
                .iter()
                .filter(|event| matches!(event, SessionEvent::TurnEnd { .. }))
                .count();
            summaries.push(SessionSummary {
                id: meta.id.clone(),
                created_at: meta.created_at,
                cwd: meta.cwd.clone(),
                model: meta.model.clone(),
                mode: meta.mode.clone(),
                messages,
                turns,
            });
        }
        summaries.sort_by_key(|summary| std::cmp::Reverse(summary.created_at));
        Ok(summaries)
    }

    pub fn delete(&self, id: &str) -> Result<()> {
        let dir = self.session_dir(id);
        fs::remove_dir_all(&dir)
            .with_context(|| format!("deleting session dir {}", dir.display()))?;
        Ok(())
    }

    /// Replay a session into a transcript + metadata.
    pub fn replay(&self, id: &str) -> Result<(SessionMeta, Vec<SessionEvent>)> {
        let events = self.load(id)?;
        let meta = events
            .iter()
            .find_map(|event| match event {
                SessionEvent::Created { meta } => Some(meta.clone()),
                _ => None,
            })
            .with_context(|| format!("session {} has no created event", id))?;
        Ok((meta, events))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn meta(id: &str) -> SessionMeta {
        SessionMeta {
            id: id.to_string(),
            cwd: "/tmp".to_string(),
            provider: "local".to_string(),
            model: "m".to_string(),
            mode: "build".to_string(),
            system_hash: "s".to_string(),
            tools_hash: "t".to_string(),
            system_prompt: "sys".to_string(),
            created_at: Utc::now(),
        }
    }

    #[test]
    fn append_and_replay_roundtrip() {
        let dir = TempDir::new().unwrap();
        let store = SessionStore::new(dir.path().to_path_buf());
        let id = "s1";
        store.create(meta(id)).unwrap();
        store
            .append(
                id,
                &SessionEvent::Message {
                    message: Message::user("hi"),
                },
            )
            .unwrap();
        let events = store.load(id).unwrap();
        assert_eq!(events.len(), 2);
        let (loaded_meta, _) = store.replay(id).unwrap();
        assert_eq!(loaded_meta.id, id);
    }

    #[test]
    fn list_reports_sessions() {
        let dir = TempDir::new().unwrap();
        let store = SessionStore::new(dir.path().to_path_buf());
        store.create(meta("a")).unwrap();
        store.create(meta("b")).unwrap();
        let list = store.list().unwrap();
        assert_eq!(list.len(), 2);
    }
}

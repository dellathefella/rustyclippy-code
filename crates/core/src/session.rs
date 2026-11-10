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
///
/// Falls back to the pre-rename `~/.deeprust` when the new home does not exist,
/// so existing sessions and configs remain visible after the rename.
pub fn default_home() -> PathBuf {
    if let Ok(home) = std::env::var("RUSTYCLIPPY_HOME").or_else(|_| std::env::var("DEEPRUST_HOME"))
    {
        if !home.trim().is_empty() {
            return PathBuf::from(home);
        }
    }
    let base = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let primary = base.join(".rustyclippy-code");
    if primary.exists() {
        return primary;
    }
    let legacy = base.join(".deeprust");
    if legacy.exists() {
        return legacy;
    }
    primary
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
    /// A deliberate in-session mode switch. The system prompt and tool block are
    /// rebuilt, so the cached prefix is intentionally cold afterwards.
    ModeSwitch {
        mode: String,
        system_prompt: String,
        tools_hash: String,
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
    /// Reasoning tokens reported by the provider (subset of completion).
    #[serde(default)]
    pub reasoning_tokens: usize,
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

    /// Most recent session recorded for a project directory.
    ///
    /// Paths are canonicalized before comparison so `--continue` works from a
    /// symlinked or relative path. Returns `None` when the directory has no
    /// sessions yet.
    pub fn latest_for_cwd(&self, cwd: &Path) -> Result<Option<SessionSummary>> {
        let target = fs::canonicalize(cwd).unwrap_or_else(|_| cwd.to_path_buf());
        let mut best: Option<SessionSummary> = None;
        for summary in self.list()? {
            let recorded = PathBuf::from(&summary.cwd);
            let recorded = fs::canonicalize(&recorded).unwrap_or(recorded);
            if recorded != target {
                continue;
            }
            if best
                .as_ref()
                .map(|current| summary.created_at > current.created_at)
                .unwrap_or(true)
            {
                best = Some(summary);
            }
        }
        Ok(best)
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
    fn a_turn_without_turn_end_is_not_counted() {
        // A turn that bails mid-loop (e.g. the step limit) appends its user and
        // assistant messages but never reaches `SessionEvent::TurnEnd`, so the
        // summary reports zero turns even though the log holds real messages.
        let dir = TempDir::new().unwrap();
        let store = SessionStore::new(dir.path().to_path_buf());
        let id = "failed";
        store.create(meta(id)).unwrap();
        store
            .append(
                id,
                &SessionEvent::Message {
                    message: Message::user("do the thing"),
                },
            )
            .unwrap();
        store
            .append(
                id,
                &SessionEvent::Message {
                    message: Message::assistant("partial"),
                },
            )
            .unwrap();

        let summaries = store.list().unwrap();
        let summary = summaries.iter().find(|s| s.id == id).unwrap();
        assert_eq!(summary.messages, 2, "messages are recorded");
        assert_eq!(summary.turns, 0, "the failed turn is invisible");
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

    fn meta_in(id: &str, cwd: &str, created_at: DateTime<Utc>) -> SessionMeta {
        SessionMeta {
            cwd: cwd.to_string(),
            created_at,
            ..meta(id)
        }
    }

    #[test]
    fn latest_for_cwd_picks_the_newest_session() {
        let dir = TempDir::new().unwrap();
        let store = SessionStore::new(dir.path().to_path_buf());
        let project = TempDir::new().unwrap();
        let cwd = project.path().to_string_lossy().to_string();

        let older = Utc::now() - chrono::Duration::hours(2);
        store.create(meta_in("old", &cwd, older)).unwrap();
        store.create(meta_in("new", &cwd, Utc::now())).unwrap();
        store
            .create(meta_in("other", "/somewhere/else", Utc::now()))
            .unwrap();

        let latest = store.latest_for_cwd(project.path()).unwrap().unwrap();
        assert_eq!(latest.id, "new");
    }

    #[test]
    fn latest_for_cwd_is_none_without_sessions() {
        let dir = TempDir::new().unwrap();
        let store = SessionStore::new(dir.path().to_path_buf());
        let project = TempDir::new().unwrap();
        assert!(store.latest_for_cwd(project.path()).unwrap().is_none());
    }
}

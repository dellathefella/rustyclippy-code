//! Prompt history: recall previous inputs with Up/Down.
//!
//! History is persisted per project directory under the rustyclippy home, so a
//! prompt typed yesterday is still recallable today. The browsing state is kept
//! separate from the stored entries: while the user walks backwards through
//! history the line they were typing is stashed as a draft and restored when
//! they walk forward past the newest entry.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

/// Entries kept per project. Older lines are dropped so the file stays small.
pub const MAX_ENTRIES: usize = 200;

/// Prompt history for one project directory.
#[derive(Debug, Default)]
pub struct History {
    entries: Vec<String>,
    /// Position while browsing; `None` means "at the draft".
    index: Option<usize>,
    /// The line the user was typing before browsing started.
    draft: String,
    /// Where the history file lives; `None` disables persistence.
    path: Option<PathBuf>,
    /// Project key the entries belong to.
    cwd: String,
}

impl History {
    /// Load history for `cwd` from `path`. A missing or corrupt file yields an
    /// empty history rather than an error: history is a convenience, never a
    /// reason to refuse to start.
    pub fn load(cwd: &Path, path: Option<PathBuf>) -> Self {
        let cwd = cwd.to_string_lossy().to_string();
        let entries = path
            .as_ref()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .and_then(|value| {
                value
                    .get(&cwd)
                    .and_then(|entries| entries.as_array())
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(|item| item.as_str().map(|text| text.to_string()))
                            .collect::<Vec<_>>()
                    })
            })
            .unwrap_or_default();
        Self {
            entries,
            index: None,
            draft: String::new(),
            path,
            cwd,
        }
    }

    /// History file under the rustyclippy home.
    pub fn default_path() -> Option<PathBuf> {
        Some(rustyclippy_core::default_home().join("history.json"))
    }

    #[cfg(test)]
    pub fn entries(&self) -> &[String] {
        &self.entries
    }

    /// Record a submitted line. Consecutive duplicates collapse so holding
    /// Enter on the same prompt does not fill the file.
    pub fn push(&mut self, line: &str) {
        let line = line.trim();
        if line.is_empty() {
            return;
        }
        if self.entries.last().map(|last| last.as_str()) == Some(line) {
            return;
        }
        self.entries.push(line.to_string());
        if self.entries.len() > MAX_ENTRIES {
            let excess = self.entries.len() - MAX_ENTRIES;
            self.entries.drain(0..excess);
        }
        self.index = None;
        self.draft.clear();
    }

    /// Walk one entry backwards. `current` is the line being edited, stashed as
    /// the draft on the first step so [`next`](Self::next) can restore it.
    pub fn previous(&mut self, current: &str) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        match self.index {
            None => {
                self.draft = current.to_string();
                self.index = Some(self.entries.len() - 1);
            }
            Some(0) => {}
            Some(index) => self.index = Some(index - 1),
        }
        self.index.map(|index| self.entries[index].clone())
    }

    /// Walk one entry forwards, returning the draft once past the newest entry.
    pub fn next(&mut self) -> Option<String> {
        let index = self.index?;
        if index + 1 < self.entries.len() {
            self.index = Some(index + 1);
            return self.index.map(|index| self.entries[index].clone());
        }
        self.index = None;
        Some(std::mem::take(&mut self.draft))
    }

    /// True while the user is browsing history rather than editing a draft.
    pub fn browsing(&self) -> bool {
        self.index.is_some()
    }

    /// Persist the entries. Failures are ignored: losing history must never
    /// interrupt a session.
    pub fn save(&self) {
        let Some(path) = &self.path else {
            return;
        };
        let mut map: BTreeMap<String, Value> = std::fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .and_then(|value| value.as_object().cloned())
            .map(|object| object.into_iter().collect())
            .unwrap_or_default();
        map.insert(self.cwd.clone(), json!(self.entries));
        let Ok(text) = serde_json::to_string_pretty(&map) else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        // Write through a temp file so a crash mid-write cannot truncate the
        // history of every other project.
        let temp = path.with_extension("json.tmp");
        if std::fs::write(&temp, text).is_ok() {
            let _ = std::fs::rename(&temp, path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history(entries: &[&str]) -> History {
        History {
            entries: entries.iter().map(|line| line.to_string()).collect(),
            ..History::default()
        }
    }

    #[test]
    fn push_records_and_collapses_consecutive_duplicates() {
        let mut history = History::default();
        history.push("first");
        history.push("first");
        history.push("second");
        assert_eq!(history.entries(), ["first", "second"]);
    }

    #[test]
    fn push_ignores_blank_lines() {
        let mut history = History::default();
        history.push("   ");
        history.push("");
        assert!(history.entries().is_empty());
    }

    #[test]
    fn push_caps_the_entry_count() {
        let mut history = History::default();
        for index in 0..MAX_ENTRIES + 10 {
            history.push(&format!("line {index}"));
        }
        assert_eq!(history.entries().len(), MAX_ENTRIES);
        assert_eq!(history.entries()[0], "line 10");
    }

    #[test]
    fn previous_walks_backwards_and_next_restores_the_draft() {
        let mut history = history(&["one", "two", "three"]);
        assert_eq!(history.previous("typing"), Some("three".into()));
        assert_eq!(history.previous("typing"), Some("two".into()));
        assert_eq!(history.previous("typing"), Some("one".into()));
        // At the oldest entry, further presses stay put.
        assert_eq!(history.previous("typing"), Some("one".into()));
        assert_eq!(history.next(), Some("two".into()));
        assert_eq!(history.next(), Some("three".into()));
        assert_eq!(history.next(), Some("typing".into()));
        assert!(!history.browsing());
    }

    #[test]
    fn previous_on_empty_history_is_none() {
        let mut history = History::default();
        assert_eq!(history.previous("typing"), None);
        assert_eq!(history.next(), None);
    }

    #[test]
    fn next_without_browsing_is_none() {
        let mut history = history(&["one"]);
        assert_eq!(history.next(), None);
    }

    #[test]
    fn load_and_save_roundtrip_per_project() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("history.json");
        let project = dir.path().join("project");
        std::fs::create_dir_all(&project).unwrap();

        let mut history = History::load(&project, Some(path.clone()));
        history.push("remember me");
        history.save();

        let reloaded = History::load(&project, Some(path.clone()));
        assert_eq!(reloaded.entries(), ["remember me"]);

        // A different project keeps its own list.
        let other = dir.path().join("other");
        std::fs::create_dir_all(&other).unwrap();
        assert!(History::load(&other, Some(path)).entries().is_empty());
    }

    #[test]
    fn corrupt_history_file_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("history.json");
        std::fs::write(&path, "not json").unwrap();
        assert!(History::load(dir.path(), Some(path)).entries().is_empty());
    }
}

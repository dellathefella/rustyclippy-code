//! Frozen system prompt and environment context.
//!
//! The system prompt is deterministic: no timestamps, no cwd, no model name.
//! Anything volatile lives in a separate environment message appended once at
//! session start, so the cached prefix never changes between turns.

use crate::config::AgentMode;
use crate::message::Message;

/// Prompt revision marker. Bump when the prompt text changes so cache
/// invalidation is explicit rather than accidental.
pub const PROMPT_VERSION: &str = "deeprust/0.1";

/// Build the frozen system prompt for a mode.
pub fn system_prompt(mode: AgentMode) -> String {
    let base = format!(
        r#"You are deeprust ({PROMPT_VERSION}), a coding agent running in the user's terminal.

Operating rules:
- Inspect before editing. Read a file before changing it; never guess file contents.
- Prefer small, verifiable steps. Run tests or builds when the project has them.
- Use the provided tools instead of describing what you would do.
- Keep replies short: state what you did, what you found, and what is next.
- Never claim a command succeeded unless its output shows success.
- If a command fails, read the error and fix the cause rather than retrying blindly.

Tool discipline:
- `read` for files (with offset/limit for large files), `glob` and `grep` to locate things.
- `bash` for commands; keep commands focused and non-interactive.
- Batch independent reads, globs, and greps in a single turn.
- `edit` replaces an exact `old_string`; `write` is for new files only.
- Do not re-read a file you just wrote unless something else may have changed it.

Context discipline:
- The conversation prefix is cached. Do not ask the user to repeat or restate prior context.
- Prefer appending information over restating it.
- When you need a file again, re-read it only if it may have changed."#
    );

    match mode {
        AgentMode::Build => base,
        AgentMode::Plan => format!(
            "{base}\n\n\
             Plan mode is active. You may read files, search, and run read-only commands, \
             but you must not modify files or run commands that change the workspace. \
             Produce a concrete, ordered plan with file paths and the exact changes you \
             would make. Stop when the plan is complete."
        ),
    }
}

/// Stable environment context message, appended once at session start.
///
/// Deliberately excludes date/time and anything that changes between turns.
pub fn environment_message(cwd: &str, model: &str) -> Message {
    let platform = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "sh".to_string());
    Message::user(format!(
        "<environment>\ncwd: {cwd}\nplatform: {platform} ({arch})\nshell: {shell}\nmodel: {model}\n</environment>"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_prompt_is_deterministic() {
        let a = system_prompt(AgentMode::Build);
        let b = system_prompt(AgentMode::Build);
        assert_eq!(a, b);
        assert!(!a.contains("2024"));
        assert!(!a.contains("/Users"));
    }

    #[test]
    fn plan_mode_adds_constraints() {
        let build = system_prompt(AgentMode::Build);
        let plan = system_prompt(AgentMode::Plan);
        assert!(plan.len() > build.len());
        assert!(plan.contains("Plan mode"));
    }

    #[test]
    fn environment_message_is_tagged() {
        let message = environment_message("/tmp", "m");
        assert!(message.content.starts_with("<environment>"));
        assert!(message.content.contains("cwd: /tmp"));
    }
}

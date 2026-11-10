//! Read-only command classification.
//!
//! The agent skips the approval prompt for shell commands that only inspect the
//! workspace, because prompting for every `ls` makes a turn unusable. That
//! shortcut is a privilege boundary, so the classifier is deliberately
//! conservative: anything it cannot prove read-only prompts.
//!
//! The rules are:
//!
//! * Redirection, backgrounding, command substitution and heredocs can write,
//!   so they always prompt.
//! * A binary must be on [`READ_ONLY_COMMANDS`]. The list is narrow on purpose:
//!   binaries that execute their arguments (`env`, `xargs`, `sudo`, `sh`) are
//!   excluded, because `env rm -rf /` would otherwise run unapproved.
//! * Per-binary flags that write (`sed -i`, `sort -o`, `find -exec`) prompt.
//! * `git` is handled separately by [`git_is_read_only`], since a blanket allow
//!   would let `git branch -D` and `git config --global` run unapproved.

/// Commands that only inspect the workspace. Anything else prompts.
///
/// `git` is deliberately absent: it is handled by [`git_is_read_only`], because
/// a blanket allow would let `git branch -D` and `git config --global` run
/// without approval.
///
/// `env`, `printenv`, `xargs`, `sudo`, `sh`, `bash`, `eval` and friends are
/// deliberately absent: they execute their arguments, so allowing them would
/// let any command run without a prompt.
const READ_ONLY_COMMANDS: &[&str] = &[
    "ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "fd", "tree", "file", "stat", "du",
    "pwd", "echo", "which", "type", "date", "uname", "whoami", "basename", "dirname", "realpath",
    "readlink", "diff", "sort", "uniq", "cut", "tr", "sed", "awk", "jq", "true", "false", "test",
];

/// Subcommands that mutate even though the binary is read-only.
const MUTATING_SUBCOMMANDS: &[&str] = &[
    "commit",
    "push",
    "pull",
    "merge",
    "rebase",
    "reset",
    "checkout",
    "switch",
    "clean",
    "rm",
    "mv",
    "add",
    "stash",
    "tag",
    "init",
    "clone",
    "install",
    "publish",
    "run",
    "build",
    "test",
    "fix",
    "fmt",
    "update",
    "remove",
    "delete",
    "apply",
    "restore",
    "cherry-pick",
    "revert",
];

/// Git subcommands that never write to the repository.
const READ_ONLY_GIT_SUBCOMMANDS: &[&str] = &[
    "status",
    "log",
    "diff",
    "show",
    "blame",
    "describe",
    "rev-parse",
    "ls-files",
    "ls-tree",
    "cat-file",
    "shortlog",
    "whatchanged",
    "grep",
    "count-objects",
    "for-each-ref",
    "show-ref",
    "merge-base",
    "name-rev",
    "rev-list",
    "help",
    "version",
    "check-ignore",
    "check-attr",
    "var",
];

/// Git subcommands that read only when their own arguments say so.
const CONDITIONAL_GIT_SUBCOMMANDS: &[&str] = &[
    "branch",
    "tag",
    "remote",
    "stash",
    "config",
    "worktree",
    "notes",
    "submodule",
];

/// Git flags that write, even under an otherwise read-only subcommand.
const MUTATING_GIT_FLAGS: &[&str] = &[
    "-d",
    "-D",
    "-m",
    "-M",
    "-f",
    "--force",
    "--delete",
    "--move",
    "--rename",
    "--set",
    "--unset",
    "--unset-all",
    "--add",
    "--replace-all",
    "--edit",
    "-e",
    "--prune",
    "--update",
    "-u",
    "--hard",
    "--soft",
    "--mixed",
    "--keep",
    "--orphan",
    "--detach",
    "--amend",
    "--squash",
    "--set-upstream-to",
    "--track",
    "--no-track",
    "--unset-upstream",
];

/// Git flags that write a file, even under a read-only subcommand.
///
/// `git log --output=FILE` and `git diff --output=FILE` write to disk, so they
/// must prompt despite `log`/`diff` being read-only subcommands.
const WRITING_GIT_FLAGS: &[&str] = &["--output", "-o"];

/// Git `config` flags that read instead of write.
const READ_ONLY_GIT_CONFIG_FLAGS: &[&str] = &["--get", "--get-all", "--get-regexp", "--list", "-l"];

/// Git subcommands whose read-only form is a specific sub-subcommand.
const READ_ONLY_GIT_SUBSUBCOMMANDS: &[(&str, &[&str])] = &[
    ("stash", &["list", "show"]),
    ("worktree", &["list"]),
    ("notes", &["list", "show"]),
    ("submodule", &["status", "summary"]),
    ("remote", &["show", "get-url"]),
];

/// True when a `git` invocation only inspects the repository.
fn git_is_read_only(words: &[&str]) -> bool {
    // Skip global flags such as `--no-pager`, `-C <dir>` or `--git-dir=<path>`.
    let mut index = 0;
    while index < words.len() && words[index].starts_with('-') {
        index += 1;
    }
    let Some(subcommand) = words.get(index) else {
        return false;
    };
    let rest = &words[index + 1..];

    // `--output=FILE` writes regardless of which subcommand follows.
    if rest.iter().any(|word| writes_git_output(word)) {
        return false;
    }

    if READ_ONLY_GIT_SUBCOMMANDS.contains(subcommand) {
        return true;
    }
    if !CONDITIONAL_GIT_SUBCOMMANDS.contains(subcommand) {
        return false;
    }

    // `git config` writes unless an explicit read flag is present.
    if *subcommand == "config" {
        return rest
            .iter()
            .any(|word| READ_ONLY_GIT_CONFIG_FLAGS.contains(word));
    }

    // Some subcommands read only through a named sub-subcommand.
    if let Some((_, allowed)) = READ_ONLY_GIT_SUBSUBCOMMANDS
        .iter()
        .find(|(name, _)| name == subcommand)
    {
        let Some(first) = rest.first() else {
            return false;
        };
        return allowed.contains(first);
    }

    // `branch`, `tag`: read only when no mutating flag is present *and* no
    // positional argument is given. `git tag v1` and `git branch new` create a
    // ref even though neither carries a mutating flag.
    if rest.iter().any(|word| mutating_git_flag(word)) {
        return false;
    }
    !rest.iter().any(|word| !word.starts_with('-'))
}

/// True when a git flag mutates, matching both `--flag` and `--flag=value`.
fn mutating_git_flag(word: &str) -> bool {
    let name = word.split('=').next().unwrap_or(word);
    MUTATING_GIT_FLAGS.contains(&name)
}

/// True when a git flag writes to a file (`--output=FILE`, `-o FILE`).
fn writes_git_output(word: &str) -> bool {
    WRITING_GIT_FLAGS.iter().any(|flag| {
        word == *flag
            || word.starts_with(&format!("{flag}="))
            || word.starts_with(&format!("{flag}="))
    })
}

/// True when a `sed` script argument can execute a shell command.
///
/// `sed` runs commands when a substitution carries the `e` flag (`s/a/b/e`) or
/// when a line address is followed by `e` (`1e cmd`). Both are rare in normal
/// use, so treating them as mutations costs nothing.
fn sed_script_executes(word: &str) -> bool {
    let script = word.trim_matches(|ch| ch == '\'' || ch == '"');
    // `s/.../.../e` — the trailing flag list contains `e`.
    if let Some(rest) = script.strip_prefix("s") {
        if let Some(last) = rest.rsplit('/').next() {
            if last.contains('e') {
                return true;
            }
        }
    }
    // `1e cmd`, `$e cmd`, `/re/e cmd`.
    script.contains("e ")
}

/// Flags that write for a specific binary. `grep -o` is read-only, so the
/// check has to be per-binary rather than a blanket flag list.
fn binary_flags_mutate(binary: &str, words: &[&str]) -> bool {
    match binary {
        "sed" => words.iter().any(|word| {
            word.starts_with("--in-place")
                || (word.starts_with("-i") && !word.starts_with("--"))
                // `sed 's/a/b/e'` executes the substitution as a command, and
                // `sed -n '1e cmd'` runs `cmd` for line 1.
                || sed_script_executes(word)
        }),
        "sort" => words
            .iter()
            .any(|word| *word == "-o" || word.starts_with("--output")),
        "awk" => words.iter().any(|word| {
            word.starts_with("--in-place")
                || (word.starts_with("-i") && !word.starts_with("--"))
                // `awk 'BEGIN{system("rm -rf /")}'` runs a shell command.
                || word.contains("system(")
                || word.contains("| getline")
                || word.contains("print >")
                || word.contains("print >>")
        }),
        "find" => words.iter().any(|word| {
            matches!(
                *word,
                "-delete"
                    | "-exec"
                    | "-execdir"
                    | "-ok"
                    | "-okdir"
                    | "-fprint"
                    | "-fwrite"
                    | "-fls"
            )
        }),
        _ => false,
    }
}

/// True when a shell command only reads. Conservative: any redirection, any
/// unknown binary, any mutating subcommand, or any mutating flag counts as a
/// mutation.
pub fn is_read_only_command(command: &str) -> bool {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return false;
    }
    // Redirection, backgrounding, command substitution, or a heredoc can write.
    if trimmed.contains('>')
        || trimmed.contains('<')
        || trimmed.contains('`')
        || trimmed.contains("$(")
        || trimmed.contains('&')
        || trimmed.contains("<<")
    {
        return false;
    }
    for segment in trimmed.split(['|', ';', '\n']) {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        let words: Vec<&str> = segment.split_whitespace().collect();
        let Some(binary) = words.first() else {
            continue;
        };
        // Strip a leading path so `/bin/ls` is recognized.
        let binary = binary.rsplit('/').next().unwrap_or(binary);

        if binary == "git" {
            if !git_is_read_only(&words[1..]) {
                return false;
            }
            continue;
        }
        if !READ_ONLY_COMMANDS.contains(&binary) {
            return false;
        }
        // `sed -i`, `sort -o`, `find -delete` write despite a read-only binary.
        if binary_flags_mutate(binary, &words[1..]) {
            return false;
        }
        // `cargo build`, `git commit` and friends write via their subcommand.
        for word in &words[1..] {
            let word = word.trim_start_matches('-');
            if MUTATING_SUBCOMMANDS.contains(&word) {
                return false;
            }
            if word == "i" || word == "delete" || word == "exec" || word == "execdir" {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_only_commands_do_not_prompt() {
        for command in [
            "ls -la",
            "cat src/main.rs",
            "grep -rn foo crates/",
            "git status",
            "git log --oneline",
            "git diff HEAD",
            "find . -name '*.rs'",
            "head -20 README.md | wc -l",
            "/bin/ls",
            // `grep -o` is read-only; the flag check must be per-binary.
            "grep -o foo file.rs",
            "sort file.txt",
            "sed 's/a/b/' file.rs",
            "git branch",
            "git branch --list",
            "git tag",
            "git remote show origin",
            "git stash list",
            "git worktree list",
            "git config --get user.name",
            "git config --list",
            "git rev-parse HEAD",
            "git --no-pager log -1",
        ] {
            assert!(
                is_read_only_command(command),
                "{command} should be read-only"
            );
        }
    }

    #[test]
    fn mutating_git_commands_prompt() {
        // These previously slipped through because `git` was blanket-allowed.
        for command in [
            "git branch -D feature",
            "git branch -d feature",
            "git branch -m old new",
            "git tag -d v1",
            "git tag -f v1",
            "git config --global user.name x",
            "git config user.email x",
            "git config --unset user.name",
            "git remote add origin url",
            "git remote remove origin",
            "git stash",
            "git stash drop",
            "git worktree add ../wt",
            "git gc",
            "git fetch origin",
            "git checkout main",
            "git reset --hard",
            "git clean -fd",
            "git submodule update",
            "git notes add -m x",
        ] {
            assert!(!is_read_only_command(command), "{command} should prompt");
        }
    }

    #[test]
    fn positional_git_args_create_refs_and_prompt() {
        // `git tag v1` and `git branch new` create refs without any mutating
        // flag, so the positional argument itself must count as a mutation.
        for command in [
            "git tag v1",
            "git tag -a v1 -m x",
            "git branch newbranch",
            "git branch --set-upstream-to=origin/main",
        ] {
            assert!(!is_read_only_command(command), "{command} should prompt");
        }
    }

    #[test]
    fn git_output_flags_write_files_and_prompt() {
        // `--output=FILE` writes to disk even under a read-only subcommand.
        for command in [
            "git log --output=/tmp/x",
            "git diff --output=/tmp/x",
            "git show --output=/tmp/x",
            "git log -o /tmp/x",
        ] {
            assert!(!is_read_only_command(command), "{command} should prompt");
        }
    }

    #[test]
    fn argument_executing_binaries_prompt() {
        // `env`, `xargs`, `sudo` and shells execute their arguments, so they
        // must never be treated as read-only.
        for command in [
            "env rm -rf /tmp/x",
            "env sh -c 'rm -rf /tmp/x'",
            "env bash -c 'rm -rf /tmp/x'",
            "env python3 -c 'import os; os.remove(\"/tmp/x\")'",
            "env sudo rm -rf /tmp/x",
            "env xargs rm",
            "env make",
            "env tee /tmp/x",
            "env dd if=/dev/zero of=/tmp/x",
            "env chmod 777 /tmp/x",
            "env ln -s /tmp/a /tmp/b",
            "env cp /tmp/a /tmp/b",
            "env touch /tmp/x",
            "env mkdir /tmp/x",
            "env curl -o /tmp/x http://e.com",
            "env tar -xf a.tar",
            "env unzip a.zip",
            "env perl -e 'unlink \"/tmp/x\"'",
            "env node -e 'require(\"fs\").unlinkSync(\"/tmp/x\")'",
            "env awk 'BEGIN{system(\"rm -rf /tmp/x\")}'",
            "env sort -o /tmp/x /tmp/y",
            "printenv rm -rf /tmp/x",
            "xargs rm",
            "sudo rm -rf /tmp/x",
            "sh -c 'rm -rf /tmp/x'",
            "bash -c 'rm -rf /tmp/x'",
            "eval rm -rf /tmp/x",
        ] {
            assert!(!is_read_only_command(command), "{command} should prompt");
        }
    }

    #[test]
    fn awk_and_sed_command_execution_prompts() {
        for command in [
            "awk 'BEGIN{system(\"rm -rf /tmp/x\")}'",
            "awk '{print > \"/tmp/x\"}' f",
            "sed 's/a/b/e' /tmp/f",
            "sed -n '1e rm -rf /tmp/x' /tmp/f",
        ] {
            assert!(!is_read_only_command(command), "{command} should prompt");
        }
    }

    #[test]
    fn mutating_flags_prompt_even_for_read_only_binaries() {
        for command in [
            "sed --in-place 's/a/b/' f.rs",
            "sed -i.bak 's/a/b/' f.rs",
            "sort -o out.txt in.txt",
            "sort --output=out.txt in.txt",
            "awk -i inplace '{print}' f",
            "find . -exec rm {} \\;",
            "find . -fprint out.txt",
        ] {
            assert!(!is_read_only_command(command), "{command} should prompt");
        }
    }

    #[test]
    fn mutating_commands_still_prompt() {
        for command in [
            "rm -rf target",
            "echo hi > file.txt",
            "git commit -m x",
            "cargo build",
            "sed -i 's/a/b/' file.rs",
            "find . -delete",
            "npm install",
            "cargo check",
            "node script.js",
            "curl http://example.com",
            "ls && rm x",
            "cat $(echo secret)",
            "cat <<EOF",
        ] {
            assert!(!is_read_only_command(command), "{command} should prompt");
        }
    }
}

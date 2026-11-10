# rustyclippy-code

A Rust coding agent for **DeepSeek V4.1 Flash** that keeps its prompt prefix
byte-stable, so a local inference server reuses its KV cache instead of
re-prefilling the whole context every turn.

Local models are expensive to prefill. Most agent harnesses silently invalidate
the prefix on every turn — a fresh timestamp in the system prompt, a reordered
tool list, a re-rendered environment block, or a history rewrite during
compaction. The server then has to re-prefill tens of thousands of tokens before
it can emit a single one. `rustyclippy-code` treats prefix stability as a measured,
enforced invariant.

## Measured effect

Real run against a local MLX server serving
`ddalcu/DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-3.0bpw`:

```text
  · turn 1 step 1 · prompt=2,275 cached=0 (0.0%) prefill=2,275 out=21 · cache cold
  · turn 1 step 2 · prompt=2,340 cached=2,274 (97.2%) prefill=66 out=56 · prefix stable ✓
```

The second step reused **97.2%** of the prompt. Only the 66 new tokens were
prefilled.

## Quick start

```bash
cargo build --release

# Detect your local endpoint, model and reasoning effort, then write a config.
./target/release/rustyclippy-code setup

# Verify the endpoint answers.
./target/release/rustyclippy-code doctor

# Interactive REPL.
./target/release/rustyclippy-code chat
```

The setup wizard reads an existing DSH harness configuration when present
(`~/.dsh/vscode/harness-home/settings.yaml`), probes common local endpoints, and
asks for the model, reasoning effort, and permission policy.

## REPL

The interactive surface is opencode-inspired: a header with model, mode,
endpoint, session and cwd; a dim rule before every answer; a left rail on the
assistant stream; `●` / `└` markers for tool calls and results; and a footer with
the turn's cache accounting. Chrome is written to stderr, assistant text to
stdout, so piped output stays clean.

```text
▌ rustyclippy-code 0.1.0
  model    ddalcu/DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-3.0bpw
  mode     build
  endpoint http://127.0.0.1:11234/v1
  session  20261109-222137-70f8b3
  cwd      /Users/jdella/project
──────────────────────────────────────────────────────────────

› explain the cache guard
──────────────────────────────────────────────────────────────
▌ assistant · turn 1
⠹ the guard hashes every message before each request, so a rewritten
⠸ prefix is caught before it reaches the server…
│ The cache guard hashes every message before each request…
│
  ● read crates/core/src/cache.rs
  └ exit 0 · fn guard(&self, plan: &RequestPlan) -> Result<()> {
── turn 1 · 1 steps · prompt=2,340 cached=2,274 (97.2%) prefill=66 out=56 ──
```

Slash commands:

```text
/exit          quit
/session       show session id
/cache         per-turn cache report
/stats         lifetime token usage for this project
/new           start a fresh session
/mode [build|plan]  show or switch agent mode
/thinking [on|off]  toggle model reasoning output (on by default; the
               spinner animates while the model thinks)
/paste         attach the clipboard image to the next prompt
/image <path>  attach an image file to the next prompt
/images        list pending attachments
/clear-images  drop pending attachments
/help          command list
```

`/mode` rebuilds the frozen system prompt and tool block, so it intentionally
resets the cached prefix: the next request is cold and re-prefills once. The
switch is recorded in the session log and replayed on resume.

### Approvals

Mutating tools ask before they run, but the prompt is designed to stay out of
your way:

- **Read-only shell commands never prompt.** `ls`, `cat`, `grep`, `find`,
  `git status`, `git diff`, `head`, `wc`, `jq` and friends run immediately.
  Anything with a redirection, a pipe into an unknown binary, a command
  substitution, or a mutating subcommand (`git commit`, `sed -i`,
  `find -delete`) still asks.
- **One prompt per batch.** When the model requests several tools in a single
  step, you get one question listing all of them instead of one question per
  call.
- **Enter approves.** The prompt is `[Y/n/a]`: Enter or `y` allows, `n` denies,
  `a` allows everything for the rest of the session.
- **Grants stick.** Approving an edit to a file (or a specific command) records
  it, so later edits to the same path do not re-prompt.

Set `permissions.bash`/`edit`/`write`/`task` to `auto` in the config to skip
prompting entirely, or pass `--yes` on `chat`/`run`.

### Prompt history

**Up/Down** recall previous prompts. History is persisted per project directory
in `~/.rustyclippy-code/history.json`, so prompts survive a restart and stay
separate between projects. The last 200 entries are kept and consecutive
duplicates collapse.

In the REPL, Up recalls history whenever you are composing. In the TUI, Up
recalls history while the input box has text (or while you are already browsing)
and scrolls the transcript when the box is empty; Down walks forward and
restores whatever you were typing once you pass the newest entry.

### Images

Vision-capable servers accept images attached to a prompt. In the REPL, press
**Ctrl+V** to attach the clipboard image directly (or use `/paste`); `/image
<path>` attaches a file; in the TUI, Ctrl+V and an empty bracketed paste both
attach the clipboard image, and the prompt box shows how many are pending. A
pasted or dragged image path in the prompt is detected automatically. `/paste`
reads the clipboard image through native tooling
(`pngpaste`/AppleScript on macOS, `wl-paste`/`xclip` on Linux, PowerShell on
Windows).
Images travel as base64 `data:` URLs in the OpenAI-compatible content-parts
shape, so they stay part of the byte-stable prefix.

Local MLX servers have two vision quirks that the provider works around: they
ignore images on the streaming path when tools/system context are present, and
they reuse cached image placeholder tokens across different images. Image turns
are therefore sent non-streaming with prefix reuse disabled, and the text part
carries an invisible zero-width fingerprint so a new image never hits a stale
cached prefix.

## Commands

```text
rustyclippy-code setup [--yes] [--force] [--path <file>]   interactive setup wizard (alias: init)
rustyclippy-code doctor                                    check endpoint and effective config
rustyclippy-code chat [--session <id>] [--continue] [--mode build|plan] interactive REPL
rustyclippy-code tui [--session <id>] [--continue] [--mode build|plan]  full-screen TUI
rustyclippy-code run <prompt> [--image <path>]... [--session <id>] [--continue] [--json] [--yes] [--max-steps N]
rustyclippy-code stats [--project <path>] [--json]         lifetime token usage for a project
rustyclippy-code sessions list|show <id>|delete <id>
rustyclippy-code cache report [--session <id>] [--json]
rustyclippy-code config init [--path <file>] [--force]
rustyclippy-code config show
```

### TUI

`tui` opens a full-screen interface styled after opencode: a header with the
model, mode, session and endpoint; a scrolling transcript pane; a bordered
prompt box; and a two-row footer with live token totals, cache-hit ratio,
decode speed and a context gauge.

The footer's second row shows how close the next request is to triggering
autocompaction: `ctx 85,000 / 250,000 (34%)`. The figure turns amber past 75%
and red past 90%, so compaction is visible before it fires. Decode speed
(`42.5 tok/s`) appears when the server reports it — llama.cpp exposes it under
`timings.predicted_per_second`; providers that do not report it simply omit the
field.

```text
↑ / ↓ / PgUp / PgDn   scroll the transcript
Home / End            jump to the oldest / newest lines (when the prompt box is empty)
Esc                   jump back to the newest line, or interrupt a running turn
Enter                 submit the prompt
Shift+Tab             cycle build ↔ plan mode
Ctrl+V                attach the clipboard image
Ctrl+C                quit
```

Prompts submitted while a turn is running are queued rather than dropped. They
stack in their own pane directly above the prompt box, dimmed with a `⏳` marker
and numbered in submission order, and each runs once the current turn finishes.
Esc interrupts the turn and clears the queue, since interrupting means
reprompting rather than running the backlog.

Approvals appear in the prompt box instead of on stderr, so a batch of tool
calls is answered without leaving the screen: `Y` allows, `n` denies, `a`
allows for the rest of the session, and `s` chooses each call individually.
`/cache`, `/stats` and `/help` render into the transcript rather than printing
over the frame, and `/clear` empties the transcript pane.

`stats` aggregates every session whose recorded cwd matches the project (the
current directory by default): sessions, turns, prompt/cached/prefill/output/
reasoning tokens, cache-hit ratio, and a per-model breakdown.

### Continuing a session

`--continue` resumes the newest session recorded for the current directory, so
you can pick up where you left off without remembering the session id:

```bash
rustyclippy-code chat --continue
rustyclippy-code run --continue "now add the tests"
```

The lookup canonicalizes paths, so it works from a symlinked or relative
directory. `--session <id>` takes precedence and the two flags are mutually
exclusive. If the directory has no sessions yet, `--continue` fails with a clear
message rather than silently starting a fresh one.

Global overrides: `--config <file>`, `--model <id>`, `--base-url <url>`,
`--reasoning-effort off|low|high|max`.

## Tools

| Tool | Purpose |
| --- | --- |
| `bash` | Run a shell command; `background=true` returns a task id |
| `read` | Read a file with line numbers (`offset`/`limit`) |
| `write` | Create or replace a file |
| `edit` | Replace an exact `old_string` (unique match by default) |
| `glob` | Find files by glob, respecting `.gitignore` |
| `grep` | Regex search with `path:line:content` results |
| `list` | List directory entries |
| `loop` | Poll a command until its output matches a regex |
| `wait` | Wait for a background task and return its output |
| `tasks` | List background tasks with status and output |
| `question` | Ask the user a question and wait for the answer |
| `task` | Delegate a self-contained job to a subagent |
| `todo` | Record the task list |
| `websearch` | DuckDuckGo / Brave / Tavily / SearXNG search |
| `webfetch` | Fetch a URL and return readable text |

Tool schemas are serialized in a fixed, name-sorted order and frozen at session
start. Parallel tool calls are executed concurrently, but results are appended
in call order so the transcript stays deterministic.

## Cache design

The full rationale is in [`docs/CACHE.md`](docs/CACHE.md). The invariants:

1. **Frozen system prompt.** No timestamps, cwd, or model name. The prompt is
   deterministic and stored in the session so resume replays the exact bytes.
2. **Frozen tool block.** Name-sorted, canonically serialized, never changed
   mid-session.
3. **Append-only transcript.** `Transcript` exposes no way to mutate a message.
4. **Volatile context appended once.** cwd/platform/shell/model go into a single
   `<environment>` system-role message at session start, never into the system
   prompt.
5. **Prefix-anchored compaction.** At 250k tokens the agent keeps the pinned head
   (environment + original task) byte-identical, summarizes the middle, and keeps
   the recent tail. The cached head survives; only the summary and tail are
   prefilled.
6. **Cache guard.** Before every request the engine hashes each message and finds
   the longest common prefix with the last committed request. Unexpected
   divergence aborts the turn instead of silently burning a full re-prefill.
7. **Canonical JSON.** Sorted keys everywhere, so logically identical requests
   serialize to identical bytes.
8. **Usage accounting.** DeepSeek's `prompt_cache_hit_tokens`, OpenAI's
   `prompt_tokens_details.cached_tokens`, and llama.cpp's `timings.cache_n` are
   parsed and written to `cache.jsonl`.

### Compaction layout

```text
before:  [ head (pinned) ][ middle (summarized) ][ tail (kept) ]
after:   [ head (identical) ][ summary ][ tail ]
```

The head bytes never change, which is what lets the KV cache survive compaction.

## DeepSeek V4.1 Flash tuning

| Setting | Default | Why |
| --- | --- | --- |
| `max_context_tokens` | `300000` | working window; the model accepts 1M, but recall degrades past ~300k |
| `max_output_tokens` | `65536` | the server's output budget (MLX reports `agent_budget.output`); the architecture allows 384K, but a local server budgets far less |
| `autocompact_at_tokens` | `234464` | window minus output, so a full-length answer still fits |
| `cache.granularity` | `64` | DeepSeek prefix cache block size |
| `reasoning_effort` | `high` | matches the harness knob (`off`/`low`/`high`/`max`) |
| `include_reasoning_in_history` | `false` | replaying reasoning bloats the prefix and reasoner models reject it |

## Configuration

`rustyclippy-code.json` (project) → `~/.rustyclippy-code/config.json` → defaults. Environment
overrides: `RUSTYCLIPPY_BASE_URL`, `RUSTYCLIPPY_MODEL`, `RUSTYCLIPPY_PROVIDER`,
`RUSTYCLIPPY_MAX_CONTEXT`, `RUSTYCLIPPY_AUTOCOMPACT`, `RUSTYCLIPPY_REASONING_EFFORT`,
`RUSTYCLIPPY_MODE`, `RUSTYCLIPPY_HOME`.

See [`rustyclippy-code.example.json`](rustyclippy-code.example.json) for every field.

## Architecture

```text
crates/
  core/   canonical JSON, messages, token estimation, cache engine,
          transcript, prefix-anchored compaction, session store, config
  llm/    OpenAI-compatible + DeepSeek streaming provider, usage parsing
  tools/  bash/read/write/edit/glob/grep/list/loop/question/task/
          tasks/todo/wait/webfetch/websearch
  cli/    agent loop, cache guard, parallel tool execution, REPL,
          setup wizard, reports
```

## Testing

```bash
cargo test --workspace
```

Tests cover canonical serialization, prefix reuse, divergence detection,
prefix-anchored compaction, tool behavior, background tasks, and cache metrics.

## License

MIT

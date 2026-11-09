# deeprust

A Rust coding agent for **DeepSeek V4.1 Flash** that keeps its prompt prefix
byte-stable, so a local inference server reuses its KV cache instead of
re-prefilling the whole context every turn.

Local models are expensive to prefill. Most agent harnesses silently invalidate
the prefix on every turn — a fresh timestamp in the system prompt, a reordered
tool list, a re-rendered environment block, or a history rewrite during
compaction. The server then has to re-prefill tens of thousands of tokens before
it can emit a single one. `deeprust` treats prefix stability as a measured,
enforced invariant.

## Measured effect

Real run against a local MLX server serving
`ddalcu/DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-3.0bpw`:

```text
[turn 1 step 1] prompt=2,275 cached=0     (0.0%) prefill=2,275 out=21 | cache cold
[turn 1 step 2] prompt=2,340 cached=2,274 (97.2%) prefill=66    out=56 | prefix stable ✓
```

The second step reused **97.2%** of the prompt. Only the 66 new tokens were
prefilled.

## Quick start

```bash
cargo build --release

# Detect your local endpoint, model and reasoning effort, then write a config.
./target/release/deeprust setup

# Verify the endpoint answers.
./target/release/deeprust doctor

# Interactive REPL.
./target/release/deeprust chat
```

The setup wizard reads an existing DSH harness configuration when present
(`~/.dsh/vscode/harness-home/settings.yaml`), probes common local endpoints, and
asks for the model, reasoning effort, and permission policy.

## Commands

```text
deeprust setup [--yes] [--force] [--path <file>]   interactive setup wizard (alias: init)
deeprust doctor                                    check endpoint and effective config
deeprust chat [--session <id>] [--mode build|plan] interactive REPL
deeprust run <prompt> [--session <id>] [--json] [--yes] [--max-steps N]
deeprust sessions list|show <id>|delete <id>
deeprust cache report [--session <id>] [--json]
deeprust config init [--path <file>] [--force]
deeprust config show
```

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
   `<environment>` message at session start, never into the system prompt.
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
| `max_context_tokens` | `262144` | 256k window |
| `autocompact_at_tokens` | `250000` | compact before the hard limit, leaving room for output |
| `cache.granularity` | `64` | DeepSeek prefix cache block size |
| `reasoning_effort` | `high` | matches the harness knob (`off`/`low`/`high`/`max`) |
| `include_reasoning_in_history` | `false` | replaying reasoning bloats the prefix and reasoner models reject it |

## Configuration

`deeprust.json` (project) → `~/.deeprust/config.json` → defaults. Environment
overrides: `DEEPRUST_BASE_URL`, `DEEPRUST_MODEL`, `DEEPRUST_PROVIDER`,
`DEEPRUST_MAX_CONTEXT`, `DEEPRUST_AUTOCOMPACT`, `DEEPRUST_REASONING_EFFORT`,
`DEEPRUST_MODE`, `DEEPRUST_HOME`.

See [`deeprust.example.json`](deeprust.example.json) for every field.

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

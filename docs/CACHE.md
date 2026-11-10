# Cache design

This document explains why `rustyclippy-code` is structured the way it is. The goal is
simple: **never re-prefill tokens the server already has.**

## The cost model

A transformer inference server computes the KV cache for a prompt once. If the
next request starts with the same tokens, the server can reuse that cache and
only prefill the new suffix. If any byte in the prefix changes, everything after
that byte must be recomputed.

On a local model this is the dominant cost of an agent loop. A 30k-token context
that is fully re-prefilled every turn can cost more time than generating the
answer. DeepSeek's hosted API behaves the same way, with automatic prefix
caching in 64-token blocks.

So the real question is not "how do I cache?" — it is "how do I guarantee the
prefix never changes?"

## Invariants

### 1. The system prompt is frozen

`system_prompt(mode)` is deterministic. It contains no date, no cwd, no model
name, no session id. The exact text is stored in the session's `Created` event,
so resuming replays the same bytes even if the prompt template changes later.

### 2. The tool block is frozen

`ToolRegistry::standard(mode)` builds a fixed set, sorted by name, and
`CacheEngine::new` hashes the canonical JSON. Adding a tool mid-session would
change the block and invalidate the cache, so the registry never changes during a
session. Plan mode simply has a different (smaller) frozen set.

### 3. Messages are append-only

`Transcript` exposes `append` and `compact` only. There is no method to edit or
remove a message. `Message::wire_hash()` hashes exactly the wire form, excluding
`reasoning` (which is never sent back).

### 4. Volatile context is appended once

cwd, platform, shell, and model live in a single `<environment>` system-role
message at session start. It is part of the transcript, not the system prompt, so
it is stable for the session. If a resumed session runs in a different cwd, a new
environment message is appended — appending is safe; rewriting is not.

### 5. Compaction preserves the head

Compaction is the only operation that rewrites history. Naive compaction throws
away the whole cached prefix. `plan_compaction` keeps a pinned head
(environment + original task), summarizes the middle, and keeps the recent tail:

```text
before:  [ head ][ middle ][ tail ]
after:   [ head ][ summary ][ tail ]
```

`head` is byte-identical, so the KV cache for it survives. The cost of compaction
is the summary plus the tail — bounded and small.

The summary request itself reuses the cached prefix: it sends `head` plus the
summary instruction, with the same frozen system and tool block. Only the
instruction is new.

### 6. The cache guard

`CacheEngine::plan` is pure: it compares the current message hashes against the
last **committed** request and reports the longest common prefix. `commit` records
what was actually sent. Planning twice cannot inflate reuse.

If the system prompt, tool block, or message prefix changed unexpectedly, the
agent refuses to send the request:

```text
cache guard tripped: history diverged at message 3 (5 messages were sent last turn).
Refusing to send a request that would force a full re-prefill.
```

The only allowed rewrite is compaction, marked `expected` when the divergence
point is at or after the pinned head.

### 7. Canonical serialization

Every prompt byte is produced by `canonical_bytes`: object keys sorted, compact
JSON. Two logically identical requests serialize identically, which is what
makes hashing meaningful.

## Accounting

Each turn records:

| Field | Meaning |
| --- | --- |
| `prompt_tokens` | tokens the server processed |
| `cached_tokens` | tokens served from the prefix cache |
| `prefill_tokens` | `prompt_tokens - cached_tokens` |
| `stable_prefix_messages` | messages byte-identical to the last request |
| `prompt_hash` | hash of the full wire prompt |
| `stable` | no unexpected divergence |

`rustyclippy-code cache report` prints the table; `--json` emits the raw records.

The engine also reports its own estimate (`reusable_tokens`, floored to the 64
token granularity) so you can see the plan before the server answers.

## Token estimation

The exact tokenizer is not shipped. `TokenEstimator` counts CJK characters as
one token each and other characters at `chars_per_token` (default 3.6), then
applies an EMA-calibrated scale derived from real `usage` numbers. Compaction
thresholds therefore track the provider's tokenizer instead of drifting.

## Autocompaction at 250k

DeepSeek V4.1 Flash has a 256k window. The agent compacts when

```text
estimated_prompt_tokens + max_output_tokens >= autocompact_at_tokens
```

with `autocompact_at_tokens = 983040` (the 1M window minus the 64k output
budget). This leaves headroom for the response and avoids a hard-context
failure. Compaction is skipped when it would free fewer
than `min_compaction_gain_tokens`, so small overages do not trigger pointless
rewrites.

## Reasoning effort

Reasoning effort is passed through as `reasoning_effort` (`off` omits the field).
Reasoning content is captured for display but **not** replayed into later
requests: replaying it bloats the prefix, and DeepSeek reasoner models reject it
in history. This keeps the wire form stable.

## What still costs a prefill

- **Session resume** is cold by default. The server may have restarted, so the
  agent does not claim reuse it cannot verify. Set `cache.assume_warm_resume` to
  opt into optimistic reuse.
- **Compaction** prefills the summary and tail.
- **New user input and tool results** must be prefilled; that is unavoidable.
  The point is that everything before them is reused.

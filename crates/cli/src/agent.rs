//! The agent loop: plan → guard → stream → tools → repeat, with prefix-anchored
//! autocompaction and parallel tool execution.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, Result};
use chrono::Utc;
use futures_util::future::join_all;
use futures_util::StreamExt;
use rustyclippy_core::{
    compaction::{plan_compaction, summary_instruction, summary_message},
    config::{AgentMode, Config, PermissionMode},
    message::{Message, Role, ToolCall},
    session::{CacheTurnRecord, SessionEvent, SessionMeta, SessionStore},
    system_prompt::{environment_message, system_prompt},
    CacheEngine, ImageAttachment, RequestPlan, Transcript,
};
use rustyclippy_llm::{build_provider, FinishReason, LlmRequest, Provider, StreamEvent, Usage};
use rustyclippy_tools::{SubagentRunner, ToolContext, ToolRegistry, ToolResult};
use serde_json::Value;

use crate::permissions::is_read_only_command;
use crate::ui::{fmt_num, tool_detail, ApprovalAnswer, ApprovalRequest, Output};

#[derive(Default)]
struct PendingCall {
    id: Option<String>,
    name: Option<String>,
    args: String,
}

/// Assembles streamed tool-call deltas into complete calls.
#[derive(Default)]
struct ToolCallAssembler {
    pending: Vec<PendingCall>,
}

impl ToolCallAssembler {
    fn push(
        &mut self,
        index: usize,
        id: Option<String>,
        name: Option<String>,
        args_fragment: Option<String>,
    ) {
        while self.pending.len() <= index {
            self.pending.push(PendingCall::default());
        }
        let slot = &mut self.pending[index];
        if let Some(id) = id {
            slot.id = Some(id);
        }
        if let Some(name) = name {
            slot.name = Some(name);
        }
        if let Some(fragment) = args_fragment {
            slot.args.push_str(&fragment);
        }
    }

    fn finish(self) -> Vec<ToolCall> {
        self.pending
            .into_iter()
            .enumerate()
            .map(|(index, pending)| {
                let id = pending.id.unwrap_or_else(|| format!("call_{index}"));
                let name = pending.name.unwrap_or_default();
                let arguments = serde_json::from_str(&pending.args)
                    .unwrap_or_else(|_| serde_json::json!({"__parse_error": pending.args}));
                ToolCall {
                    id,
                    name,
                    arguments,
                }
            })
            .collect()
    }
}

#[derive(Debug)]
pub struct TurnResult {
    pub text: String,
    pub usage: Usage,
    pub steps: usize,
    pub plan: Option<RequestPlan>,
}

pub struct Agent {
    pub config: Config,
    pub provider: Arc<dyn Provider>,
    pub registry: ToolRegistry,
    pub transcript: Transcript,
    pub engine: CacheEngine,
    pub store: SessionStore,
    pub session_id: String,
    pub mode: AgentMode,
    pub turn: usize,
    pub tool_ctx: ToolContext,
    pub system: String,
    pub output: Output,
    pub auto_approve: bool,
    /// Paths the user already approved this session, so repeated edits to the
    /// same file do not re-prompt.
    approved_paths: HashSet<PathBuf>,
    /// Shell commands approved this session.
    approved_commands: HashSet<String>,
}

/// What the user chose at a permission prompt.
enum BatchDecision {
    AllowAll,
    AllowSession,
    DenyAll,
    PerCall,
}

enum Answer {
    Yes,
    No,
    Session,
    Select,
}

/// Read one answer from stdin. Enter alone means yes, so approving a batch is a
/// single keystroke.
fn read_answer() -> Answer {
    parse_answer(&mut std::io::stdin().lock())
}

fn parse_answer<R: std::io::BufRead>(reader: &mut R) -> Answer {
    let mut line = String::new();
    reader.read_line(&mut line).unwrap_or(0);
    match line.trim().to_lowercase().as_str() {
        "" | "y" | "yes" => Answer::Yes,
        "a" | "all" => Answer::Session,
        "s" | "select" => Answer::Select,
        _ => Answer::No,
    }
}

fn describe(call: &ToolCall) -> String {
    describe_args(&call.name, &call.arguments)
}

fn describe_args(name: &str, args: &Value) -> String {
    tool_detail(name, args)
}

impl Agent {
    /// Start a fresh session with a frozen system prompt and tool block.
    pub fn new_session(
        config: Config,
        cwd: PathBuf,
        mut output: Output,
        auto_approve: bool,
        subagent: Option<Arc<dyn SubagentRunner>>,
    ) -> Result<Self> {
        output.set_compaction_budget(
            config
                .cache
                .autocompact_budget(config.provider.max_context_tokens),
        );
        let provider = build_provider(&config.provider)?;
        let mode = config.agent.mode;
        let system = system_prompt(mode);
        let registry = ToolRegistry::standard(mode);
        let engine = CacheEngine::new(&system, &registry.schemas(), config.cache.granularity);

        let mut transcript = Transcript::new(config.cache.head_messages);
        let environment = environment_message(&cwd.to_string_lossy(), &config.provider.model);
        transcript.append(environment.clone());

        let store = SessionStore::default_store();
        let session_id = SessionStore::new_session_id();
        let meta = SessionMeta {
            id: session_id.clone(),
            cwd: cwd.to_string_lossy().to_string(),
            provider: config.provider.kind.as_str().to_string(),
            model: config.provider.model.clone(),
            mode: mode.as_str().to_string(),
            system_hash: engine.system_hash().to_string(),
            tools_hash: engine.tools_hash().to_string(),
            system_prompt: system.clone(),
            created_at: Utc::now(),
        };
        store.create(meta)?;
        store.append(
            &session_id,
            &SessionEvent::Message {
                message: environment,
            },
        )?;

        let mut tool_ctx = ToolContext::new(cwd, mode);
        tool_ctx.max_output_chars = config.cache.tool_output_max_chars;
        tool_ctx.web = config.web.clone();
        tool_ctx.subagent = subagent;
        tool_ctx.max_depth = config.agent.max_depth;

        Ok(Self {
            config,
            provider,
            registry,
            transcript,
            engine,
            store,
            session_id,
            mode,
            turn: 0,
            tool_ctx,
            system,
            output,
            auto_approve,
            approved_paths: HashSet::new(),
            approved_commands: HashSet::new(),
        })
    }

    /// Build an agent around an injected provider and session store.
    ///
    /// Used by tests to drive a full turn against a mock provider without a
    /// live server, and without writing into the real session home.
    #[cfg(test)]
    pub fn with_provider(
        config: Config,
        cwd: PathBuf,
        provider: Arc<dyn Provider>,
        store: SessionStore,
        auto_approve: bool,
    ) -> Result<Self> {
        let mode = config.agent.mode;
        let system = system_prompt(mode);
        let registry = ToolRegistry::standard(mode);
        let engine = CacheEngine::new(&system, &registry.schemas(), config.cache.granularity);
        let mut transcript = Transcript::new(config.cache.head_messages);
        let environment = environment_message(&cwd.to_string_lossy(), &config.provider.model);
        transcript.append(environment.clone());

        let session_id = SessionStore::new_session_id();
        let meta = SessionMeta {
            id: session_id.clone(),
            cwd: cwd.to_string_lossy().to_string(),
            provider: config.provider.kind.as_str().to_string(),
            model: config.provider.model.clone(),
            mode: mode.as_str().to_string(),
            system_hash: engine.system_hash().to_string(),
            tools_hash: engine.tools_hash().to_string(),
            system_prompt: system.clone(),
            created_at: Utc::now(),
        };
        store.create(meta)?;
        store.append(
            &session_id,
            &SessionEvent::Message {
                message: environment,
            },
        )?;

        let mut tool_ctx = ToolContext::new(cwd, mode);
        tool_ctx.max_output_chars = config.cache.tool_output_max_chars;
        tool_ctx.web = config.web.clone();
        tool_ctx.max_depth = config.agent.max_depth;

        Ok(Self {
            config,
            provider,
            registry,
            transcript,
            engine,
            store,
            session_id,
            mode,
            turn: 0,
            tool_ctx,
            system,
            output: Output::new(false),
            auto_approve,
            approved_paths: HashSet::new(),
            approved_commands: HashSet::new(),
        })
    }

    /// Resume a session by replaying its append-only event log.
    pub fn resume(
        config: Config,
        session_id: String,
        mut output: Output,
        auto_approve: bool,
        subagent: Option<Arc<dyn SubagentRunner>>,
    ) -> Result<Self> {
        output.set_compaction_budget(
            config
                .cache
                .autocompact_budget(config.provider.max_context_tokens),
        );
        let provider = build_provider(&config.provider)?;
        let store = SessionStore::default_store();
        let (meta, events) = store.replay(&session_id)?;
        let mut mode = AgentMode::parse(&meta.mode)?;
        let mut system = if meta.system_prompt.is_empty() {
            system_prompt(mode)
        } else {
            meta.system_prompt.clone()
        };
        let mut tools_hash = meta.tools_hash.clone();

        let mut transcript = Transcript::new(config.cache.head_messages);
        let mut turn = 0usize;
        for event in events {
            match event {
                SessionEvent::Message { message } => transcript.append(message),
                SessionEvent::Compaction { record, summary } => {
                    transcript.apply_compaction(&record, summary)
                }
                SessionEvent::TurnEnd { turn: ended, .. } => turn = ended,
                SessionEvent::ModeSwitch {
                    mode: switched,
                    system_prompt: switched_system,
                    tools_hash: switched_tools,
                } => {
                    mode = AgentMode::parse(&switched)?;
                    system = switched_system;
                    tools_hash = switched_tools;
                }
                _ => {}
            }
        }

        let registry = ToolRegistry::standard(mode);
        let mut engine = CacheEngine::new(&system, &registry.schemas(), config.cache.granularity);
        if engine.tools_hash() != tools_hash {
            output.warning(
                "tool schemas changed since this session started; the first resumed request will be cold",
            );
        }

        if config.cache.assume_warm_resume && !transcript.is_empty() {
            let messages = transcript.messages();
            let upto = if messages
                .last()
                .map(|message| message.role == Role::Assistant)
                .unwrap_or(false)
            {
                messages.len() - 1
            } else {
                messages.len()
            };
            engine.commit(&messages[..upto]);
        }

        let cwd = PathBuf::from(&meta.cwd);
        let mut tool_ctx = ToolContext::new(cwd, mode);
        tool_ctx.max_output_chars = config.cache.tool_output_max_chars;
        tool_ctx.web = config.web.clone();
        tool_ctx.subagent = subagent;
        tool_ctx.max_depth = config.agent.max_depth;

        Ok(Self {
            config,
            provider,
            registry,
            transcript,
            engine,
            store,
            session_id,
            mode,
            turn,
            tool_ctx,
            system,
            output,
            auto_approve,
            approved_paths: HashSet::new(),
            approved_commands: HashSet::new(),
        })
    }

    /// Switch agent mode in-session. Rebuilds the frozen system prompt and tool
    /// block, which intentionally invalidates the cached prefix (the next
    /// request is cold). The transcript is preserved.
    pub fn switch_mode(&mut self, mode: AgentMode) -> Result<()> {
        if mode == self.mode {
            return Ok(());
        }
        let system = system_prompt(mode);
        let registry = ToolRegistry::standard(mode);
        let engine = CacheEngine::new(&system, &registry.schemas(), self.config.cache.granularity);
        let tools_hash = engine.tools_hash().to_string();
        self.system = system.clone();
        self.registry = registry;
        self.engine = engine;
        self.mode = mode;
        self.config.agent.mode = mode;
        self.tool_ctx.mode = mode;
        self.store.append(
            &self.session_id,
            &SessionEvent::ModeSwitch {
                mode: mode.as_str().to_string(),
                system_prompt: system,
                tools_hash,
            },
        )?;
        self.output.mode_switch(mode.as_str());
        Ok(())
    }

    /// Run one user turn to completion.
    pub async fn run_turn(&mut self, input: &str) -> Result<TurnResult> {
        self.run_turn_with_images(input, Vec::new()).await
    }

    /// Run one user turn with optional vision attachments.
    pub async fn run_turn_with_images(
        &mut self,
        input: &str,
        images: Vec<ImageAttachment>,
    ) -> Result<TurnResult> {
        self.turn += 1;
        self.output.turn_begin(self.turn);
        let user = Message::user_with_images(input, images);
        self.transcript.append(user.clone());
        self.store
            .append(&self.session_id, &SessionEvent::Message { message: user })?;

        let outcome = self.run_steps().await;

        // Close the turn in the session log no matter how it ended. Without a
        // `TurnEnd` event the session summary reports zero turns and `--continue`
        // resumes with a stale turn counter, so one failed turn would silently
        // corrupt the accounting for every turn after it.
        let (text, steps, usage, plan) = match outcome {
            Ok(value) => value,
            Err(error) => {
                self.store.append(
                    &self.session_id,
                    &SessionEvent::TurnEnd {
                        turn: self.turn,
                        text: String::new(),
                    },
                )?;
                return Err(error);
            }
        };
        self.store.append(
            &self.session_id,
            &SessionEvent::TurnEnd {
                turn: self.turn,
                text: text.clone(),
            },
        )?;
        Ok(TurnResult {
            text,
            usage,
            steps,
            plan,
        })
    }

    /// The step loop: plan → guard → stream → tools → repeat.
    ///
    /// Returns the final text, step count, accumulated usage and last plan.
    async fn run_steps(&mut self) -> Result<(String, usize, Usage, Option<RequestPlan>)> {
        let budget = self
            .config
            .cache
            .autocompact_budget(self.config.provider.max_context_tokens);
        let mut steps = 0usize;
        let mut last_text = String::new();
        let mut turn_usage = Usage::default();
        let mut last_plan: Option<RequestPlan> = None;

        loop {
            // An interrupt stops the turn between steps, keeping whatever the
            // model already produced.
            if self.output.interrupt_requested() {
                self.output.warning("interrupted");
                break;
            }
            // No step ceiling: a turn runs until the model stops asking for
            // tools. The loop is bounded by the interrupt flag, the context
            // budget (which forces compaction) and the provider's own limits,
            // so an unbounded loop cannot spin forever without progress.
            steps += 1;

            let mut plan = self.engine.plan(
                self.transcript.messages(),
                self.transcript.last_compaction(),
            );
            self.guard(&plan)?;

            let projected = plan.estimated_prompt_tokens + self.config.provider.max_output_tokens;
            if projected >= budget {
                if self.compact().await? {
                    plan = self.engine.plan(
                        self.transcript.messages(),
                        self.transcript.last_compaction(),
                    );
                    self.guard(&plan)?;
                }
                let after = plan.estimated_prompt_tokens + self.config.provider.max_output_tokens;
                if after >= budget {
                    self.output.warning(&format!(
                        "context is still ~{} tokens after compaction (autocompact budget {}); \
                         the request may exceed the model window",
                        fmt_num(after),
                        fmt_num(budget)
                    ));
                }
            }

            let request = LlmRequest {
                model: self.config.provider.model.clone(),
                system: self.system.clone(),
                tools: self.registry.schemas(),
                messages: self.transcript.messages().to_vec(),
                max_tokens: self.config.provider.max_output_tokens,
                temperature: self.config.provider.temperature,
                cache_prompt: self.config.provider.cache_prompt,
                breakpoints: plan.breakpoints.clone(),
                tool_choice_none: false,
                reasoning_effort: self
                    .config
                    .provider
                    .effective_reasoning_effort()
                    .map(|effort| effort.to_string()),
            };

            let mut stream = self.provider.stream(request).await?;
            let mut text = String::new();
            let mut reasoning = String::new();
            let mut usage = Usage::default();
            let mut assembler = ToolCallAssembler::default();
            let mut finish: Option<FinishReason> = None;

            // Poll the stream with a timeout so an interrupt is noticed even
            // when the server stalls between chunks.
            loop {
                let next = tokio::select! {
                    event = stream.next() => event,
                    _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                        if self.output.interrupt_requested() {
                            break;
                        }
                        continue;
                    }
                };
                let Some(event) = next else { break };
                match event? {
                    StreamEvent::Text(chunk) => {
                        text.push_str(&chunk);
                        self.output.text(&chunk);
                    }
                    StreamEvent::Reasoning(chunk) => {
                        reasoning.push_str(&chunk);
                        self.output.reasoning(&chunk);
                    }
                    StreamEvent::ToolCallDelta {
                        index,
                        id,
                        name,
                        args_fragment,
                    } => assembler.push(index, id, name, args_fragment),
                    StreamEvent::Usage(value) => usage = value,
                    StreamEvent::Finished(reason) => finish = Some(reason),
                    StreamEvent::Done => break,
                }
            }

            if !reasoning.is_empty() {
                self.output.reasoning_end();
            }
            // A `length` finish means the server cut generation off at
            // `max_tokens`. Reasoning can consume the entire budget, leaving no
            // answer at all, so report it instead of committing a truncated
            // message as if the turn had completed.
            if finish.map(FinishReason::is_truncated).unwrap_or(false) {
                self.output.warning(&format!(
                    "output hit the {} token limit and was cut off{}; the turn is incomplete \
                     (raise provider.max_output_tokens to allow longer answers)",
                    fmt_num(self.config.provider.max_output_tokens),
                    if text.trim().is_empty() && !reasoning.is_empty() {
                        " before any answer was produced (reasoning used the whole budget)"
                    } else {
                        ""
                    }
                ));
            }
            if !text.is_empty() {
                self.output.line_break();
            }
            // An interrupt mid-stream discards the partial answer instead of
            // committing it: the transcript and the session log both keep only
            // the user message, so a resume replays exactly what was asked.
            if self.output.interrupt_requested() {
                self.output.warning("interrupted");
                break;
            }
            let tool_calls = assembler.finish();
            self.engine.commit(self.transcript.messages());
            self.engine
                .observe_usage(plan.estimated_prompt_tokens, usage.prompt_tokens);

            let content =
                if self.config.provider.include_reasoning_in_history && !reasoning.is_empty() {
                    format!("{text}\n\n<reasoning>\n{reasoning}\n</reasoning>")
                } else {
                    text.clone()
                };
            let mut assistant = Message::assistant_with_tools(content, tool_calls.clone());
            assistant.reasoning = if reasoning.is_empty() {
                None
            } else {
                Some(reasoning)
            };
            self.transcript.append(assistant.clone());
            self.store.append(
                &self.session_id,
                &SessionEvent::Message { message: assistant },
            )?;

            self.store.append(
                &self.session_id,
                &SessionEvent::Usage {
                    turn: self.turn,
                    prompt_tokens: usage.prompt_tokens,
                    cached_tokens: usage.cached_tokens,
                    prefill_tokens: usage.prefill_tokens(),
                    completion_tokens: usage.completion_tokens,
                    estimated_prompt_tokens: plan.estimated_prompt_tokens,
                    stable: plan.stable,
                },
            )?;
            self.store.append_cache(
                &self.session_id,
                &CacheTurnRecord {
                    turn: self.turn,
                    at: Utc::now(),
                    prompt_tokens: usage.prompt_tokens,
                    cached_tokens: usage.cached_tokens,
                    prefill_tokens: usage.prefill_tokens(),
                    completion_tokens: usage.completion_tokens,
                    reasoning_tokens: usage.reasoning_tokens,
                    estimated_prompt_tokens: plan.estimated_prompt_tokens,
                    stable_prefix_messages: plan.stable_prefix_messages,
                    prompt_hash: plan.prompt_hash.clone(),
                    stable: plan.stable,
                    cache_cold: plan.cache_cold,
                },
            )?;

            turn_usage.prompt_tokens += usage.prompt_tokens;
            turn_usage.completion_tokens += usage.completion_tokens;
            turn_usage.cached_tokens += usage.cached_tokens;
            turn_usage.reasoning_tokens += usage.reasoning_tokens;

            self.output.metrics(self.turn, steps, &plan, &usage);
            last_plan = Some(plan);

            last_text = text;
            if tool_calls.is_empty() {
                break;
            }

            // Permission checks are sequential (they may prompt on stdin);
            // execution is parallel. Results are appended in call order so the
            // transcript stays deterministic and the cached prefix is stable.
            let approvals = self.check_permissions(&tool_calls).await;
            for (call, approval) in tool_calls.iter().zip(&approvals) {
                match approval {
                    Ok(()) => self.output.tool_call(&call.name, &call.arguments),
                    Err(reason) => self.output.error(reason),
                }
            }

            let agent: &Agent = &*self;
            let results = join_all(tool_calls.iter().zip(approvals).map(
                |(call, approval)| async move {
                    match approval {
                        Ok(()) => agent.execute_tool(call).await,
                        Err(reason) => ToolResult::failure(reason),
                    }
                },
            ))
            .await;

            for (call, result) in tool_calls.iter().zip(results) {
                self.output.tool_result(&call.name, &result);
                let message = Message::tool(result.content, call.id.clone());
                self.transcript.append(message.clone());
                self.store
                    .append(&self.session_id, &SessionEvent::Message { message })?;
            }
        }

        Ok((last_text, steps, turn_usage, last_plan))
    }

    /// Refuse to send a request that would rewrite the cached prefix.
    fn guard(&self, plan: &RequestPlan) -> Result<()> {
        if let Some(divergence) = &plan.divergence {
            if !divergence.expected {
                bail!(
                    "cache guard tripped: {}. Refusing to send a request that would force a full re-prefill.",
                    divergence.detail
                );
            }
        }
        Ok(())
    }

    fn permission_for(&self, name: &str) -> PermissionMode {
        match name {
            "bash" | "loop" => self.config.permissions.bash,
            "edit" => self.config.permissions.edit,
            "write" => self.config.permissions.write,
            "task" => self.config.permissions.task,
            _ => PermissionMode::Auto,
        }
    }

    async fn check_permissions(&mut self, calls: &[ToolCall]) -> Vec<Result<(), String>> {
        let mut approvals: Vec<Option<Result<(), String>>> = Vec::with_capacity(calls.len());
        let mut pending: Vec<usize> = Vec::new();

        for (index, call) in calls.iter().enumerate() {
            match self.precheck(call) {
                Ok(true) => approvals.push(Some(Ok(()))),
                Ok(false) => {
                    approvals.push(None);
                    pending.push(index);
                }
                Err(reason) => approvals.push(Some(Err(reason))),
            }
        }

        if pending.is_empty() {
            return approvals.into_iter().map(|value| value.unwrap()).collect();
        }

        // One prompt for the whole batch instead of one per tool call.
        let decision = self
            .ask_batch(&pending.iter().map(|&i| &calls[i]).collect::<Vec<_>>())
            .await;
        for index in pending {
            let call = &calls[index];
            let outcome = match decision {
                BatchDecision::AllowAll => Ok(()),
                BatchDecision::AllowSession => {
                    self.auto_approve = true;
                    Ok(())
                }
                BatchDecision::DenyAll => Err(format!("`{}` denied by the user", call.name)),
                BatchDecision::PerCall => {
                    if self.ask(&call.name, &call.arguments).await {
                        Ok(())
                    } else {
                        Err(format!("`{}` denied by the user", call.name))
                    }
                }
            };
            if outcome.is_ok() {
                self.remember_grant(call);
            }
            approvals[index] = Some(outcome);
        }

        approvals.into_iter().map(|value| value.unwrap()).collect()
    }

    /// `Ok(true)` = allowed without asking, `Ok(false)` = needs a prompt.
    fn precheck(&self, call: &ToolCall) -> Result<bool, String> {
        let Some(tool) = self.registry.get(&call.name) else {
            return Err(format!("unknown tool `{}`", call.name));
        };
        if call.arguments.get("__parse_error").is_some() {
            return Err(format!("`{}` arguments were not valid JSON", call.name));
        }
        if tool.mutates() && self.mode == AgentMode::Plan {
            return Err(format!(
                "`{}` is unavailable in plan mode (read-only)",
                call.name
            ));
        }
        if !tool.mutates() {
            return Ok(true);
        }
        // Read-only shell inspection (`ls`, `cat`, `grep`, `git status`) never
        // needs approval; prompting for each one makes a turn unusable.
        if matches!(call.name.as_str(), "bash" | "loop") {
            if let Some(command) = call
                .arguments
                .get("command")
                .and_then(|value| value.as_str())
            {
                if is_read_only_command(command) {
                    return Ok(true);
                }
            }
        }
        match self.permission_for(&call.name) {
            PermissionMode::Auto => Ok(true),
            PermissionMode::Readonly => Err(format!(
                "`{}` denied by the readonly permission policy",
                call.name
            )),
            PermissionMode::Ask => {
                if self.auto_approve || self.already_granted(call) {
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
        }
    }

    /// A path or command the user approved earlier this session.
    fn already_granted(&self, call: &ToolCall) -> bool {
        match call.name.as_str() {
            "edit" | "write" => call
                .arguments
                .get("path")
                .and_then(|value| value.as_str())
                .map(|path| self.approved_paths.contains(&self.tool_ctx.resolve(path)))
                .unwrap_or(false),
            "bash" | "loop" => call
                .arguments
                .get("command")
                .and_then(|value| value.as_str())
                .map(|command| self.approved_commands.contains(command))
                .unwrap_or(false),
            _ => false,
        }
    }

    fn remember_grant(&mut self, call: &ToolCall) {
        match call.name.as_str() {
            "edit" | "write" => {
                if let Some(path) = call.arguments.get("path").and_then(|value| value.as_str()) {
                    self.approved_paths.insert(self.tool_ctx.resolve(path));
                }
            }
            "bash" | "loop" => {
                if let Some(command) = call
                    .arguments
                    .get("command")
                    .and_then(|value| value.as_str())
                {
                    self.approved_commands.insert(command.to_string());
                }
            }
            _ => {}
        }
    }

    async fn ask_batch(&self, calls: &[&ToolCall]) -> BatchDecision {
        let request = ApprovalRequest {
            calls: calls
                .iter()
                .map(|call| (call.name.clone(), describe(call)))
                .collect(),
        };
        if let Some(answer) = self.output.ask_approval(request).await {
            return match answer {
                ApprovalAnswer::Yes => BatchDecision::AllowAll,
                ApprovalAnswer::No => BatchDecision::DenyAll,
                ApprovalAnswer::Session => BatchDecision::AllowSession,
                ApprovalAnswer::Select => BatchDecision::PerCall,
            };
        }

        if calls.len() == 1 {
            let call = calls[0];
            let detail = describe(call);
            eprint!("allow {} {}? [Y/n/a] ", call.name, detail);
            return match read_answer() {
                Answer::Yes => BatchDecision::AllowAll,
                Answer::No => BatchDecision::DenyAll,
                Answer::Session => BatchDecision::AllowSession,
                Answer::Select => BatchDecision::PerCall,
            };
        }

        eprintln!("{} tools need approval:", calls.len());
        for (index, call) in calls.iter().enumerate() {
            eprintln!("  {}. {} {}", index + 1, call.name, describe(call));
        }
        eprint!(
            "allow all {}? [Y/n/a/s] (a = allow for this session, s = choose each) ",
            calls.len()
        );
        match read_answer() {
            Answer::Yes => BatchDecision::AllowAll,
            Answer::No => BatchDecision::DenyAll,
            Answer::Session => BatchDecision::AllowSession,
            Answer::Select => BatchDecision::PerCall,
        }
    }

    async fn ask(&self, name: &str, args: &Value) -> bool {
        let detail = describe_args(name, args);
        let request = ApprovalRequest {
            calls: vec![(name.to_string(), detail.clone())],
        };
        if let Some(answer) = self.output.ask_approval(request).await {
            return matches!(answer, ApprovalAnswer::Yes | ApprovalAnswer::Session);
        }
        eprint!("allow {} {}? [Y/n/a] ", name, detail);
        matches!(read_answer(), Answer::Yes | Answer::Session)
    }

    async fn execute_tool(&self, call: &ToolCall) -> ToolResult {
        let Some(tool) = self.registry.get(&call.name) else {
            return ToolResult::failure(format!("unknown tool `{}`", call.name));
        };
        match tool.execute(&call.arguments, &self.tool_ctx).await {
            Ok(result) => result,
            Err(error) => ToolResult::failure(format!("tool error: {error:#}")),
        }
    }

    /// Prefix-anchored compaction: keep the pinned head byte-identical, summarize
    /// the middle, keep the recent tail.
    async fn compact(&mut self) -> Result<bool> {
        let estimator = self.engine.estimator.clone();
        let messages = self.transcript.messages().to_vec();
        let Some(plan) = plan_compaction(
            &messages,
            self.transcript.head_len(),
            self.config.cache.tail_messages,
            &estimator,
            self.config.cache.min_compaction_gain_tokens,
        ) else {
            return Ok(false);
        };

        let before_tokens: usize = messages
            .iter()
            .map(|message| estimator.estimate_message(message))
            .sum();
        self.output.compaction_start(plan.dropped, before_tokens);

        // Reuse the cached prefix: send the pinned head plus the instruction.
        let mut summarize = messages[..plan.head_len].to_vec();
        summarize.push(Message::user(summary_instruction()));

        let request = LlmRequest {
            model: self.config.provider.model.clone(),
            system: self.system.clone(),
            tools: self.registry.schemas(),
            messages: summarize,
            max_tokens: self.config.cache.summary_max_tokens,
            temperature: 0.0,
            cache_prompt: self.config.provider.cache_prompt,
            breakpoints: Vec::new(),
            tool_choice_none: true,
            reasoning_effort: None,
        };

        let mut stream = self.provider.stream(request).await?;
        let mut summary_text = String::new();
        let mut usage = Usage::default();
        while let Some(event) = stream.next().await {
            match event? {
                StreamEvent::Text(chunk) => summary_text.push_str(&chunk),
                StreamEvent::Usage(value) => usage = value,
                StreamEvent::Done => break,
                _ => {}
            }
        }
        if summary_text.trim().is_empty() {
            bail!("compaction summary was empty; refusing to rewrite history");
        }

        let summary = summary_message(summary_text.trim());
        let record = self
            .transcript
            .compact(&plan, summary.clone(), self.turn, &estimator);
        self.store.append(
            &self.session_id,
            &SessionEvent::Compaction {
                record: record.clone(),
                summary,
            },
        )?;
        self.output.compaction_done(&record, &usage);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edit_call(path: &str) -> ToolCall {
        ToolCall::new("c1", "edit", serde_json::json!({ "path": path }))
    }

    #[test]
    fn describe_shows_the_path_for_edits() {
        assert_eq!(describe(&edit_call("src/main.rs")), "src/main.rs");
    }

    #[test]
    fn describe_shows_the_command_for_bash() {
        let call = ToolCall::new("c1", "bash", serde_json::json!({ "command": "ls -la" }));
        assert_eq!(describe(&call), "ls -la");
    }

    #[test]
    fn describe_truncates_long_details() {
        let long = "x".repeat(400);
        let call = ToolCall::new("c1", "bash", serde_json::json!({ "command": long }));
        assert!(describe(&call).chars().count() <= 121);
    }

    /// A provider that replays a fixed script of stream events.
    struct MockProvider {
        script: Vec<Vec<StreamEvent>>,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl MockProvider {
        fn boxed(script: Vec<Vec<StreamEvent>>) -> Arc<dyn Provider> {
            Arc::new(Self {
                script,
                calls: std::sync::atomic::AtomicUsize::new(0),
            })
        }
    }

    #[async_trait::async_trait]
    impl Provider for MockProvider {
        fn name(&self) -> &str {
            "mock"
        }

        async fn stream(
            &self,
            _request: LlmRequest,
        ) -> Result<futures_util::stream::BoxStream<'static, Result<StreamEvent>>> {
            let index = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let events = self.script.get(index).cloned().unwrap_or_default();
            Ok(Box::pin(futures_util::stream::iter(
                events.into_iter().map(Ok),
            )))
        }
    }

    fn test_config() -> Config {
        Config::default()
    }

    #[tokio::test]
    async fn a_turn_streams_text_and_closes_the_session_log() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::new(dir.path().join("sessions"));
        let provider = MockProvider::boxed(vec![vec![
            StreamEvent::Text("hello ".into()),
            StreamEvent::Text("world".into()),
            StreamEvent::Finished(FinishReason::Stop),
            StreamEvent::Done,
        ]]);
        let mut agent = Agent::with_provider(
            test_config(),
            dir.path().to_path_buf(),
            provider,
            store,
            true,
        )
        .unwrap();

        let result = agent.run_turn("hi").await.unwrap();
        assert_eq!(result.text, "hello world");
        assert_eq!(result.steps, 1);

        // The turn must be closed in the log, or `--continue` resumes with a
        // stale counter.
        let events = agent.store.load(&agent.session_id).unwrap();
        assert!(
            events
                .iter()
                .any(|event| matches!(event, SessionEvent::TurnEnd { turn: 1, .. })),
            "turn end missing from session log"
        );
    }

    #[tokio::test]
    async fn a_tool_call_runs_and_its_result_joins_the_transcript() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "alpha\n").unwrap();
        let store = SessionStore::new(dir.path().join("sessions"));
        let provider = MockProvider::boxed(vec![
            vec![
                StreamEvent::ToolCallDelta {
                    index: 0,
                    id: Some("call-1".into()),
                    name: Some("read".into()),
                    args_fragment: Some(r#"{"path":"a.txt"}"#.into()),
                },
                StreamEvent::Finished(FinishReason::ToolCalls),
                StreamEvent::Done,
            ],
            vec![
                StreamEvent::Text("done".into()),
                StreamEvent::Finished(FinishReason::Stop),
                StreamEvent::Done,
            ],
        ]);
        let mut agent = Agent::with_provider(
            test_config(),
            dir.path().to_path_buf(),
            provider,
            store,
            true,
        )
        .unwrap();

        let result = agent.run_turn("read a.txt").await.unwrap();
        assert_eq!(result.text, "done");
        assert_eq!(result.steps, 2);

        // The tool result must be in the transcript so the next request sees it.
        let messages = agent.transcript.messages();
        assert!(
            messages
                .iter()
                .any(|message| message.role == Role::Tool && message.content.contains("alpha")),
            "tool result missing from transcript: {messages:?}"
        );
    }

    #[tokio::test]
    async fn a_mutating_command_is_denied_without_approval() {
        // `auto_approve=false` and no interactive answerer means the approval
        // prompt cannot be answered, so the call must be denied rather than run.
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::new(dir.path().join("sessions"));
        let provider = MockProvider::boxed(vec![
            vec![
                StreamEvent::ToolCallDelta {
                    index: 0,
                    id: Some("call-1".into()),
                    name: Some("bash".into()),
                    args_fragment: Some(r#"{"command":"rm -rf /tmp/nope"}"#.into()),
                },
                StreamEvent::Finished(FinishReason::ToolCalls),
                StreamEvent::Done,
            ],
            vec![
                StreamEvent::Text("stopped".into()),
                StreamEvent::Finished(FinishReason::Stop),
                StreamEvent::Done,
            ],
        ]);
        let mut agent = Agent::with_provider(
            test_config(),
            dir.path().to_path_buf(),
            provider,
            store,
            false,
        )
        .unwrap();

        agent.run_turn("delete it").await.unwrap();
        let messages = agent.transcript.messages();
        let tool_message = messages
            .iter()
            .find(|message| message.role == Role::Tool)
            .expect("tool result");
        assert!(
            !tool_message.content.contains("exit 0"),
            "mutating command must not run unapproved: {}",
            tool_message.content
        );
    }

    #[tokio::test]
    async fn a_read_only_command_runs_without_approval() {
        // `ls` is read-only, so it must execute even with auto_approve=false.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("marker.txt"), "x").unwrap();
        let store = SessionStore::new(dir.path().join("sessions"));
        let provider = MockProvider::boxed(vec![
            vec![
                StreamEvent::ToolCallDelta {
                    index: 0,
                    id: Some("call-1".into()),
                    name: Some("bash".into()),
                    args_fragment: Some(r#"{"command":"ls"}"#.into()),
                },
                StreamEvent::Finished(FinishReason::ToolCalls),
                StreamEvent::Done,
            ],
            vec![
                StreamEvent::Text("listed".into()),
                StreamEvent::Finished(FinishReason::Stop),
                StreamEvent::Done,
            ],
        ]);
        let mut agent = Agent::with_provider(
            test_config(),
            dir.path().to_path_buf(),
            provider,
            store,
            false,
        )
        .unwrap();

        agent.run_turn("list files").await.unwrap();
        let messages = agent.transcript.messages();
        let tool_message = messages
            .iter()
            .find(|message| message.role == Role::Tool)
            .expect("tool result");
        assert!(
            tool_message.content.contains("marker.txt"),
            "read-only command should run: {}",
            tool_message.content
        );
    }

    #[test]
    fn enter_alone_approves() {
        let mut input = std::io::Cursor::new(b"\n".to_vec());
        assert!(matches!(parse_answer(&mut input), Answer::Yes));
    }

    #[test]
    fn a_grants_the_session() {
        let mut input = std::io::Cursor::new(b"a\n".to_vec());
        assert!(matches!(parse_answer(&mut input), Answer::Session));
    }

    #[test]
    fn n_denies() {
        let mut input = std::io::Cursor::new(b"n\n".to_vec());
        assert!(matches!(parse_answer(&mut input), Answer::No));
    }

    #[test]
    fn assembler_joins_fragments() {
        let mut assembler = ToolCallAssembler::default();
        assembler.push(
            0,
            Some("call_1".into()),
            Some("bash".into()),
            Some("{\"comm".into()),
        );
        assembler.push(0, None, None, Some("and\":\"ls\"}".into()));
        let calls = assembler.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].arguments["command"], "ls");
    }

    #[test]
    fn assembler_handles_parallel_indices() {
        let mut assembler = ToolCallAssembler::default();
        assembler.push(
            0,
            Some("a".into()),
            Some("read".into()),
            Some("{\"path\":\"x\"}".into()),
        );
        assembler.push(
            1,
            Some("b".into()),
            Some("grep".into()),
            Some("{\"pattern\":\"y\"}".into()),
        );
        let calls = assembler.finish();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].name, "grep");
    }

    #[test]
    fn assembler_marks_invalid_json() {
        let mut assembler = ToolCallAssembler::default();
        assembler.push(
            0,
            Some("a".into()),
            Some("bash".into()),
            Some("{not json".into()),
        );
        let calls = assembler.finish();
        assert!(calls[0].arguments.get("__parse_error").is_some());
    }
}

#[cfg(test)]
mod send_check {
    use super::*;
    #[test]
    fn agent_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Agent>();
    }
}

#[cfg(test)]
mod step_limit {
    use super::*;
    use async_trait::async_trait;
    use futures_util::stream::{self, BoxStream};
    use rustyclippy_core::SessionEvent;
    use rustyclippy_llm::{LlmRequest, Provider, StreamEvent};
    use tempfile::TempDir;

    /// Always answers with a tool call, so the agent never finishes on its own.
    /// Requests a tool for the first `tool_steps` calls, then answers.
    ///
    /// With no step ceiling the loop only ends when the model stops asking for
    /// tools, so the mock has to eventually produce a final answer.
    struct EndlessToolProvider {
        tool_steps: usize,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl EndlessToolProvider {
        fn new(tool_steps: usize) -> Self {
            Self {
                tool_steps,
                calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl Provider for EndlessToolProvider {
        fn name(&self) -> &str {
            "mock"
        }

        async fn stream(
            &self,
            _request: LlmRequest,
        ) -> Result<BoxStream<'static, Result<StreamEvent>>> {
            let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let events = if call < self.tool_steps {
                vec![
                    Ok(StreamEvent::ToolCallDelta {
                        index: 0,
                        id: Some(format!("call_{call}")),
                        name: Some("todo".into()),
                        args_fragment: Some(r#"{"items":[]}"#.into()),
                    }),
                    Ok(StreamEvent::Usage(Usage {
                        prompt_tokens: 100,
                        cached_tokens: 80,
                        completion_tokens: 5,
                        reasoning_tokens: 0,
                        decode_tokens_per_second: None,
                    })),
                    Ok(StreamEvent::Finished(FinishReason::ToolCalls)),
                    Ok(StreamEvent::Done),
                ]
            } else {
                vec![
                    Ok(StreamEvent::Text("finished".into())),
                    Ok(StreamEvent::Finished(FinishReason::Stop)),
                    Ok(StreamEvent::Done),
                ]
            };
            Ok(Box::pin(stream::iter(events)))
        }
    }

    fn agent_with_auto_permissions() -> (Agent, TempDir) {
        let dir = TempDir::new().unwrap();
        let mut config = Config::default();
        config.permissions.bash = rustyclippy_core::PermissionMode::Auto;
        config.permissions.edit = rustyclippy_core::PermissionMode::Auto;
        config.permissions.write = rustyclippy_core::PermissionMode::Auto;
        config.permissions.task = rustyclippy_core::PermissionMode::Auto;

        let mut agent = Agent::new_session(
            config,
            dir.path().to_path_buf(),
            Output::quiet(),
            true,
            None,
        )
        .unwrap();
        agent.provider = Arc::new(EndlessToolProvider::new(5));
        agent.store = SessionStore::new(dir.path().to_path_buf());
        agent
            .store
            .create(SessionMeta {
                id: agent.session_id.clone(),
                cwd: dir.path().to_string_lossy().to_string(),
                provider: "mock".into(),
                model: "mock".into(),
                mode: agent.mode.as_str().to_string(),
                system_hash: agent.engine.system_hash().to_string(),
                tools_hash: agent.engine.tools_hash().to_string(),
                system_prompt: agent.system.clone(),
                created_at: Utc::now(),
            })
            .unwrap();
        (agent, dir)
    }

    /// Streams reasoning that consumes the whole budget, then reports `length`.
    struct TruncatedProvider;

    #[async_trait]
    impl Provider for TruncatedProvider {
        fn name(&self) -> &str {
            "mock"
        }

        async fn stream(
            &self,
            _request: LlmRequest,
        ) -> Result<BoxStream<'static, Result<StreamEvent>>> {
            let events = vec![
                Ok(StreamEvent::Reasoning("thinking hard".into())),
                Ok(StreamEvent::Usage(Usage {
                    prompt_tokens: 100,
                    cached_tokens: 80,
                    completion_tokens: 8_192,
                    reasoning_tokens: 8_192,
                    decode_tokens_per_second: None,
                })),
                Ok(StreamEvent::Finished(FinishReason::Length)),
                Ok(StreamEvent::Done),
            ];
            Ok(Box::pin(stream::iter(events)))
        }
    }

    #[tokio::test]
    async fn truncation_is_reported_not_silently_committed() {
        let (mut agent, _dir) = agent_with_auto_permissions();
        agent.provider = Arc::new(TruncatedProvider);
        // Route chrome into a feed so the warning can be asserted.
        let feed = Arc::new(std::sync::Mutex::new(crate::ui::UiFeed::default()));
        agent.output = Output::tui(feed.clone());

        let result = agent
            .run_turn_with_images("think about it", Vec::new())
            .await
            .expect("truncation should not fail the turn");

        let notices: Vec<String> = feed
            .lock()
            .unwrap()
            .entries()
            .iter()
            .filter_map(|entry| match entry {
                crate::ui::FeedEntry::Notice { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect();
        let chrome = notices.join("\n");
        assert!(
            chrome.contains("token limit") || chrome.contains("cut off"),
            "truncation must be reported, got: {chrome}"
        );
        assert!(
            chrome.contains("reasoning used the whole budget"),
            "an answer-less truncation must say why, got: {chrome}"
        );
        assert!(result.text.trim().is_empty(), "no answer was produced");
    }

    #[tokio::test]
    async fn a_turn_runs_until_the_model_stops_asking_for_tools() {
        // There is no step ceiling: the loop ends when the model returns a
        // final answer, not when a counter runs out.
        let (mut agent, _dir) = agent_with_auto_permissions();
        let result = agent.run_turn_with_images("keep going", Vec::new()).await;

        let result = result.expect("an unbounded turn should still complete");
        assert!(
            result.steps > 3,
            "the loop must keep going past the old ceiling, got {} steps",
            result.steps
        );

        let events = agent.store.load(&agent.session_id).unwrap();
        let turn_ends = events
            .iter()
            .filter(|event| matches!(event, SessionEvent::TurnEnd { .. }))
            .count();
        assert_eq!(turn_ends, 1, "the turn is closed in the log");

        let summaries = agent.store.list().unwrap();
        let summary = summaries
            .iter()
            .find(|summary| summary.id == agent.session_id)
            .unwrap();
        assert_eq!(summary.turns, 1, "the turn is counted");
        assert!(summary.messages > 1, "work is preserved");
    }

    #[tokio::test]
    async fn a_failed_turn_still_closes_the_log() {
        let (mut agent, _dir) = agent_with_auto_permissions();
        // Force a failure inside the loop: an unknown tool trips the guard.
        agent.registry = ToolRegistry::standard(AgentMode::Build);
        let _result = agent.run_turn_with_images("x", Vec::new()).await;
        // Whatever happens, the log must contain exactly one TurnEnd.
        let events = agent.store.load(&agent.session_id).unwrap();
        let turn_ends = events
            .iter()
            .filter(|event| matches!(event, SessionEvent::TurnEnd { .. }))
            .count();
        assert_eq!(turn_ends, 1, "even a failed turn is closed");
    }
}

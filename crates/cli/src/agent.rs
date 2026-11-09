//! The agent loop: plan → guard → stream → tools → repeat, with prefix-anchored
//! autocompaction and parallel tool execution.

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
use rustyclippy_llm::{build_provider, LlmRequest, Provider, StreamEvent, Usage};
use rustyclippy_tools::{SubagentRunner, ToolContext, ToolRegistry, ToolResult};
use serde_json::Value;

use crate::ui::{fmt_num, Output};

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
}

impl Agent {
    /// Start a fresh session with a frozen system prompt and tool block.
    pub fn new_session(
        config: Config,
        cwd: PathBuf,
        output: Output,
        auto_approve: bool,
        subagent: Option<Arc<dyn SubagentRunner>>,
    ) -> Result<Self> {
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
        })
    }

    /// Resume a session by replaying its append-only event log.
    pub fn resume(
        config: Config,
        session_id: String,
        output: Output,
        auto_approve: bool,
        subagent: Option<Arc<dyn SubagentRunner>>,
    ) -> Result<Self> {
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
    #[allow(unused_assignments)]
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

        let budget = self
            .config
            .cache
            .autocompact_budget(self.config.provider.max_context_tokens);
        let mut steps = 0usize;
        let mut final_text = String::new();
        let mut turn_usage = Usage::default();
        let mut last_plan: Option<RequestPlan> = None;

        loop {
            steps += 1;
            if steps > self.config.agent.max_steps {
                bail!(
                    "agent exceeded {} steps without finishing",
                    self.config.agent.max_steps
                );
            }

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

            while let Some(event) = stream.next().await {
                match event? {
                    StreamEvent::Text(chunk) => {
                        text.push_str(&chunk);
                        self.output.text(&chunk);
                    }
                    StreamEvent::Reasoning(chunk) => reasoning.push_str(&chunk),
                    StreamEvent::ToolCallDelta {
                        index,
                        id,
                        name,
                        args_fragment,
                    } => assembler.push(index, id, name, args_fragment),
                    StreamEvent::Usage(value) => usage = value,
                    StreamEvent::Done => break,
                }
            }

            if !text.is_empty() {
                self.output.line_break();
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

            if tool_calls.is_empty() {
                final_text = text;
                break;
            }

            // Permission checks are sequential (they may prompt on stdin);
            // execution is parallel. Results are appended in call order so the
            // transcript stays deterministic and the cached prefix is stable.
            let approvals: Vec<Result<(), String>> = tool_calls
                .iter()
                .map(|call| self.check_permission(call))
                .collect();
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

        self.store.append(
            &self.session_id,
            &SessionEvent::TurnEnd {
                turn: self.turn,
                text: final_text.clone(),
            },
        )?;

        Ok(TurnResult {
            text: final_text,
            usage: turn_usage,
            steps,
            plan: last_plan,
        })
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

    fn check_permission(&self, call: &ToolCall) -> Result<(), String> {
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
            return Ok(());
        }
        match self.permission_for(&call.name) {
            PermissionMode::Auto => Ok(()),
            PermissionMode::Readonly => Err(format!(
                "`{}` denied by the readonly permission policy",
                call.name
            )),
            PermissionMode::Ask => {
                if self.auto_approve || self.ask(&call.name, &call.arguments) {
                    Ok(())
                } else {
                    Err(format!("`{}` denied by the user", call.name))
                }
            }
        }
    }

    fn ask(&self, name: &str, args: &Value) -> bool {
        let detail = match name {
            "bash" | "loop" => args
                .get("command")
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .to_string(),
            "edit" | "write" => args
                .get("path")
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .to_string(),
            "task" => args
                .get("description")
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .to_string(),
            _ => String::new(),
        };
        eprintln!("allow {} {}? [y/N] ", name, detail);
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap_or(0);
        matches!(line.trim().to_lowercase().as_str(), "y" | "yes")
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

//! OpenAI-compatible provider.
//!
//! Works with DeepSeek's hosted API, MLX servers, llama.cpp, vLLM, LM Studio
//! and Ollama's `/v1` endpoint. Streaming SSE is parsed incrementally so text
//! appears as it is generated.

use std::time::Duration;

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use futures_util::{stream::BoxStream, StreamExt};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use rustyclippy_core::{canonical_string, Message, ProviderConfig, ProviderKind, Role};
use serde_json::{json, Value};

use crate::provider::{FinishReason, LlmRequest, Provider, StreamEvent};
use crate::usage::parse_usage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flavor {
    /// DeepSeek hosted API: automatic prefix caching, explicit cache usage fields.
    DeepSeek,
    /// Generic OpenAI-compatible server (MLX, llama.cpp, vLLM, ...).
    Generic,
}

pub struct OpenAiProvider {
    base_url: String,
    api_key: Option<String>,
    flavor: Flavor,
    client: reqwest::Client,
    timeout: Duration,
}

impl OpenAiProvider {
    pub fn new(config: &ProviderConfig) -> Result<Self> {
        let base_url = config.base_url.trim().trim_end_matches('/').to_string();
        if base_url.is_empty() {
            bail!("provider base_url is empty");
        }
        let flavor = match config.kind {
            ProviderKind::Deepseek => Flavor::DeepSeek,
            ProviderKind::OpenAiCompat => Flavor::Generic,
        };
        let api_key = config.api_key();
        if flavor == Flavor::DeepSeek && api_key.is_none() {
            bail!("DeepSeek API requires {} to be set", config.api_key_env);
        }
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .context("building HTTP client")?;
        Ok(Self {
            base_url,
            api_key,
            flavor,
            client,
            timeout: Duration::from_secs(config.timeout_secs),
        })
    }

    pub fn flavor(&self) -> Flavor {
        self.flavor
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    fn build_body(&self, request: &LlmRequest) -> Value {
        let mut messages = Vec::with_capacity(request.messages.len() + 1);
        messages.push(json!({"role": "system", "content": request.system}));
        for message in &request.messages {
            messages.push(to_openai_message(message));
        }

        let tools: Vec<Value> = request
            .tools
            .iter()
            .map(|schema| json!({"type": "function", "function": schema}))
            .collect();

        // Vision caveat: MLX servers mishandle images on the streaming path
        // when tools/system context are present (the model sees a blank or
        // stale image). Send image requests non-streaming and synthesize the
        // event stream from the full response instead.
        let has_images = request
            .messages
            .iter()
            .any(|message| !message.images.is_empty());

        let mut body = json!({
            "model": request.model,
            "messages": messages,
            "stream": !has_images,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        });
        if !has_images {
            body["stream_options"] = json!({"include_usage": true});
        }

        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
            if request.tool_choice_none {
                body["tool_choice"] = json!("none");
            }
        }

        // llama.cpp honours `cache_prompt`; DeepSeek caches automatically and
        // rejects unknown fields on some deployments.
        //
        // Vision caveat: MLX servers key their prefix cache on text tokens and
        // reuse image placeholder tokens across different images, which makes
        // the model see a stale/blank image. Disable prefix reuse whenever a
        // request carries attachments so the image is re-encoded.
        if self.flavor == Flavor::Generic {
            if has_images {
                body["cache_prompt"] = json!(false);
            } else if request.cache_prompt {
                body["cache_prompt"] = json!(true);
            }
        }

        // Reasoning effort mirrors the DSH harness knob. Local MLX servers
        // advertise `supportsReasoningEffort`; `off` omits the field entirely.
        if let Some(effort) = &request.reasoning_effort {
            if !effort.trim().is_empty() && !effort.eq_ignore_ascii_case("off") {
                body["reasoning_effort"] = json!(effort.trim());
            }
        }

        if let Ok(path) = std::env::var("RUSTYCLIPPY_DUMP_REQUEST") {
            if !path.trim().is_empty() {
                let _ = std::fs::write(
                    path,
                    serde_json::to_string_pretty(&body).unwrap_or_default(),
                );
            }
        }

        body
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    fn name(&self) -> &str {
        match self.flavor {
            Flavor::DeepSeek => "deepseek",
            Flavor::Generic => "openai-compat",
        }
    }

    async fn stream(&self, request: LlmRequest) -> Result<BoxStream<'static, Result<StreamEvent>>> {
        let has_images = request
            .messages
            .iter()
            .any(|message| !message.images.is_empty());
        let body = self.build_body(&request);
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(key) = &self.api_key {
            let value = HeaderValue::from_str(&format!("Bearer {key}"))
                .context("API key contains invalid header characters")?;
            headers.insert(AUTHORIZATION, value);
        }

        let response = self
            .client
            .post(self.endpoint())
            .headers(headers)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {}", self.endpoint()))?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            bail!(
                "provider returned HTTP {status}: {}",
                truncate(&text, 2_000)
            );
        }

        // Image requests are sent non-streaming (see `build_body`); turn the
        // complete response into the same event sequence the SSE path emits.
        if has_images {
            let text = response.text().await.unwrap_or_default();
            let value: Value = serde_json::from_str(&text).with_context(|| {
                format!(
                    "parsing non-streaming vision response: {}",
                    truncate(&text, 500)
                )
            })?;
            let stream = async_stream::try_stream! {
                if let Some(choices) = value.get("choices").and_then(|v| v.as_array()) {
                    for choice in choices {
                        let message = choice.get("message").unwrap_or(&Value::Null);
                        if let Some(reasoning) =
                            message.get("reasoning_content").and_then(|v| v.as_str())
                        {
                            if !reasoning.is_empty() {
                                yield StreamEvent::Reasoning(reasoning.to_string());
                            }
                        }
                        if let Some(content) = message.get("content").and_then(|v| v.as_str()) {
                            if !content.is_empty() {
                                yield StreamEvent::Text(content.to_string());
                            }
                        }
                        if let Some(calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
                            for (index, call) in calls.iter().enumerate() {
                                let id = call
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string());
                                let function = call.get("function");
                                let name = function
                                    .and_then(|f| f.get("name"))
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string());
                                let args = function
                                    .and_then(|f| f.get("arguments"))
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string());
                                yield StreamEvent::ToolCallDelta {
                                    index,
                                    id,
                                    name,
                                    args_fragment: args,
                                };
                            }
                        }
                    }
                }
                if let Some(usage_value) = value.get("usage") {
                    if usage_value.is_object() {
                        yield StreamEvent::Usage(parse_usage(usage_value));
                    }
                }
                if let Some(reason) = value
                    .get("choices")
                    .and_then(|v| v.as_array())
                    .and_then(|choices| choices.first())
                    .and_then(|choice| choice.get("finish_reason"))
                    .and_then(|v| v.as_str())
                {
                    yield StreamEvent::Finished(FinishReason::parse(reason));
                }
                yield StreamEvent::Done;
            };
            return Ok(Box::pin(stream));
        }

        let mut bytes = response.bytes_stream();
        let stream = async_stream::try_stream! {
            let mut buffer = SseBuffer::default();
            let mut finished = false;
            let mut finish: Option<FinishReason> = None;

            while let Some(chunk) = bytes.next().await {
                let chunk = chunk.context("reading SSE chunk")?;

                for event in buffer.push(&chunk) {
                    for line in event.lines() {
                        let line = line.trim();
                        let Some(data) = line.strip_prefix("data:") else { continue };
                        let data = data.trim();
                        if data.is_empty() {
                            continue;
                        }
                        if data == "[DONE]" {
                            finished = true;
                            break;
                        }
                        let value: Value =
                            serde_json::from_str(data).context("parsing SSE JSON")?;

                        if let Some(choices) = value.get("choices").and_then(|v| v.as_array()) {
                            for choice in choices {
                                // The server reports why it stopped on the final
                                // chunk; `length` means the output was cut off.
                                if let Some(reason) =
                                    choice.get("finish_reason").and_then(|v| v.as_str())
                                {
                                    finish = Some(FinishReason::parse(reason));
                                }
                                let delta = choice.get("delta").unwrap_or(&Value::Null);
                                if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
                                    if !content.is_empty() {
                                        yield StreamEvent::Text(content.to_string());
                                    }
                                }
                                if let Some(reasoning) =
                                    delta.get("reasoning_content").and_then(|v| v.as_str())
                                {
                                    if !reasoning.is_empty() {
                                        yield StreamEvent::Reasoning(reasoning.to_string());
                                    }
                                }
                                if let Some(calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                                    for call in calls {
                                        let index = call
                                            .get("index")
                                            .and_then(|v| v.as_u64())
                                            .unwrap_or(0) as usize;
                                        let id = call
                                            .get("id")
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.to_string());
                                        let function = call.get("function");
                                        let name = function
                                            .and_then(|f| f.get("name"))
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.to_string());
                                        let args = function
                                            .and_then(|f| f.get("arguments"))
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.to_string());
                                        yield StreamEvent::ToolCallDelta {
                                            index,
                                            id,
                                            name,
                                            args_fragment: args,
                                        };
                                    }
                                }
                            }
                        }

                        if let Some(usage_value) = value.get("usage") {
                            if usage_value.is_object() {
                                yield StreamEvent::Usage(parse_usage(usage_value));
                            }
                        }
                    }

                    if finished {
                        break;
                    }
                }

                if finished {
                    break;
                }
            }

            if let Some(reason) = finish {
                yield StreamEvent::Finished(reason);
            }
            yield StreamEvent::Done;
        };

        Ok(Box::pin(stream))
    }
}

/// Incremental SSE framing over raw bytes.
///
/// Chunks arrive at arbitrary byte boundaries, so a multi-byte UTF-8 character
/// can straddle two chunks. Decoding each chunk independently would corrupt it;
/// this buffer accumulates bytes and decodes only complete events.
#[derive(Default)]
struct SseBuffer {
    pending: Vec<u8>,
}

impl SseBuffer {
    /// Append a network chunk and return every complete event it completes.
    fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.pending.extend_from_slice(chunk);
        // Raw `\r` bytes only ever appear as SSE line terminators; JSON escapes
        // them as `\\r`, so stripping is safe and lets one `\n\n` separator
        // match both CRLF and LF framing.
        self.pending.retain(|byte| *byte != b'\r');

        let mut events = Vec::new();
        while let Some(position) = self.pending.windows(2).position(|pair| pair == b"\n\n") {
            let event = String::from_utf8_lossy(&self.pending[..position]).into_owned();
            self.pending.drain(..position + 2);
            events.push(event);
        }
        events
    }
}

fn to_openai_message(message: &Message) -> Value {
    match message.role {
        Role::Assistant => {
            let mut object = json!({"role": "assistant", "content": message.content});
            if !message.tool_calls.is_empty() {
                let calls: Vec<Value> = message
                    .tool_calls
                    .iter()
                    .map(|call| {
                        json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": canonical_string(&call.arguments),
                            }
                        })
                    })
                    .collect();
                object["tool_calls"] = Value::Array(calls);
            }
            object
        }
        Role::Tool => json!({
            "role": "tool",
            "tool_call_id": message.tool_call_id.clone().unwrap_or_default(),
            "content": message.content,
        }),
        Role::User => json!({"role": "user", "content": message.content_value()}),
        Role::System => json!({"role": "system", "content": message.content}),
    }
}

fn truncate(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut end = max;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &text[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finish_reason_parses_known_values() {
        assert_eq!(FinishReason::parse("stop"), FinishReason::Stop);
        assert_eq!(FinishReason::parse("length"), FinishReason::Length);
        assert_eq!(FinishReason::parse("tool_calls"), FinishReason::ToolCalls);
        assert_eq!(
            FinishReason::parse("content_filter"),
            FinishReason::ContentFilter
        );
        assert_eq!(FinishReason::parse("weird"), FinishReason::Other);
    }

    #[test]
    fn only_length_counts_as_truncation() {
        assert!(FinishReason::Length.is_truncated());
        assert!(!FinishReason::Stop.is_truncated());
        assert!(!FinishReason::ToolCalls.is_truncated());
        assert!(!FinishReason::ContentFilter.is_truncated());
        assert!(!FinishReason::Other.is_truncated());
    }

    #[test]
    fn sse_buffer_decodes_utf8_split_across_chunks() {
        // A multi-byte character straddling a chunk boundary must survive.
        let event = "data: {\"choices\":[{\"delta\":{\"content\":\"héllo 世界\"}}]}\n\n";
        let bytes = event.as_bytes();
        // Split inside the 2-byte `é`.
        let split = bytes.iter().position(|b| *b == 0xC3).unwrap() + 1;
        let mut buffer = SseBuffer::default();
        assert!(buffer.push(&bytes[..split]).is_empty());
        let events = buffer.push(&bytes[split..]);
        assert_eq!(events.len(), 1);
        assert!(events[0].contains("héllo 世界"), "got {:?}", events[0]);
        assert!(
            !events[0].contains('\u{FFFD}'),
            "replacement char leaked in"
        );
    }

    #[test]
    fn sse_buffer_handles_crlf_and_multiple_events() {
        let mut buffer = SseBuffer::default();
        let events = buffer.push(b"data: one\r\n\r\ndata: two\r\n\r\n");
        assert_eq!(
            events,
            vec!["data: one".to_string(), "data: two".to_string()]
        );
    }

    #[test]
    fn sse_buffer_keeps_partial_event_pending() {
        let mut buffer = SseBuffer::default();
        assert!(buffer.push(b"data: partial").is_empty());
        let events = buffer.push(b"\n\n");
        assert_eq!(events, vec!["data: partial".to_string()]);
    }

    #[test]
    fn sse_buffer_preserves_escaped_carriage_return() {
        // JSON escapes `\r` as a backslash-r pair, which must not be stripped.
        let mut buffer = SseBuffer::default();
        let events = buffer.push(b"data: {\"a\":\"x\\ry\"}\n\n");
        assert_eq!(events.len(), 1);
        assert!(
            events[0].contains("\\r"),
            "escaped \\r was stripped: {:?}",
            events[0]
        );
    }
    use rustyclippy_core::ToolCall;

    fn provider() -> OpenAiProvider {
        OpenAiProvider::new(&ProviderConfig::local_mlx(None, None)).unwrap()
    }

    #[test]
    fn body_includes_stable_system_and_tools() {
        let provider = provider();
        let request = LlmRequest {
            model: "m".into(),
            system: "sys".into(),
            tools: vec![
                json!({"name": "bash", "description": "run", "parameters": {"type": "object"}}),
            ],
            messages: vec![Message::user("hi")],
            max_tokens: 100,
            temperature: 0.0,
            cache_prompt: true,
            breakpoints: vec![],
            tool_choice_none: false,
            reasoning_effort: Some("high".to_string()),
        };
        let body = provider.build_body(&request);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "sys");
        assert_eq!(body["tools"][0]["function"]["name"], "bash");
        assert_eq!(body["cache_prompt"], true);
        assert_eq!(body["stream"], true);
        assert_eq!(body["reasoning_effort"], "high");
    }

    #[test]
    fn tool_calls_serialize_as_json_strings() {
        let message = Message::assistant_with_tools(
            "",
            vec![ToolCall::new("call_1", "bash", json!({"command": "ls"}))],
        );
        let wire = to_openai_message(&message);
        assert_eq!(
            wire["tool_calls"][0]["function"]["arguments"],
            r#"{"command":"ls"}"#
        );
        assert_eq!(wire["tool_calls"][0]["type"], "function");
    }

    #[test]
    fn tool_choice_none_is_set_when_requested() {
        let provider = provider();
        let request = LlmRequest {
            model: "m".into(),
            system: "s".into(),
            tools: vec![json!({"name": "bash"})],
            messages: vec![],
            max_tokens: 10,
            temperature: 0.0,
            cache_prompt: false,
            breakpoints: vec![],
            tool_choice_none: true,
            reasoning_effort: None,
        };
        assert_eq!(provider.build_body(&request)["tool_choice"], "none");
    }

    #[test]
    fn user_images_become_content_parts() {
        let message = Message::user_with_images(
            "what is this?",
            vec![rustyclippy_core::ImageAttachment::from_bytes(
                "image/png",
                b"abc",
            )],
        );
        let wire = to_openai_message(&message);
        assert_eq!(wire["content"][0]["type"], "text");
        let text = wire["content"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("what is this?"));
        assert!(text.chars().any(|ch| ch == '\u{200B}' || ch == '\u{200C}'));
        assert_eq!(wire["content"][1]["type"], "image_url");
        assert_eq!(
            wire["content"][1]["image_url"]["url"],
            "data:image/png;base64,YWJj"
        );
    }

    #[test]
    fn images_disable_prefix_cache() {
        let provider = provider();
        let request = LlmRequest {
            model: "m".into(),
            system: "s".into(),
            tools: vec![],
            messages: vec![Message::user_with_images(
                "look",
                vec![rustyclippy_core::ImageAttachment::from_bytes(
                    "image/png",
                    b"abc",
                )],
            )],
            max_tokens: 10,
            temperature: 0.0,
            cache_prompt: true,
            breakpoints: vec![],
            tool_choice_none: false,
            reasoning_effort: None,
        };
        let body = provider.build_body(&request);
        assert_eq!(body["cache_prompt"], false);
        assert_eq!(body["stream"], false);
        assert!(body.get("stream_options").is_none());
    }
}

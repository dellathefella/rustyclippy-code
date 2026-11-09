//! OpenAI-compatible provider.
//!
//! Works with DeepSeek's hosted API, MLX servers, llama.cpp, vLLM, LM Studio
//! and Ollama's `/v1` endpoint. Streaming SSE is parsed incrementally so text
//! appears as it is generated.

use std::time::Duration;

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use deeprust_core::{canonical_string, Message, ProviderConfig, ProviderKind, Role};
use futures_util::{stream::BoxStream, StreamExt};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Value};

use crate::provider::{LlmRequest, Provider, StreamEvent};
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

        let mut body = json!({
            "model": request.model,
            "messages": messages,
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        });

        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
            if request.tool_choice_none {
                body["tool_choice"] = json!("none");
            }
        }

        // llama.cpp honours `cache_prompt`; DeepSeek caches automatically and
        // rejects unknown fields on some deployments.
        if self.flavor == Flavor::Generic && request.cache_prompt {
            body["cache_prompt"] = json!(true);
        }

        // Reasoning effort mirrors the DSH harness knob. Local MLX servers
        // advertise `supportsReasoningEffort`; `off` omits the field entirely.
        if let Some(effort) = &request.reasoning_effort {
            if !effort.trim().is_empty() && !effort.eq_ignore_ascii_case("off") {
                body["reasoning_effort"] = json!(effort.trim());
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

        let mut bytes = response.bytes_stream();
        let stream = async_stream::try_stream! {
            let mut buffer = String::new();
            let mut finished = false;

            while let Some(chunk) = bytes.next().await {
                let chunk = chunk.context("reading SSE chunk")?;
                let text = String::from_utf8_lossy(&chunk).replace("\r\n", "\n");
                buffer.push_str(&text);

                while let Some(position) = buffer.find("\n\n") {
                    let event = buffer[..position].to_string();
                    buffer.drain(..position + 2);

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

            yield StreamEvent::Done;
        };

        Ok(Box::pin(stream))
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
        Role::User => json!({"role": "user", "content": message.content}),
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
    use deeprust_core::ToolCall;

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
}

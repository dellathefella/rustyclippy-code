//! Conversation message model.
//!
//! Messages are immutable once created. The transcript only ever appends new
//! messages, which keeps the serialized prompt prefix stable across turns.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::canonical::hash_value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// A tool invocation requested by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Always a JSON object; providers stream it as a string fragment.
    pub arguments: Value,
}

impl ToolCall {
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }
}

/// One conversation message.
///
/// `reasoning` is deliberately *not* part of the wire form: DeepSeek models
/// emit `reasoning_content` during a turn, but replaying it into later requests
/// both bloats the context and (for reasoner models) is rejected. Keeping it
/// out of the canonical form means it never perturbs the cached prefix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Captured for display/logging only; excluded from the wire form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self::text(Role::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::text(Role::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::text(Role::Assistant, content)
    }

    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
            reasoning: None,
        }
    }

    pub fn assistant_with_tools(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls,
            tool_call_id: None,
            reasoning: None,
        }
    }

    fn text(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            reasoning: None,
        }
    }

    /// Canonical wire form: exactly what gets sent to the model.
    ///
    /// `reasoning` is omitted on purpose (see type docs).
    pub fn wire_json(&self) -> Value {
        let mut object = json!({
            "role": self.role.as_str(),
            "content": self.content,
        });
        if !self.tool_calls.is_empty() {
            let calls: Vec<Value> = self
                .tool_calls
                .iter()
                .map(|call| {
                    json!({
                        "id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                    })
                })
                .collect();
            object["tool_calls"] = Value::Array(calls);
        }
        if let Some(id) = &self.tool_call_id {
            object["tool_call_id"] = Value::String(id.clone());
        }
        object
    }

    /// Stable hash of the wire form. Used by the cache engine to detect the
    /// longest unchanged prefix.
    pub fn wire_hash(&self) -> String {
        hash_value(&self.wire_json())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reasoning_is_not_wire_visible() {
        let mut message = Message::assistant("hi");
        message.reasoning = Some("secret thoughts".to_string());
        assert_eq!(
            message.wire_json(),
            json!({"role": "assistant", "content": "hi"})
        );
    }

    #[test]
    fn tool_call_wire_shape() {
        let message = Message::assistant_with_tools(
            "",
            vec![ToolCall::new("call_1", "bash", json!({"command": "ls"}))],
        );
        let wire = message.wire_json();
        assert_eq!(wire["tool_calls"][0]["name"], "bash");
        assert_eq!(wire["tool_calls"][0]["arguments"]["command"], "ls");
    }

    #[test]
    fn wire_hash_ignores_reasoning() {
        let mut a = Message::assistant("same");
        let b = Message::assistant("same");
        a.reasoning = Some("different".into());
        assert_eq!(a.wire_hash(), b.wire_hash());
    }
}

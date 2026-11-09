//! Conversation message model.
//!
//! Messages are immutable once created. The transcript only ever appends new
//! messages, which keeps the serialized prompt prefix stable across turns.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::canonical::{hash_str, hash_value};
use crate::image::ImageAttachment;

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
    /// Vision attachments. Serialized into the wire form as content parts.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ImageAttachment>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self::text(Role::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::text(Role::User, content)
    }

    /// A user message carrying text plus vision attachments.
    pub fn user_with_images(content: impl Into<String>, images: Vec<ImageAttachment>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            reasoning: None,
            images,
        }
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
            images: Vec::new(),
        }
    }

    pub fn assistant_with_tools(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls,
            tool_call_id: None,
            reasoning: None,
            images: Vec::new(),
        }
    }

    fn text(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            reasoning: None,
            images: Vec::new(),
        }
    }

    /// Stable fingerprint of the attached images.
    ///
    /// Some servers cache image placeholder tokens and reuse them across
    /// different images, which makes the model see stale pixels. We encode the
    /// fingerprint as invisible zero-width characters in the text part ahead of
    /// the image, so the text prefix differs per image without the model seeing
    /// a placeholder marker.
    pub fn image_fingerprint(&self) -> Option<String> {
        if self.images.is_empty() {
            return None;
        }
        let mut input = String::new();
        for image in &self.images {
            input.push_str(&image.mime);
            input.push(':');
            input.push_str(&image.data);
            input.push('\n');
        }
        Some(hash_str(&input).chars().take(16).collect())
    }

    /// Wire content: a plain string, or a multimodal parts array when images
    /// are attached (OpenAI-compatible vision shape).
    pub fn content_value(&self) -> Value {
        if self.images.is_empty() {
            return Value::String(self.content.clone());
        }
        let fingerprint = self.image_fingerprint().unwrap_or_default();
        let text = format!("{}{}", self.content, invisible_marker(&fingerprint));
        let mut parts = Vec::with_capacity(self.images.len() + 1);
        parts.push(json!({"type": "text", "text": text}));
        for image in &self.images {
            parts.push(json!({
                "type": "image_url",
                "image_url": {"url": image.data_url()},
            }));
        }
        Value::Array(parts)
    }

    /// Canonical wire form: exactly what gets sent to the model.
    ///
    /// `reasoning` is omitted on purpose (see type docs).
    pub fn wire_json(&self) -> Value {
        let mut object = json!({
            "role": self.role.as_str(),
            "content": self.content_value(),
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

/// Encode a hex fingerprint as invisible zero-width characters.
///
/// Four invisible symbols represent two bits each, so a 16-hex-digit
/// fingerprint becomes a 32-character sequence that tokenizes differently for
/// every image while staying invisible to the model.
fn invisible_marker(fingerprint: &str) -> String {
    const SYMBOLS: [char; 4] = ['\u{200B}', '\u{200C}', '\u{200D}', '\u{2060}'];
    let mut out = String::with_capacity(fingerprint.len() * 2);
    for ch in fingerprint.chars() {
        let value = ch.to_digit(16).unwrap_or(0) as usize;
        out.push(SYMBOLS[(value >> 2) & 3]);
        out.push(SYMBOLS[value & 3]);
    }
    out
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

    #[test]
    fn images_serialize_as_content_parts() {
        let message = Message::user_with_images(
            "what is this?",
            vec![ImageAttachment::from_bytes("image/png", b"abc")],
        );
        let wire = message.wire_json();
        assert_eq!(wire["role"], "user");
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
    fn image_only_message_has_a_marker_text_part() {
        let message =
            Message::user_with_images("", vec![ImageAttachment::from_bytes("image/png", b"abc")]);
        let wire = message.wire_json();
        assert_eq!(wire["content"].as_array().unwrap().len(), 2);
        assert_eq!(wire["content"][0]["type"], "text");
        assert_eq!(wire["content"][1]["type"], "image_url");
    }

    #[test]
    fn image_fingerprint_is_stable_and_image_specific() {
        let a =
            Message::user_with_images("x", vec![ImageAttachment::from_bytes("image/png", b"abc")]);
        let b =
            Message::user_with_images("x", vec![ImageAttachment::from_bytes("image/png", b"abc")]);
        let c =
            Message::user_with_images("x", vec![ImageAttachment::from_bytes("image/png", b"abd")]);
        assert_eq!(a.image_fingerprint(), b.image_fingerprint());
        assert_ne!(a.image_fingerprint(), c.image_fingerprint());
        assert_eq!(Message::user("x").image_fingerprint(), None);
    }

    #[test]
    fn wire_hash_changes_with_image() {
        let plain = Message::user("same");
        let with_image = Message::user_with_images(
            "same",
            vec![ImageAttachment::from_bytes("image/png", b"abc")],
        );
        assert_ne!(plain.wire_hash(), with_image.wire_hash());
    }
}

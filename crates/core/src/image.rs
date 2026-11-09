//! Image attachments for vision-capable providers.
//!
//! Images travel as base64 data URLs inside a user message's content array
//! (the OpenAI-compatible multimodal shape). Keeping the encoded bytes in the
//! message means the wire form — and therefore the cached prefix hash — stays
//! byte-stable across turns.

use serde::{Deserialize, Serialize};

/// A base64-encoded image attached to a user message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageAttachment {
    /// MIME type, e.g. `image/png` or `image/jpeg`.
    pub mime: String,
    /// Base64-encoded image bytes (no data URL prefix).
    pub data: String,
}

impl ImageAttachment {
    pub fn from_bytes(mime: impl Into<String>, bytes: &[u8]) -> Self {
        Self {
            mime: mime.into(),
            data: base64_encode(bytes),
        }
    }

    /// The `data:` URL the provider expects.
    pub fn data_url(&self) -> String {
        format!("data:{};base64,{}", self.mime, self.data)
    }

    /// Approximate decoded byte length (for display).
    pub fn approx_bytes(&self) -> usize {
        self.data.len() * 3 / 4
    }
}

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 encoding (RFC 4648, with padding).
pub fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(triple >> 18) as usize & 63] as char);
        out.push(ALPHABET[(triple >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[(triple >> 6) as usize & 63] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[triple as usize & 63] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// Standard base64 decoding. Returns `None` on invalid input.
pub fn base64_decode(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let mut buffer: u32 = 0;
    let mut bits = 0u32;
    for byte in text.bytes() {
        if byte == b'=' {
            break;
        }
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        };
        buffer = (buffer << 6) | value as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buffer >> bits) as u8);
        }
    }
    Some(out)
}

/// Sniff an image MIME type from magic bytes.
pub fn detect_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        Some("image/png")
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_roundtrips() {
        for input in [
            b"".as_slice(),
            b"a".as_slice(),
            b"ab".as_slice(),
            b"abc".as_slice(),
            b"hello world".as_slice(),
            &[0u8, 255, 128, 1, 2, 3],
        ] {
            let encoded = base64_encode(input);
            assert_eq!(base64_decode(&encoded).unwrap(), input);
        }
        assert_eq!(base64_encode(b"abc"), "YWJj");
        assert_eq!(base64_encode(b"ab"), "YWI=");
        assert_eq!(base64_encode(b"a"), "YQ==");
    }

    #[test]
    fn mime_sniffing() {
        assert_eq!(
            detect_mime(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]),
            Some("image/png")
        );
        assert_eq!(detect_mime(&[0xFF, 0xD8, 0xFF, 0xE0]), Some("image/jpeg"));
        assert_eq!(detect_mime(b"GIF89a...."), Some("image/gif"));
        assert_eq!(detect_mime(b"RIFFxxxxWEBP"), Some("image/webp"));
        assert_eq!(detect_mime(b"not an image"), None);
    }

    #[test]
    fn attachment_data_url() {
        let image = ImageAttachment::from_bytes("image/png", b"abc");
        assert_eq!(image.data_url(), "data:image/png;base64,YWJj");
    }
}

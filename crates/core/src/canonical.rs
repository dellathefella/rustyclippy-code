//! Canonical, byte-stable serialization primitives.
//!
//! Every byte that goes into a prompt is produced through these helpers so that
//! two logically identical requests serialize to *exactly* the same bytes. This
//! is the foundation of prefix-cache reuse: a local inference server can only
//! reuse its KV cache when the prompt prefix is byte-for-byte identical.

use serde_json::{Map, Value};

/// Recursively sort object keys so serialization order is deterministic.
///
/// `serde_json`'s default `Map` is a `BTreeMap`, but we canonicalize explicitly
/// so the guarantee holds even if the `preserve_order` feature is enabled by a
/// downstream crate.
pub fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = Map::new();
            for key in keys {
                out.insert(key.clone(), canonicalize(&map[key]));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonicalize).collect()),
        other => other.clone(),
    }
}

/// Canonical JSON bytes (compact, sorted keys).
pub fn canonical_bytes(value: &Value) -> Vec<u8> {
    serde_json::to_vec(&canonicalize(value)).expect("canonical JSON serialization cannot fail")
}

/// Canonical JSON string.
pub fn canonical_string(value: &Value) -> String {
    serde_json::to_string(&canonicalize(value)).expect("canonical JSON serialization cannot fail")
}

/// BLAKE3 hex digest of arbitrary bytes.
pub fn hash_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// Hash a canonical JSON value.
pub fn hash_value(value: &Value) -> String {
    hash_bytes(&canonical_bytes(value))
}

/// Hash a UTF-8 string.
pub fn hash_str(value: &str) -> String {
    hash_bytes(value.as_bytes())
}

/// Combine two hashes into a stable composite hash.
pub fn hash_pair(left: &str, right: &str) -> String {
    let mut input = Vec::with_capacity(left.len() + right.len() + 1);
    input.extend_from_slice(left.as_bytes());
    input.push(b'\n');
    input.extend_from_slice(right.as_bytes());
    hash_bytes(&input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn key_order_is_normalized() {
        let a = json!({"b": 1, "a": {"z": 2, "y": 3}});
        let b = json!({"a": {"y": 3, "z": 2}, "b": 1});
        assert_eq!(canonical_string(&a), canonical_string(&b));
        assert_eq!(canonical_string(&a), r#"{"a":{"y":3,"z":2},"b":1}"#);
    }

    #[test]
    fn arrays_keep_order() {
        let a = json!([1, 2, 3]);
        let b = json!([3, 2, 1]);
        assert_ne!(canonical_string(&a), canonical_string(&b));
    }

    #[test]
    fn hashes_are_stable() {
        let value = json!({"x": "hello", "y": [1, 2]});
        assert_eq!(hash_value(&value), hash_value(&value));
        assert_eq!(hash_str("abc"), hash_str("abc"));
        assert_ne!(hash_str("abc"), hash_str("abd"));
    }
}

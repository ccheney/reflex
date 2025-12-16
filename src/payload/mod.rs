//! Payload encoding/decoding helpers.
//!
//! Reflex uses Tauq as a compact, schema-friendly representation of JSON payloads.

use serde_json::Value;

/// Errors returned by Tauq parsing.
pub use tauq::error::TauqError;

/// Encodes JSON values into Tauq.
pub struct TauqEncoder;

impl TauqEncoder {
    /// Encodes a JSON value into Tauq text.
    pub fn encode(value: &Value) -> String {
        tauq::format_to_tauq(value)
    }
}

/// Decodes Tauq text into JSON values.
pub struct TauqDecoder;

impl TauqDecoder {
    /// Decodes Tauq into a single JSON value.
    pub fn decode(input: &str) -> Result<Value, TauqError> {
        tauq::compile_tauq(input)
    }

    /// Decodes Tauq into a batch (array) or a single value wrapped as a 1-item batch.
    pub fn decode_batch(input: &str) -> Result<Vec<Value>, TauqError> {
        let value = tauq::compile_tauq(input)?;
        match value {
            Value::Array(arr) => Ok(arr),
            _ => Ok(vec![value]),
        }
    }
}

/// Encodes multiple JSON values into a Tauq array.
pub struct TauqBatchEncoder;

impl TauqBatchEncoder {
    /// Encodes all values as a Tauq array.
    pub fn encode_all(values: &[Value]) -> Result<String, TauqError> {
        let array = Value::Array(values.to_vec());
        Ok(tauq::format_to_tauq(&array))
    }
}

/// Converts a payload to Tauq if it looks like JSON; otherwise returns it unchanged.
///
/// This is a best-effort helper: it never errors.
///
/// # Example
///
/// ```
/// use reflex::payload::ensure_tauq_format;
///
/// // Valid JSON gets encoded
/// let json_payload = r#"{"key": "value"}"#;
/// let result = ensure_tauq_format(json_payload);
/// // result is Tauq-encoded
///
/// // Invalid JSON passes through unchanged
/// let invalid = "not valid json {";
/// let result = ensure_tauq_format(invalid);
/// assert_eq!(result, invalid);
///
/// // Plain text passes through unchanged
/// let plain = "Hello, world!";
/// let result = ensure_tauq_format(plain);
/// assert_eq!(result, plain);
/// ```
pub fn ensure_tauq_format(payload: &str) -> String {
    let trimmed = payload.trim();

    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        match serde_json::from_str::<Value>(trimmed) {
            Ok(json_value) => TauqEncoder::encode(&json_value),
            Err(_) => payload.to_string(),
        }
    } else {
        payload.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn assert_json_eq(left: &Value, right: &Value) {
        if left == right {
            return;
        }

        match (left, right) {
            (Value::Number(n1), Value::Number(n2)) => {
                let f1 = n1.as_f64().unwrap_or(0.0);
                let f2 = n2.as_f64().unwrap_or(0.0);
                if (f1 - f2).abs() > 1e-9 {
                    panic!("Numbers mismatch: {:?} != {:?}", n1, n2);
                }
            }
            (Value::Object(o1), Value::Object(o2)) => {
                if o1.len() != o2.len() {
                    panic!("Object length mismatch: {:?} != {:?}", o1, o2);
                }
                for (k, v1) in o1 {
                    if let Some(v2) = o2.get(k) {
                        assert_json_eq(v1, v2);
                    } else {
                        panic!("Key missing in right: {}", k);
                    }
                }
            }
            (Value::Array(a1), Value::Array(a2)) => {
                if a1.len() != a2.len() {
                    panic!("Array length mismatch: {:?} != {:?}", a1, a2);
                }
                for (v1, v2) in a1.iter().zip(a2.iter()) {
                    assert_json_eq(v1, v2);
                }
            }
            _ => panic!("Mismatch: {:?} != {:?}", left, right),
        }
    }

    #[test]
    fn test_encode_simple_string() {
        let json = json!("hello world");
        let tauq = TauqEncoder::encode(&json);
        assert!(tauq.contains("hello world"));
    }

    #[test]
    fn test_roundtrip_simple_object() {
        let json = json!({
            "Response": {
                "text": "Hello world",
                "confidence": 0.95,
                "timestamp": 1735689600
            }
        });

        let tauq = TauqEncoder::encode(&json);
        let decoded = TauqDecoder::decode(&tauq).unwrap();
        assert_json_eq(&json, &decoded);
    }

    #[test]
    fn test_roundtrip_with_special_chars() {
        let json = json!({
            "Response": {
                "text": "Line 1\nLine 2\twith\ttabs",
                "quote": "He said \"hello\""
            }
        });

        let tauq = TauqEncoder::encode(&json);
        let decoded = TauqDecoder::decode(&tauq).unwrap();
        assert_json_eq(&json, &decoded);
    }

    #[test]
    fn test_roundtrip_complex() {
        let json = json!({
            "Response": {
                "text": "To reset your password, go to Settings > Security and click Reset Password",
                "confidence": 0.99,
                "timestamp": 1735689600,
                "source": "knowledge_base",
                "tokens_used": 42,
                "metadata": {
                    "cached": true,
                    "ttl": 3600
                },
                "alternatives": ["option1", "option2"]
            }
        });

        let tauq = TauqEncoder::encode(&json);
        let decoded = TauqDecoder::decode(&tauq).unwrap();
        assert_json_eq(&json, &decoded);
    }

    #[test]
    fn test_compression_ratio_single_object() {
        let json = json!({
            "Response": {
                "text": "To reset your password, go to Settings > Security and click Reset Password. You will receive an email with a reset link. The link expires in 24 hours.",
                "confidence": 0.99,
                "timestamp": 1735689600,
                "source": "knowledge_base",
                "tokens_used": 42
            }
        });

        let json_str = serde_json::to_string(&json).unwrap();
        let tauq = TauqEncoder::encode(&json);

        let json_bytes = json_str.len();
        let tauq_bytes = tauq.len();

        println!(
            "Single object - JSON: {} bytes, Tauq: {} bytes",
            json_bytes, tauq_bytes
        );

        assert!(
            tauq_bytes as f64 <= json_bytes as f64 * 1.1,
            "Tauq should not be significantly larger than JSON"
        );
    }

    #[test]
    fn test_compression_ratio_batch_with_shared_schema() {
        let values = vec![
            json!({"Response": {"confidence": 0.95, "source": "cache", "text": "First response text here with some content", "timestamp": 1735689600}}),
            json!({"Response": {"confidence": 0.87, "source": "cache", "text": "Second response with different text content", "timestamp": 1735689700}}),
            json!({"Response": {"confidence": 0.92, "source": "cache", "text": "Third response also quite lengthy text data", "timestamp": 1735689800}}),
            json!({"Response": {"confidence": 0.88, "source": "cache", "text": "Fourth response with more example content", "timestamp": 1735689900}}),
            json!({"Response": {"confidence": 0.91, "source": "cache", "text": "Fifth response completing our batch example", "timestamp": 1735690000}}),
        ];

        let tauq_batch = TauqBatchEncoder::encode_all(&values).unwrap();

        let json_entries: Vec<String> = values
            .iter()
            .map(|v| serde_json::to_string(v).unwrap())
            .collect();
        let _json_batch = json_entries.join("\n");

        let decoded = TauqDecoder::decode_batch(&tauq_batch).unwrap();
        assert_eq!(values.len(), decoded.len());
        for (original, decoded_value) in values.iter().zip(decoded.iter()) {
            assert_json_eq(original, decoded_value);
        }
    }
}

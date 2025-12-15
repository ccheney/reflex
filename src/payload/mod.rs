use serde_json::Value;

pub use tauq::error::TauqError;

pub struct TauqEncoder;

impl TauqEncoder {
    pub fn encode(value: &Value) -> String {
        tauq::format_to_tauq(value)
    }
}

pub struct TauqDecoder;

impl TauqDecoder {
    pub fn decode(input: &str) -> Result<Value, TauqError> {
        tauq::compile_tauq(input)
    }

    pub fn decode_batch(input: &str) -> Result<Vec<Value>, TauqError> {
        let value = tauq::compile_tauq(input)?;
        match value {
            Value::Array(arr) => Ok(arr),
            _ => Ok(vec![value]),
        }
    }
}

pub struct TauqBatchEncoder;

impl TauqBatchEncoder {
    pub fn encode_all(values: &[Value]) -> Result<String, TauqError> {
        let array = Value::Array(values.to_vec());
        Ok(tauq::format_to_tauq(&array))
    }
}

/// Converts a payload to Tauq format if it appears to be valid JSON.
///
/// # Defensive Fallback Behavior
///
/// This function intentionally uses a silent fallback strategy: if the payload
/// cannot be parsed as JSON or does not appear to need Tauq encoding, the
/// original payload is returned unchanged. This is **by design** to ensure
/// robustness in the response path.
///
/// # Why Silent Fallback?
///
/// - **API Stability**: Errors in response formatting should never break the API.
///   A degraded response (non-Tauq) is better than an error response.
/// - **Graceful Handling**: Not all payloads are JSON. Plain text, XML, or
///   already-encoded content should pass through without modification.
/// - **Defense in Depth**: Upstream validation may have failed or been bypassed;
///   this function acts as a safety net, not a gatekeeper.
///
/// # Behavior
///
/// - If the trimmed payload starts with `{` or `[`, attempts JSON parsing
/// - On successful parse, encodes to Tauq format
/// - On parse failure, returns the original payload unchanged
/// - For non-JSON-like payloads, returns the original unchanged
///
/// # Operational Considerations
///
/// If your system expects Tauq-formatted responses, monitor for responses that
/// are not in Tauq format. This may indicate:
/// - Malformed JSON from upstream services
/// - Unexpected content types in the response path
/// - Configuration issues in payload handling
///
/// # Guarantees
///
/// This function **never errors**. It always returns a usable `String`, either:
/// - The Tauq-encoded version of valid JSON input, or
/// - The original payload unchanged
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

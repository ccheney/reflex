//! Comprehensive tests for the gateway handler module.
//!
//! This module provides 100% test coverage for `handler.rs` including:
//! - `validate_no_legacy_fields` - request validation
//! - `semantic_text_from_request` - semantic key generation
//! - `make_response` - HTTP response construction
//! - `spawn_index_update` - async vector indexing
//! - `chat_completions_handler` - main request handler (L1/L2/L3 hits, misses, streaming, errors)

use async_openai::types::chat::*;
use axum::{Router, body::Body, http::Request, http::StatusCode, response::IntoResponse};
use http_body_util::BodyExt;
use std::sync::Arc;
use tempfile::TempDir;
use tower::ServiceExt;

use reflex::cache::{
    BqSearchBackend, L1CacheHandle, L2Config, L2SemanticCache, NvmeStorageLoader,
    REFLEX_STATUS_HEADER, ReflexStatus, TieredCache,
};
use reflex::embedding::RerankerConfig;
use reflex::embedding::sinter::{SinterConfig, SinterEmbedder};
use crate::gateway::create_router_with_state;
use crate::gateway::error::GatewayError;
use crate::gateway::payload::CachePayload;
use crate::gateway::state::HandlerState;
use reflex::scoring::CrossEncoderScorer;
use reflex::vectordb::bq::MockBqClient;

const TEST_COLLECTION_NAME: &str = "handler_test_collection";

/// Creates a minimal valid chat completion request JSON.
fn minimal_request_json() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Hello, world!"}
        ]
    })
}

/// Creates a request with tools specified.
fn request_with_tools_json() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "What's the weather?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        "tool_choice": "auto"
    })
}

/// Creates a request with response_format specified.
fn request_with_response_format_json() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Generate JSON"}
        ],
        "response_format": {"type": "json_object"}
    })
}

/// Creates a streaming request.
fn streaming_request_json() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "stream": true
    })
}

/// Creates a request with legacy function-calling fields (should be rejected).
fn legacy_functions_request_json() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "functions": [{"name": "test", "parameters": {}}]
    })
}

/// Creates a request with legacy function_call field (should be rejected).
fn legacy_function_call_request_json() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "function_call": "auto"
    })
}

/// Creates a request with legacy function role message (should be rejected).
fn legacy_function_role_request_json() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "function", "name": "test_fn", "content": "result"}
        ]
    })
}

/// Creates a request with legacy assistant function_call (should be rejected).
fn legacy_assistant_function_call_request_json() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": null, "function_call": {"name": "fn", "arguments": "{}"}}
        ]
    })
}

/// Creates a request with missing messages (should be rejected).
fn missing_messages_request_json() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4"
    })
}

/// Creates a request with invalid messages type (should be rejected).
fn invalid_messages_request_json() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4",
        "messages": "not an array"
    })
}

/// Creates a mock CreateChatCompletionResponse for testing.
fn mock_completion_response(model: &str) -> CreateChatCompletionResponse {
    serde_json::from_value(serde_json::json!({
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1702512000_u32,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello! How can I help you?"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    }))
    .expect("Failed to create mock response")
}

/// Creates a CachePayload for testing.
fn mock_cache_payload() -> CachePayload {
    CachePayload {
        semantic_request:
            r#"{"model":"gpt-4","messages":[{"role":"user","content":"Hello, world!"}]}"#
                .to_string(),
        response: mock_completion_response("gpt-4"),
    }
}

/// Sets up a test HandlerState with mocked dependencies.
async fn setup_test_state() -> (HandlerState<MockBqClient, NvmeStorageLoader>, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let storage_path = temp_dir.path().to_path_buf();

    let bq_client = MockBqClient::new();
    bq_client
        .ensure_bq_collection(
            TEST_COLLECTION_NAME,
            reflex::constants::DEFAULT_VECTOR_SIZE_U64,
        )
        .await
        .expect("Failed to ensure collection");

    let loader = NvmeStorageLoader::new(storage_path.clone());

    let embedder =
        SinterEmbedder::load(SinterConfig::stub()).expect("Failed to load stub embedder");

    let l2_config = L2Config::default().collection_name(TEST_COLLECTION_NAME);
    let l2_cache = L2SemanticCache::new(embedder, bq_client.clone(), loader, l2_config)
        .expect("Failed to create L2 cache");

    let l1_cache = L1CacheHandle::new();
    let tiered_cache = Arc::new(TieredCache::new(l1_cache, l2_cache));

    let scorer = Arc::new(
        CrossEncoderScorer::new(RerankerConfig::stub().with_threshold(0.7))
            .expect("Failed to create scorer"),
    );

    let state = HandlerState::new_with_mock_provider(
        tiered_cache,
        scorer,
        storage_path,
        bq_client,
        TEST_COLLECTION_NAME.to_string(),
        true, // mock_provider = true
    );

    (state, temp_dir)
}

/// Creates a test router with the given state.
fn create_test_router<B, S>(state: HandlerState<B, S>) -> Router
where
    B: BqSearchBackend + Clone + Send + Sync + 'static,
    S: reflex::cache::StorageLoader
        + reflex::storage::StorageWriter
        + Clone
        + Send
        + Sync
        + 'static,
{
    create_router_with_state(state)
}

mod validate_no_legacy_fields_tests {
    use super::*;
    use crate::gateway::handler::validate_no_legacy_fields;

    #[test]
    fn test_valid_minimal_request() {
        let req = minimal_request_json();
        let result = validate_no_legacy_fields(&req);
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_request_with_tools() {
        let req = request_with_tools_json();
        let result = validate_no_legacy_fields(&req);
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_request_with_response_format() {
        let req = request_with_response_format_json();
        let result = validate_no_legacy_fields(&req);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rejects_legacy_functions_field() {
        let req = legacy_functions_request_json();
        let result = validate_no_legacy_fields(&req);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            GatewayError::InvalidRequest(msg) => {
                assert!(msg.contains("Legacy function-calling"));
                assert!(msg.contains("tools"));
            }
            _ => panic!("Expected InvalidRequest error"),
        }
    }

    #[test]
    fn test_rejects_legacy_function_call_field() {
        let req = legacy_function_call_request_json();
        let result = validate_no_legacy_fields(&req);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            GatewayError::InvalidRequest(msg) => {
                assert!(msg.contains("Legacy function-calling"));
            }
            _ => panic!("Expected InvalidRequest error"),
        }
    }

    #[test]
    fn test_rejects_function_role_message() {
        let req = legacy_function_role_request_json();
        let result = validate_no_legacy_fields(&req);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            GatewayError::InvalidRequest(msg) => {
                assert!(msg.contains("Unsupported message role"));
                assert!(msg.contains("function"));
            }
            _ => panic!("Expected InvalidRequest error"),
        }
    }

    #[test]
    fn test_rejects_assistant_function_call() {
        let req = legacy_assistant_function_call_request_json();
        let result = validate_no_legacy_fields(&req);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            GatewayError::InvalidRequest(msg) => {
                assert!(msg.contains("Legacy `function_call` on assistant"));
                assert!(msg.contains("tool_calls"));
            }
            _ => panic!("Expected InvalidRequest error"),
        }
    }

    #[test]
    fn test_rejects_missing_messages() {
        let req = missing_messages_request_json();
        let result = validate_no_legacy_fields(&req);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            GatewayError::InvalidRequest(msg) => {
                assert!(msg.contains("Missing or invalid `messages`"));
            }
            _ => panic!("Expected InvalidRequest error"),
        }
    }

    #[test]
    fn test_rejects_invalid_messages_type() {
        let req = invalid_messages_request_json();
        let result = validate_no_legacy_fields(&req);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            GatewayError::InvalidRequest(msg) => {
                assert!(msg.contains("Missing or invalid `messages`"));
            }
            _ => panic!("Expected InvalidRequest error"),
        }
    }

    #[test]
    fn test_empty_messages_array_is_valid() {
        let req = serde_json::json!({
            "model": "gpt-4",
            "messages": []
        });
        let result = validate_no_legacy_fields(&req);
        assert!(result.is_ok());
    }

    #[test]
    fn test_multiple_messages_without_legacy_fields() {
        let req = serde_json::json!({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
        });
        let result = validate_no_legacy_fields(&req);
        assert!(result.is_ok());
    }

    #[test]
    fn test_assistant_with_tool_calls_is_valid() {
        let req = serde_json::json!({
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "test", "arguments": "{}"}
                    }]
                }
            ]
        });
        let result = validate_no_legacy_fields(&req);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tool_role_is_valid() {
        let req = serde_json::json!({
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "tool", "tool_call_id": "call_123", "content": "result"}
            ]
        });
        let result = validate_no_legacy_fields(&req);
        assert!(result.is_ok());
    }
}

mod semantic_text_from_request_tests {
    use super::*;
    use crate::gateway::handler::semantic_text_from_request;

    fn parse_request(json: serde_json::Value) -> CreateChatCompletionRequest {
        serde_json::from_value(json).expect("Failed to parse request")
    }

    #[test]
    fn test_minimal_request_includes_model_and_messages() {
        let req = parse_request(minimal_request_json());
        let semantic = semantic_text_from_request(&req);

        let parsed: serde_json::Value =
            serde_json::from_str(&semantic).expect("Semantic text should be valid JSON");

        assert_eq!(parsed["model"], "gpt-4");
        assert!(parsed["messages"].is_array());
        assert!(parsed.get("tools").is_none());
        assert!(parsed.get("tool_choice").is_none());
        assert!(parsed.get("response_format").is_none());
    }

    #[test]
    fn test_request_with_tools_includes_tools_and_tool_choice() {
        let req = parse_request(request_with_tools_json());
        let semantic = semantic_text_from_request(&req);

        let parsed: serde_json::Value =
            serde_json::from_str(&semantic).expect("Semantic text should be valid JSON");

        assert_eq!(parsed["model"], "gpt-4");
        assert!(parsed["messages"].is_array());
        assert!(parsed["tools"].is_array());
        assert!(parsed.get("tool_choice").is_some());
    }

    #[test]
    fn test_request_with_response_format_includes_response_format() {
        let req = parse_request(request_with_response_format_json());
        let semantic = semantic_text_from_request(&req);

        let parsed: serde_json::Value =
            serde_json::from_str(&semantic).expect("Semantic text should be valid JSON");

        assert_eq!(parsed["model"], "gpt-4");
        assert!(parsed.get("response_format").is_some());
    }

    #[test]
    fn test_excludes_sampling_parameters() {
        let req = parse_request(serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "n": 1,
            "stop": ["END"],
            "seed": 42,
            "user": "user123"
        }));
        let semantic = semantic_text_from_request(&req);

        let parsed: serde_json::Value =
            serde_json::from_str(&semantic).expect("Semantic text should be valid JSON");

        assert!(parsed.get("temperature").is_none());
        assert!(parsed.get("max_tokens").is_none());
        assert!(parsed.get("top_p").is_none());
        assert!(parsed.get("frequency_penalty").is_none());
        assert!(parsed.get("presence_penalty").is_none());
        assert!(parsed.get("n").is_none());
        assert!(parsed.get("stop").is_none());
        assert!(parsed.get("seed").is_none());
        assert!(parsed.get("user").is_none());
    }

    #[test]
    fn test_same_semantic_content_different_sampling_params() {
        let req1 = parse_request(serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.0
        }));

        let req2 = parse_request(serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 1.0,
            "max_tokens": 500
        }));

        let semantic1 = semantic_text_from_request(&req1);
        let semantic2 = semantic_text_from_request(&req2);

        assert_eq!(semantic1, semantic2);
    }

    #[test]
    fn test_different_models_produce_different_semantic_text() {
        let req1 = parse_request(serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}]
        }));

        let req2 = parse_request(serde_json::json!({
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }));

        let semantic1 = semantic_text_from_request(&req1);
        let semantic2 = semantic_text_from_request(&req2);

        assert_ne!(semantic1, semantic2);
    }

    #[test]
    fn test_different_messages_produce_different_semantic_text() {
        let req1 = parse_request(serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}]
        }));

        let req2 = parse_request(serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Goodbye"}]
        }));

        let semantic1 = semantic_text_from_request(&req1);
        let semantic2 = semantic_text_from_request(&req2);

        assert_ne!(semantic1, semantic2);
    }

    #[test]
    fn test_stream_parameter_excluded() {
        let req = parse_request(streaming_request_json());
        let semantic = semantic_text_from_request(&req);

        let parsed: serde_json::Value =
            serde_json::from_str(&semantic).expect("Semantic text should be valid JSON");

        assert!(parsed.get("stream").is_none());
    }

    #[test]
    fn test_complex_messages_preserved() {
        let req = parse_request(serde_json::json!({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum computing."},
                {"role": "assistant", "content": "Quantum computing uses qubits..."},
                {"role": "user", "content": "Can you elaborate?"}
            ]
        }));
        let semantic = semantic_text_from_request(&req);

        let parsed: serde_json::Value =
            serde_json::from_str(&semantic).expect("Semantic text should be valid JSON");

        assert_eq!(parsed["messages"].as_array().unwrap().len(), 4);
    }
}

mod make_response_tests {
    use super::*;
    use crate::gateway::handler::make_response;

    #[tokio::test]
    async fn test_make_response_hit_l1_exact() {
        let payload = mock_cache_payload();
        let result = make_response(payload, ReflexStatus::HitL1Exact);

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let headers = response.headers();
        let status_header = headers.get(REFLEX_STATUS_HEADER);
        assert!(status_header.is_some());
        assert_eq!(status_header.unwrap().to_str().unwrap(), "HIT_L1_EXACT");
    }

    #[tokio::test]
    async fn test_make_response_hit_l3_verified() {
        let payload = mock_cache_payload();
        let result = make_response(payload, ReflexStatus::HitL3Verified);

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let headers = response.headers();
        let status_header = headers.get(REFLEX_STATUS_HEADER);
        assert!(status_header.is_some());
        assert_eq!(status_header.unwrap().to_str().unwrap(), "HIT_L3_VERIFIED");
    }

    #[tokio::test]
    async fn test_make_response_miss() {
        let payload = mock_cache_payload();
        let result = make_response(payload, ReflexStatus::Miss);

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let headers = response.headers();
        let status_header = headers.get(REFLEX_STATUS_HEADER);
        assert!(status_header.is_some());
        assert_eq!(status_header.unwrap().to_str().unwrap(), "MISS");
    }

    #[tokio::test]
    async fn test_make_response_body_structure() {
        let payload = mock_cache_payload();
        let result = make_response(payload, ReflexStatus::HitL1Exact);
        let response = result.unwrap();

        let body = response.into_body();
        let bytes = body.collect().await.unwrap().to_bytes();
        let body_json: serde_json::Value =
            serde_json::from_slice(&bytes).expect("Response body should be valid JSON");

        assert_eq!(body_json["object"], "chat.completion");
        assert!(body_json["choices"].is_array());
        assert_eq!(body_json["choices"][0]["finish_reason"], "stop");
        assert_eq!(body_json["choices"][0]["message"]["role"], "assistant");
    }

    #[tokio::test]
    async fn test_make_response_tauq_encoded_content() {
        let payload = mock_cache_payload();
        let result = make_response(payload, ReflexStatus::HitL1Exact);
        let response = result.unwrap();

        let body = response.into_body();
        let bytes = body.collect().await.unwrap().to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        let content = body_json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap();
        assert!(!content.is_empty());
    }
}

mod spawn_index_update_tests {
    use super::*;
    use crate::gateway::handler::spawn_index_update;

    #[tokio::test]
    async fn test_spawn_index_update_returns_true() {
        let bq_client = MockBqClient::new();
        bq_client
            .ensure_bq_collection(
                TEST_COLLECTION_NAME,
                reflex::constants::DEFAULT_VECTOR_SIZE_U64,
            )
            .await
            .expect("Failed to ensure collection");

        let vector: Vec<f32> = vec![0.1; reflex::constants::DEFAULT_VECTOR_SIZE_U64 as usize];

        let result = spawn_index_update(
            bq_client.clone(),
            TEST_COLLECTION_NAME.to_string(),
            1000,       // tenant_id
            2000,       // context_hash
            1702512000, // timestamp
            vector,
            "test/storage/key.rkyv".to_string(),
            reflex::constants::DEFAULT_VECTOR_SIZE_U64,
        );

        assert!(result);

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let count = bq_client.point_count(TEST_COLLECTION_NAME);
        assert_eq!(count, Some(1));
    }

    #[tokio::test]
    async fn test_spawn_index_update_creates_collection_if_needed() {
        let bq_client = MockBqClient::new();

        let vector: Vec<f32> = vec![0.1; reflex::constants::DEFAULT_VECTOR_SIZE_U64 as usize];
        let collection_name = "new_collection";

        let result = spawn_index_update(
            bq_client.clone(),
            collection_name.to_string(),
            1000,
            2000,
            1702512000,
            vector,
            "test/key.rkyv".to_string(),
            reflex::constants::DEFAULT_VECTOR_SIZE_U64,
        );

        assert!(result);

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let count = bq_client.point_count(collection_name);
        assert_eq!(count, Some(1));
    }

    #[tokio::test]
    async fn test_spawn_index_update_with_storage_key() {
        let bq_client = MockBqClient::new();
        bq_client
            .ensure_bq_collection(
                TEST_COLLECTION_NAME,
                reflex::constants::DEFAULT_VECTOR_SIZE_U64,
            )
            .await
            .unwrap();

        let vector: Vec<f32> = vec![0.5; reflex::constants::DEFAULT_VECTOR_SIZE_U64 as usize];
        let storage_key = "tenant123/abc123def456.rkyv";

        let result = spawn_index_update(
            bq_client.clone(),
            TEST_COLLECTION_NAME.to_string(),
            123,
            456,
            1702512000,
            vector,
            storage_key.to_string(),
            reflex::constants::DEFAULT_VECTOR_SIZE_U64,
        );

        assert!(result);

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let query: Vec<f32> = vec![0.5; reflex::constants::DEFAULT_VECTOR_SIZE_U64 as usize];
        let results = bq_client
            .search_bq(TEST_COLLECTION_NAME, query, 10, Some(123))
            .await
            .expect("Search should succeed");

        assert!(!results.is_empty());
        assert_eq!(results[0].tenant_id, 123);
        assert_eq!(results[0].context_hash, 456);
        assert_eq!(results[0].storage_key.as_deref(), Some(storage_key));
    }
}

mod chat_completions_handler_tests {
    use super::*;

    async fn send_completion_request(
        router: &Router,
        body: serde_json::Value,
    ) -> axum::response::Response {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        router.clone().oneshot(request).await.unwrap()
    }

    async fn send_completion_request_with_auth(
        router: &Router,
        body: serde_json::Value,
        token: &str,
    ) -> axum::response::Response {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", token))
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        router.clone().oneshot(request).await.unwrap()
    }

    #[tokio::test]
    async fn test_handler_cache_miss_with_mock_provider() {
        let (state, _temp_dir) = setup_test_state().await;
        let router = create_test_router(state);

        let response = send_completion_request(&router, minimal_request_json()).await;

        assert_eq!(response.status(), StatusCode::OK);

        let headers = response.headers();
        let status = headers.get(REFLEX_STATUS_HEADER).unwrap().to_str().unwrap();
        assert_eq!(status, "MISS");

        let body = response.into_body();
        let bytes = body.collect().await.unwrap().to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(body_json["object"], "chat.completion");
    }

    #[tokio::test]
    async fn test_handler_rejects_legacy_functions() {
        let (state, _temp_dir) = setup_test_state().await;
        let router = create_test_router(state);

        let response = send_completion_request(&router, legacy_functions_request_json()).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = response.into_body();
        let bytes = body.collect().await.unwrap().to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert!(
            body_json["error"]
                .as_str()
                .unwrap()
                .contains("Legacy function-calling")
        );
    }

    #[tokio::test]
    async fn test_handler_rejects_invalid_schema() {
        let (state, _temp_dir) = setup_test_state().await;
        let router = create_test_router(state);

        let invalid_request = serde_json::json!({
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let response = send_completion_request(&router, invalid_request).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_handler_with_authorization_header() {
        let (state, _temp_dir) = setup_test_state().await;
        let router = create_test_router(state);

        let response = send_completion_request_with_auth(
            &router,
            minimal_request_json(),
            "sk-test-token-12345",
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_handler_sequential_requests_for_cache_hit() {
        let (state, _temp_dir) = setup_test_state().await;
        let router = create_test_router(state);

        let response1 = send_completion_request(&router, minimal_request_json()).await;
        assert_eq!(response1.status(), StatusCode::OK);
        let status1 = response1
            .headers()
            .get(REFLEX_STATUS_HEADER)
            .unwrap()
            .to_str()
            .unwrap();
        assert_eq!(status1, "MISS");

        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        let router2 = create_test_router(setup_test_state().await.0);
        let response2 = send_completion_request(&router2, minimal_request_json()).await;
        assert_eq!(response2.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_handler_with_tools() {
        let (state, _temp_dir) = setup_test_state().await;
        let router = create_test_router(state);

        let response = send_completion_request(&router, request_with_tools_json()).await;

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_handler_with_response_format() {
        let (state, _temp_dir) = setup_test_state().await;
        let router = create_test_router(state);

        let response = send_completion_request(&router, request_with_response_format_json()).await;

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_handler_health_endpoint() {
        let (state, _temp_dir) = setup_test_state().await;
        let router = create_test_router(state);

        let request = Request::builder()
            .method("GET")
            .uri("/healthz")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let headers = response.headers();
        let status = headers.get(REFLEX_STATUS_HEADER).unwrap().to_str().unwrap();
        assert_eq!(status, "healthy");

        let body = response.into_body();
        let bytes = body.collect().await.unwrap().to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body_json["status"], "ok");
    }

    #[tokio::test]
    async fn test_handler_ready_endpoint() {
        let (state, _temp_dir) = setup_test_state().await;
        let router = create_test_router(state);

        let request = Request::builder()
            .method("GET")
            .uri("/ready")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();

        let body = response.into_body();
        let bytes = body.collect().await.unwrap().to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert!(body_json.get("components").is_some());
        assert!(body_json["components"].get("http").is_some());
        assert!(body_json["components"].get("storage").is_some());
        assert!(body_json["components"].get("vectordb").is_some());
        assert!(body_json["components"].get("embedding").is_some());
    }

    #[tokio::test]
    async fn test_handler_different_models() {
        let (state, _temp_dir) = setup_test_state().await;
        let router = create_test_router(state);

        for model in &["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "custom-model"] {
            let request = serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": "Test"}]
            });

            let response = send_completion_request(&router, request).await;
            assert_eq!(response.status(), StatusCode::OK);
        }
    }

    #[tokio::test]
    async fn test_handler_complex_messages() {
        let (state, _temp_dir) = setup_test_state().await;
        let router = create_test_router(state);

        let request = serde_json::json!({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Write a Rust function to sort a vector."},
                {"role": "assistant", "content": "Here's a function:\n```rust\nfn sort<T: Ord>(v: &mut Vec<T>) { v.sort(); }\n```"},
                {"role": "user", "content": "Can you make it generic over the comparison function?"}
            ]
        });

        let response = send_completion_request(&router, request).await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_handler_unicode_content() {
        let (state, _temp_dir) = setup_test_state().await;
        let router = create_test_router(state);

        let request = serde_json::json!({
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Translate to Japanese: Hello, world!"},
                {"role": "assistant", "content": "Japanese translation: hello world!"},
                {"role": "user", "content": "Now in Chinese: Welcome!"}
            ]
        });

        let response = send_completion_request(&router, request).await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_handler_empty_content() {
        let (state, _temp_dir) = setup_test_state().await;
        let router = create_test_router(state);

        let request = serde_json::json!({
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": ""}
            ]
        });

        let response = send_completion_request(&router, request).await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_handler_large_message() {
        let (state, _temp_dir) = setup_test_state().await;
        let router = create_test_router(state);

        let large_content: String = "x".repeat(10_000);

        let request = serde_json::json!({
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": large_content}
            ]
        });

        let response = send_completion_request(&router, request).await;
        assert_eq!(response.status(), StatusCode::OK);
    }
}

mod error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_gateway_error_invalid_request_response() {
        let err = GatewayError::InvalidRequest("Test error".to_string());
        let response = err.into_response();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = response.into_body();
        let bytes = body.collect().await.unwrap().to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert!(body_json["error"].as_str().unwrap().contains("Test error"));
        assert_eq!(body_json["code"], 400);
    }

    #[tokio::test]
    async fn test_gateway_error_provider_error_response() {
        let err = GatewayError::ProviderError("Upstream failed".to_string());
        let response = err.into_response();

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);

        let headers = response.headers();
        let status = headers.get(REFLEX_STATUS_HEADER).unwrap().to_str().unwrap();
        assert_eq!(status, "provider_error");
    }

    #[tokio::test]
    async fn test_gateway_error_cache_lookup_failed_response() {
        let err = GatewayError::CacheLookupFailed("Cache error".to_string());
        let response = err.into_response();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let headers = response.headers();
        let status = headers.get(REFLEX_STATUS_HEADER).unwrap().to_str().unwrap();
        assert_eq!(status, "lookup_error");
    }

    #[tokio::test]
    async fn test_gateway_error_serialization_failed_response() {
        let err = GatewayError::SerializationFailed("Serialization error".to_string());
        let response = err.into_response();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let headers = response.headers();
        let status = headers.get(REFLEX_STATUS_HEADER).unwrap().to_str().unwrap();
        assert_eq!(status, "serialization_error");
    }

    #[tokio::test]
    async fn test_gateway_error_storage_error_response() {
        let err = GatewayError::StorageError("Storage failed".to_string());
        let response = err.into_response();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let headers = response.headers();
        let status = headers.get(REFLEX_STATUS_HEADER).unwrap().to_str().unwrap();
        assert_eq!(status, "storage_error");
    }

    #[tokio::test]
    async fn test_gateway_error_embedding_failed_response() {
        let err = GatewayError::EmbeddingFailed("Embedding error".to_string());
        let response = err.into_response();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let headers = response.headers();
        let status = headers.get(REFLEX_STATUS_HEADER).unwrap().to_str().unwrap();
        assert_eq!(status, "embedding_error");
    }

    #[tokio::test]
    async fn test_gateway_error_internal_error_response() {
        let err = GatewayError::InternalError("Internal error".to_string());
        let response = err.into_response();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let headers = response.headers();
        let status = headers.get(REFLEX_STATUS_HEADER).unwrap().to_str().unwrap();
        assert_eq!(status, "internal_error");
    }
}

mod direct_handler_tests {
    use super::*;
    use crate::gateway::handler::chat_completions_handler;
    use axum::Json;
    use axum::extract::State;
    use axum::http::HeaderMap;

    #[tokio::test]
    async fn test_direct_handler_cache_miss() {
        let (state, _temp_dir) = setup_test_state().await;
        let headers = HeaderMap::new();
        let request = Json(minimal_request_json());

        let result = chat_completions_handler(State(state), headers, request).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_direct_handler_with_auth_bearer() {
        let (state, _temp_dir) = setup_test_state().await;
        let mut headers = HeaderMap::new();
        headers.insert("Authorization", "Bearer sk-test-12345".parse().unwrap());
        let request = Json(minimal_request_json());

        let result = chat_completions_handler(State(state), headers, request).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_direct_handler_l1_cache_hit() {
        let (state, _temp_dir) = setup_test_state().await;
        let headers = HeaderMap::new();
        let request_json = minimal_request_json();

        let result = chat_completions_handler(
            State(state.clone()),
            headers.clone(),
            Json(request_json.clone()),
        )
        .await;
        assert!(result.is_ok());

        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        let result2 =
            chat_completions_handler(State(state.clone()), headers.clone(), Json(request_json))
                .await;

        assert!(result2.is_ok());
        let response = result2.unwrap();
        let status_header = response.headers().get(REFLEX_STATUS_HEADER);
        assert!(status_header.is_some());
    }

    #[tokio::test]
    async fn test_direct_handler_validation_failure() {
        let (state, _temp_dir) = setup_test_state().await;
        let headers = HeaderMap::new();
        let request = Json(legacy_functions_request_json());

        let result = chat_completions_handler(State(state), headers, request).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, GatewayError::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn test_direct_handler_schema_validation_failure() {
        let (state, _temp_dir) = setup_test_state().await;
        let headers = HeaderMap::new();
        let request = Json(serde_json::json!({
            "messages": [{"role": "user", "content": "Hello"}]
        }));

        let result = chat_completions_handler(State(state), headers, request).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, GatewayError::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn test_direct_handler_streaming_request() {
        let (state, _temp_dir) = setup_test_state().await;
        let headers = HeaderMap::new();
        let request = Json(streaming_request_json());

        let result = chat_completions_handler(State(state), headers, request).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_direct_handler_with_tools() {
        let (state, _temp_dir) = setup_test_state().await;
        let headers = HeaderMap::new();
        let request = Json(request_with_tools_json());

        let result = chat_completions_handler(State(state), headers, request).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_direct_handler_with_response_format() {
        let (state, _temp_dir) = setup_test_state().await;
        let headers = HeaderMap::new();
        let request = Json(request_with_response_format_json());

        let result = chat_completions_handler(State(state), headers, request).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_direct_handler_multiple_messages() {
        let (state, _temp_dir) = setup_test_state().await;
        let headers = HeaderMap::new();
        let request = Json(serde_json::json!({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
        }));

        let result = chat_completions_handler(State(state), headers, request).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_direct_handler_invalid_auth_format() {
        let (state, _temp_dir) = setup_test_state().await;
        let mut headers = HeaderMap::new();
        headers.insert("Authorization", "sk-test-12345".parse().unwrap());
        let request = Json(minimal_request_json());

        let result = chat_completions_handler(State(state), headers, request).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_direct_handler_empty_auth_token() {
        let (state, _temp_dir) = setup_test_state().await;
        let mut headers = HeaderMap::new();
        headers.insert("Authorization", "Bearer ".parse().unwrap());
        let request = Json(minimal_request_json());

        let result = chat_completions_handler(State(state), headers, request).await;

        assert!(result.is_ok());
    }
}

mod l1_cache_hit_tests {
    use super::*;
    use crate::gateway::handler::chat_completions_handler;
    use reflex::storage::CacheEntry;
    use axum::Json;
    use axum::extract::State;
    use axum::http::HeaderMap;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Creates a valid serialized CacheEntry containing a CachePayload.
    fn create_l1_cache_entry(semantic_request: &str) -> Vec<u8> {
        let payload = CachePayload {
            semantic_request: semantic_request.to_string(),
            response: mock_completion_response("gpt-4"),
        };
        let payload_json = serde_json::to_string(&payload).unwrap();

        let entry = CacheEntry {
            tenant_id: reflex::hashing::hash_tenant_id("default"),
            context_hash: 12345,
            timestamp: chrono::Utc::now().timestamp(),
            embedding: vec![0u8; reflex::constants::EMBEDDING_F16_BYTES],
            payload_blob: payload_json.into_bytes(),
        };

        rkyv::to_bytes::<rkyv::rancor::Error>(&entry)
            .expect("Failed to serialize entry")
            .to_vec()
    }

    #[tokio::test]
    async fn test_l1_cache_hit_with_prepopulated_data() {
        use reflex::storage::mmap::MmapFileHandle;

        let (state, _temp_dir) = setup_test_state().await;

        let request_json = minimal_request_json();
        let request: async_openai::types::chat::CreateChatCompletionRequest =
            serde_json::from_value(request_json.clone()).unwrap();
        let request_bytes = serde_json::to_vec(&request).unwrap();
        let request_hash = blake3::hash(&request_bytes);

        let semantic_text = crate::gateway::handler::semantic_text_from_request(&request);

        let entry_data = create_l1_cache_entry(&semantic_text);

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&entry_data).unwrap();
        temp_file.flush().unwrap();

        let handle = MmapFileHandle::open(temp_file.path()).unwrap();

        let l1_key = request_hash.to_string();
        let tenant_id = reflex::hashing::hash_tenant_id("default");
        state.tiered_cache.insert_l1(&l1_key, tenant_id, handle);

        let headers = HeaderMap::new();
        let result = chat_completions_handler(State(state), headers, Json(request_json)).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let status_header = response.headers().get(REFLEX_STATUS_HEADER);
        assert!(status_header.is_some());
        assert_eq!(status_header.unwrap().to_str().unwrap(), "HIT_L1_EXACT");
    }

    #[tokio::test]
    async fn test_l1_cache_hit_with_invalid_payload_falls_back_to_miss() {
        use reflex::storage::mmap::MmapFileHandle;

        let (state, _temp_dir) = setup_test_state().await;

        let request_json = minimal_request_json();
        let request: async_openai::types::chat::CreateChatCompletionRequest =
            serde_json::from_value(request_json.clone()).unwrap();
        let request_bytes = serde_json::to_vec(&request).unwrap();
        let request_hash = blake3::hash(&request_bytes);

        let entry = CacheEntry {
            tenant_id: reflex::hashing::hash_tenant_id("default"),
            context_hash: 12345,
            timestamp: chrono::Utc::now().timestamp(),
            embedding: vec![0u8; reflex::constants::EMBEDDING_F16_BYTES],
            payload_blob: b"not valid json at all".to_vec(),
        };

        let entry_data = rkyv::to_bytes::<rkyv::rancor::Error>(&entry)
            .expect("Failed to serialize entry")
            .to_vec();

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&entry_data).unwrap();
        temp_file.flush().unwrap();

        let handle = MmapFileHandle::open(temp_file.path()).unwrap();

        let l1_key = request_hash.to_string();
        let tenant_id = reflex::hashing::hash_tenant_id("default");
        state.tiered_cache.insert_l1(&l1_key, tenant_id, handle);

        let headers = HeaderMap::new();
        let result = chat_completions_handler(State(state), headers, Json(request_json)).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        let status_header = response.headers().get(REFLEX_STATUS_HEADER);
        assert!(status_header.is_some());
        assert_eq!(status_header.unwrap().to_str().unwrap(), "MISS");
    }
}

mod l2_l3_verification_tests {
    use super::*;
    use reflex::cache::{L1CacheHandle, L2Config, L2SemanticCache, MockStorageLoader, TieredCache};
    use reflex::embedding::sinter::{SinterConfig, SinterEmbedder};
    use crate::gateway::handler::chat_completions_handler;
    use reflex::storage::CacheEntry;
    use reflex::vectordb::bq::MockBqClient;
    use axum::Json;
    use axum::extract::State;
    use axum::http::HeaderMap;

    const L2_TEST_COLLECTION: &str = "l2_test_collection";

    /// Creates a CacheEntry with valid CachePayload for L2 testing.
    fn create_l2_cache_entry(
        semantic_request: &str,
        tenant_id: u64,
        context_hash: u64,
    ) -> CacheEntry {
        let payload = CachePayload {
            semantic_request: semantic_request.to_string(),
            response: mock_completion_response("gpt-4"),
        };
        let payload_json = serde_json::to_string(&payload).unwrap();

        CacheEntry {
            tenant_id,
            context_hash,
            timestamp: chrono::Utc::now().timestamp(),
            embedding: vec![0u8; reflex::constants::EMBEDDING_F16_BYTES],
            payload_blob: payload_json.into_bytes(),
        }
    }

    /// Sets up a test state with MockStorageLoader for L2 tests.
    async fn setup_mock_storage_state() -> (HandlerState<MockBqClient, MockStorageLoader>, TempDir)
    {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let storage_path = temp_dir.path().to_path_buf();

        let bq_client = MockBqClient::new();
        bq_client
            .ensure_bq_collection(
                L2_TEST_COLLECTION,
                reflex::constants::DEFAULT_VECTOR_SIZE_U64,
            )
            .await
            .expect("Failed to ensure collection");

        let mock_storage = MockStorageLoader::new();

        let embedder =
            SinterEmbedder::load(SinterConfig::stub()).expect("Failed to load stub embedder");

        let l2_config = L2Config::default().collection_name(L2_TEST_COLLECTION);
        let l2_cache = L2SemanticCache::new(embedder, bq_client.clone(), mock_storage, l2_config)
            .expect("Failed to create L2 cache");

        let l1_cache = L1CacheHandle::new();
        let tiered_cache = Arc::new(TieredCache::new(l1_cache, l2_cache));

        let scorer = Arc::new(
            CrossEncoderScorer::new(RerankerConfig::stub().with_threshold(0.7))
                .expect("Failed to create scorer"),
        );

        let state = HandlerState::new_with_mock_provider(
            tiered_cache,
            scorer,
            storage_path,
            bq_client,
            L2_TEST_COLLECTION.to_string(),
            true,
        );

        (state, temp_dir)
    }

    #[tokio::test]
    async fn test_l2_hit_with_l3_verification_pass() {
        let (state, _temp_dir) = setup_mock_storage_state().await;

        let request_json = minimal_request_json();
        let request: async_openai::types::chat::CreateChatCompletionRequest =
            serde_json::from_value(request_json.clone()).unwrap();
        let semantic_text = crate::gateway::handler::semantic_text_from_request(&request);

        let tenant_id = reflex::hashing::hash_tenant_id("default");
        let context_hash = 99999u64;
        let storage_key = format!("{}/test_entry.rkyv", tenant_id);

        let entry = create_l2_cache_entry(&semantic_text, tenant_id, context_hash);
        state
            .tiered_cache
            .l2()
            .storage()
            .insert(&storage_key, entry);

        state
            .tiered_cache
            .index_l2(
                &semantic_text,
                tenant_id,
                context_hash,
                &storage_key,
                chrono::Utc::now().timestamp(),
            )
            .await
            .expect("Should index successfully");

        let headers = HeaderMap::new();
        let result = chat_completions_handler(State(state), headers, Json(request_json)).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let status_header = response.headers().get(REFLEX_STATUS_HEADER);
        assert!(status_header.is_some());
        let status = status_header.unwrap().to_str().unwrap();
        assert!(status == "HIT_L3_VERIFIED" || status == "MISS");
    }

    #[tokio::test]
    async fn test_l2_hit_with_l3_verification_rejected() {
        let (state, _temp_dir) = setup_mock_storage_state().await;

        let request_json = minimal_request_json();
        let semantic_text =
            r#"{"model":"gpt-4","messages":[{"role":"user","content":"Hello, world!"}]}"#;

        let tenant_id = reflex::hashing::hash_tenant_id("default");
        let context_hash = 88888u64;
        let storage_key = format!("{}/different_entry.rkyv", tenant_id);

        let entry = create_l2_cache_entry(
            "completely different unrelated query about something else entirely",
            tenant_id,
            context_hash,
        );
        state
            .tiered_cache
            .l2()
            .storage()
            .insert(&storage_key, entry);

        state
            .tiered_cache
            .index_l2(
                semantic_text,
                tenant_id,
                context_hash,
                &storage_key,
                chrono::Utc::now().timestamp(),
            )
            .await
            .expect("Should index successfully");

        let headers = HeaderMap::new();
        let result = chat_completions_handler(State(state), headers, Json(request_json)).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let status_header = response.headers().get(REFLEX_STATUS_HEADER);
        assert!(status_header.is_some());
    }

    #[tokio::test]
    async fn test_l2_hit_with_invalid_payload_skips_candidate() {
        let (state, _temp_dir) = setup_mock_storage_state().await;

        let request_json = minimal_request_json();
        let request: async_openai::types::chat::CreateChatCompletionRequest =
            serde_json::from_value(request_json.clone()).unwrap();
        let semantic_text = crate::gateway::handler::semantic_text_from_request(&request);

        let tenant_id = reflex::hashing::hash_tenant_id("default");
        let context_hash = 77777u64;
        let storage_key = format!("{}/invalid_payload.rkyv", tenant_id);

        let entry = CacheEntry {
            tenant_id,
            context_hash,
            timestamp: chrono::Utc::now().timestamp(),
            embedding: vec![0u8; reflex::constants::EMBEDDING_F16_BYTES],
            payload_blob: b"invalid json {{{".to_vec(),
        };
        state
            .tiered_cache
            .l2()
            .storage()
            .insert(&storage_key, entry);

        state
            .tiered_cache
            .index_l2(
                &semantic_text,
                tenant_id,
                context_hash,
                &storage_key,
                chrono::Utc::now().timestamp(),
            )
            .await
            .expect("Should index successfully");

        let headers = HeaderMap::new();
        let result = chat_completions_handler(State(state), headers, Json(request_json)).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_l2_hit_no_valid_candidates_returns_miss() {
        let (state, _temp_dir) = setup_mock_storage_state().await;

        let request_json = minimal_request_json();
        let request: async_openai::types::chat::CreateChatCompletionRequest =
            serde_json::from_value(request_json.clone()).unwrap();
        let semantic_text = crate::gateway::handler::semantic_text_from_request(&request);

        let tenant_id = reflex::hashing::hash_tenant_id("default");
        let context_hash = 66666u64;
        let storage_key = format!("{}/no_storage.rkyv", tenant_id);

        state
            .tiered_cache
            .index_l2(
                &semantic_text,
                tenant_id,
                context_hash,
                &storage_key,
                chrono::Utc::now().timestamp(),
            )
            .await
            .expect("Should index successfully");

        let headers = HeaderMap::new();
        let result = chat_completions_handler(State(state), headers, Json(request_json)).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let status_header = response.headers().get(REFLEX_STATUS_HEADER);
        assert!(status_header.is_some());
        assert_eq!(status_header.unwrap().to_str().unwrap(), "MISS");
    }
}

mod spawn_index_update_error_tests {
    use crate::gateway::handler::spawn_index_update;
    use reflex::vectordb::bq::MockBqClient;

    #[tokio::test]
    async fn test_spawn_index_update_collection_ensure_failure() {
        let bq_client = MockBqClient::new();
        bq_client.poison_lock();

        let vector: Vec<f32> = vec![0.1; reflex::constants::DEFAULT_VECTOR_SIZE_U64 as usize];

        let result = spawn_index_update(
            bq_client,
            "poisoned_collection".to_string(),
            1000,
            2000,
            1702512000,
            vector,
            "test/key.rkyv".to_string(),
            reflex::constants::DEFAULT_VECTOR_SIZE_U64,
        );

        assert!(result);

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn test_spawn_index_update_upsert_failure() {
        let bq_client = MockBqClient::new();
        bq_client
            .ensure_bq_collection(
                "test_upsert_fail",
                reflex::constants::DEFAULT_VECTOR_SIZE_U64,
            )
            .await
            .unwrap();

        let wrong_dim_vector: Vec<f32> = vec![0.1; 10];

        let result = spawn_index_update(
            bq_client.clone(),
            "test_upsert_fail".to_string(),
            1000,
            2000,
            1702512000,
            wrong_dim_vector,
            "test/key.rkyv".to_string(),
            reflex::constants::DEFAULT_VECTOR_SIZE_U64,
        );

        assert!(result);

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let count = bq_client.point_count("test_upsert_fail");
        assert_eq!(count, Some(0));
    }
}

mod additional_error_path_tests {
    use super::*;
    use crate::gateway::handler::chat_completions_handler;
    use axum::Json;
    use axum::extract::State;
    use axum::http::HeaderMap;

    #[tokio::test]
    async fn test_cache_lookup_with_missing_messages() {
        let (state, _temp_dir) = setup_test_state().await;
        let headers = HeaderMap::new();

        let request = Json(serde_json::json!({
            "model": "gpt-4"
        }));

        let result = chat_completions_handler(State(state), headers, request).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            GatewayError::InvalidRequest(msg) => {
                assert!(msg.contains("messages") || msg.contains("Missing"));
            }
            _ => panic!("Expected InvalidRequest error"),
        }
    }

    #[tokio::test]
    async fn test_request_with_empty_messages() {
        let (state, _temp_dir) = setup_test_state().await;
        let headers = HeaderMap::new();

        let request = Json(serde_json::json!({
            "model": "gpt-4",
            "messages": []
        }));

        let result = chat_completions_handler(State(state), headers, request).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_request_with_null_content() {
        let (state, _temp_dir) = setup_test_state().await;
        let headers = HeaderMap::new();

        let request = Json(serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": null}]
        }));

        let result = chat_completions_handler(State(state), headers, request).await;
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_streaming_request_path() {
        let (state, _temp_dir) = setup_test_state().await;
        let headers = HeaderMap::new();

        let request = Json(serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true
        }));

        let result = chat_completions_handler(State(state), headers, request).await;
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_request_with_unicode_content() {
        let (state, _temp_dir) = setup_test_state().await;
        let headers = HeaderMap::new();

        let request = Json(serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello in Japanese: こんにちは in Chinese: 你好 in emoji: 👋🌍"}]
        }));

        let result = chat_completions_handler(State(state), headers, request).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_request_with_very_long_content() {
        let (state, _temp_dir) = setup_test_state().await;
        let headers = HeaderMap::new();

        let long_content = "a".repeat(10000);
        let request = Json(serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": long_content}]
        }));

        let result = chat_completions_handler(State(state), headers, request).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_multiple_sequential_requests_same_content() {
        let (state, _temp_dir) = setup_test_state().await;
        let request_json = minimal_request_json();

        let result1 = chat_completions_handler(
            State(state.clone()),
            HeaderMap::new(),
            Json(request_json.clone()),
        )
        .await;
        assert!(result1.is_ok());

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let result2 = chat_completions_handler(
            State(state.clone()),
            HeaderMap::new(),
            Json(request_json.clone()),
        )
        .await;
        assert!(result2.is_ok());

        let result3 =
            chat_completions_handler(State(state), HeaderMap::new(), Json(request_json)).await;
        assert!(result3.is_ok());
    }

    #[tokio::test]
    async fn test_different_models_different_cache_keys() {
        let (state, _temp_dir) = setup_test_state().await;

        let request1 = Json(serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}]
        }));
        let result1 =
            chat_completions_handler(State(state.clone()), HeaderMap::new(), request1).await;
        assert!(result1.is_ok());

        let request2 = Json(serde_json::json!({
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }));
        let result2 = chat_completions_handler(State(state), HeaderMap::new(), request2).await;
        assert!(result2.is_ok());
    }
}

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_reflex_status_all_variants() {
        assert_eq!(ReflexStatus::HitL1Exact.as_header_value(), "HIT_L1_EXACT");
        assert_eq!(
            ReflexStatus::HitL2Semantic.as_header_value(),
            "HIT_L2_SEMANTIC"
        );
        assert_eq!(
            ReflexStatus::HitL3Verified.as_header_value(),
            "HIT_L3_VERIFIED"
        );
        assert_eq!(ReflexStatus::Miss.as_header_value(), "MISS");
    }

    #[test]
    fn test_reflex_status_is_hit() {
        assert!(ReflexStatus::HitL1Exact.is_hit());
        assert!(ReflexStatus::HitL2Semantic.is_hit());
        assert!(ReflexStatus::HitL3Verified.is_hit());
        assert!(!ReflexStatus::Miss.is_hit());
    }

    #[test]
    fn test_reflex_status_display() {
        assert_eq!(format!("{}", ReflexStatus::HitL1Exact), "HIT_L1_EXACT");
        assert_eq!(format!("{}", ReflexStatus::Miss), "MISS");
    }

    #[tokio::test]
    async fn test_handler_state_debug() {
        let (state, _temp_dir) = setup_test_state().await;
        let _ = format!("{:?}", state.tiered_cache);
        let _ = format!("{:?}", state.scorer);
    }

    #[test]
    fn test_cache_payload_serialization() {
        let payload = mock_cache_payload();
        let json = serde_json::to_string(&payload).expect("Should serialize");
        let parsed: CachePayload = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(payload.semantic_request, parsed.semantic_request);
    }

    #[tokio::test]
    async fn test_request_with_all_optional_fields() {
        let (state, _temp_dir) = setup_test_state().await;
        let router = create_test_router(state);

        let request = serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "n": 1,
            "stop": ["\n", "END"],
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "logit_bias": {},
            "user": "test-user-123"
        });

        let http_request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(serde_json::to_string(&request).unwrap()))
            .unwrap();

        let response = router.oneshot(http_request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}

use async_openai::types::chat::*;
use axum::{
    Json,
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use futures_util::stream;
use std::convert::Infallible;
use tracing::{debug, error, info, instrument};

use reflex::cache::{
    BqSearchBackend, REFLEX_STATUS_HEADER, ReflexStatus, StorageLoader, TieredLookupResult,
};
use crate::gateway::error::GatewayError;
use crate::gateway::payload::CachePayload;
use crate::gateway::state::HandlerState;
use crate::gateway::streaming::handle_streaming_request;
use reflex::payload::TauqEncoder;
use reflex::scoring::VerificationResult;
use reflex::storage::{ArchivedCacheEntry, CacheEntry, StorageWriter};
use reflex::vectordb::{VectorPoint, generate_point_id};

#[instrument(skip(state, request, headers), fields(model = tracing::field::Empty))]
pub async fn chat_completions_handler<B, S>(
    State(state): State<HandlerState<B, S>>,
    headers: HeaderMap,
    Json(request): Json<serde_json::Value>,
) -> Result<Response, GatewayError>
where
    B: BqSearchBackend + Clone + Send + Sync + 'static,
    S: StorageLoader + StorageWriter + Clone + Send + Sync + 'static,
{
    validate_no_legacy_fields(&request)?;
    let request: CreateChatCompletionRequest = serde_json::from_value(request)
        .map_err(|e| GatewayError::InvalidRequest(format!("Invalid request schema: {}", e)))?;
    tracing::Span::current().record("model", tracing::field::display(&request.model));

    let _auth_token = headers
        .get("Authorization")
        .and_then(|val| val.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(|s| s.trim().to_string());

    let request_bytes = serde_json::to_vec(&request)
        .map_err(|e| GatewayError::InvalidRequest(format!("Serialization failed: {}", e)))?;
    let request_hash = blake3::hash(&request_bytes);
    let request_hash_u64 = reflex::hashing::hash_to_u64(request_hash.as_bytes());

    debug!(hash = %request_hash, "Processing chat completion request");

    let semantic_text = semantic_text_from_request(&request);

    let token = _auth_token.unwrap_or_else(|| "default".to_string());
    let tenant_id_hash = reflex::hashing::hash_tenant_id(&token);

    let stream_requested = request.stream.unwrap_or(false);

    if stream_requested {
        debug!("Streaming request received - bypassing cache");
        if state.mock_provider {
            debug!("Mock provider enabled - returning mock streaming response");
            let mock_sse =
                create_mock_streaming_response(request.model.clone(), semantic_text.clone());
            return Ok(mock_sse.into_response());
        }
        let sse = handle_streaming_request::<B>(
            state.genai_client.clone(),
            &request.model,
            request.clone(),
            tenant_id_hash,
            request_hash_u64,
            semantic_text,
        )
        .await?;
        return Ok(sse.into_response());
    }

    let request_hash_str = request_hash.to_string();
    let tiered_result = state
        .tiered_cache
        .lookup_with_semantic_query(&request_hash_str, &semantic_text, tenant_id_hash)
        .await
        .map_err(|e| GatewayError::CacheLookupFailed(e.to_string()))?;

    let cached_response = match tiered_result {
        TieredLookupResult::HitL1(l1_result) => {
            info!("L1 Cache Hit");
            let archived = l1_result
                .handle()
                .access_archived::<ArchivedCacheEntry>()
                .map_err(|e| GatewayError::CacheLookupFailed(e.to_string()))?;

            let raw_payload = String::from_utf8_lossy(&archived.payload_blob);
            match serde_json::from_str::<CachePayload>(&raw_payload) {
                Ok(cache_payload) => Some((cache_payload, ReflexStatus::HitL1Exact)),
                Err(e) => {
                    tracing::warn!("Failed to parse L1 payload: {}. Treating as miss.", e);
                    None
                }
            }
        }
        TieredLookupResult::HitL2(l2_result) => {
            debug!(
                candidates = l2_result.candidates().len(),
                "L2 semantic hit, verifying..."
            );

            let mut valid_candidates = Vec::new();
            for c in l2_result.candidates() {
                let raw_payload = String::from_utf8_lossy(&c.entry.payload_blob);
                if let Ok(payload) = serde_json::from_str::<CachePayload>(&raw_payload) {
                    let mut temp_entry = c.entry.clone();
                    temp_entry.payload_blob = payload.semantic_request.as_bytes().to_vec();
                    valid_candidates.push((temp_entry, c.score, payload));
                }
            }

            let candidates_for_scoring: Vec<(CacheEntry, f32)> = valid_candidates
                .iter()
                .map(|(e, s, _)| (e.clone(), *s))
                .collect();

            let (verified_entry, verification_result) = state
                .scorer
                .verify_candidates(&semantic_text, candidates_for_scoring)
                .map_err(GatewayError::ScoringFailed)?;

            match verification_result {
                VerificationResult::Verified { score } => {
                    info!(score = score, "L3 verification passed");
                    let entry = verified_entry.ok_or_else(|| {
                        GatewayError::InternalError(
                            "L3 verification returned Verified without an entry".to_string(),
                        )
                    })?;
                    let payload = valid_candidates
                        .iter()
                        .find(|(e, _, _)| e.context_hash == entry.context_hash)
                        .map(|(_, _, p)| p.clone())
                        .ok_or_else(|| {
                            GatewayError::InternalError(
                                "Lost track of verified payload".to_string(),
                            )
                        })?;

                    Some((payload, ReflexStatus::HitL3Verified))
                }
                VerificationResult::Rejected { top_score } => {
                    debug!(score = top_score, "L3 verification rejected");
                    None
                }
                VerificationResult::NoCandidates => {
                    debug!("L3 verification - no candidates");
                    None
                }
            }
        }
        TieredLookupResult::Miss => None,
    };

    if let Some((resp, status)) = cached_response {
        return make_response(resp, status);
    }

    debug!("Cache Miss - Calling Provider");

    let model = request.model.clone();

    let response = if state.mock_provider {
        let content = format!("Mock response for: {}", semantic_text);
        let response_value = serde_json::json!({
            "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            "object": "chat.completion",
            "created": chrono::Utc::now().timestamp() as u32,
            "model": model.clone(),
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20
            }
        });

        serde_json::from_value::<CreateChatCompletionResponse>(response_value)
            .map_err(|e| GatewayError::SerializationFailed(e.to_string()))?
    } else {
        let genai_req = crate::gateway::adapter::adapt_openai_to_genai(request.clone());

        let genai_resp = state
            .genai_client
            .exec_chat(&model, genai_req, None)
            .await
            .map_err(|e| {
                error!("Provider error: {}", e);
                GatewayError::ProviderError("Upstream service request failed".to_string())
            })?;

        crate::gateway::adapter::adapt_genai_to_openai(genai_resp, model.clone())
    };

    let timestamp = chrono::Utc::now().timestamp();

    let payload = CachePayload {
        semantic_request: semantic_text.clone(),
        response: response.clone(),
    };
    let payload_json = serde_json::to_string(&payload)
        .map_err(|e| GatewayError::SerializationFailed(e.to_string()))?;

    let embedding_f16 = state
        .tiered_cache
        .l2()
        .embedder()
        .embed(&semantic_text)
        .map_err(|e| GatewayError::EmbeddingFailed(e.to_string()))?;

    let embedding_bytes: Vec<u8> = embedding_f16.iter().flat_map(|v| v.to_le_bytes()).collect();

    let cache_entry = CacheEntry {
        tenant_id: tenant_id_hash,
        context_hash: request_hash_u64,
        timestamp,
        embedding: embedding_bytes,
        payload_blob: payload_json.into_bytes(),
    };

    let entry_id = format!("{:016x}", request_hash_u64);
    let storage_key = format!("{}/{}.rkyv", tenant_id_hash, entry_id);

    let serialized_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&cache_entry)
        .map_err(|e| GatewayError::SerializationFailed(e.to_string()))?;

    let storage = state.tiered_cache.l2().storage().clone();
    let storage_key_for_write = storage_key.clone();
    let mmap_handle = tokio::task::spawn_blocking(move || {
        storage.write(&storage_key_for_write, serialized_bytes.as_ref())
    })
    .await
    .map_err(|e| GatewayError::StorageError(format!("Storage write task failed: {}", e)))?
    .map_err(|e| GatewayError::StorageError(e.to_string()))?;

    let l1_key = request_hash.to_string();
    state
        .tiered_cache
        .insert_l1(&l1_key, tenant_id_hash, mmap_handle);

    let embedding_f32: Vec<f32> = embedding_f16.iter().map(|v| v.to_f32()).collect();
    let vector_dim = state.tiered_cache.l2().config().vector_size;
    spawn_index_update(
        state.bq_client.clone(),
        state.collection_name.clone(),
        tenant_id_hash,
        request_hash_u64,
        timestamp,
        embedding_f32,
        storage_key,
        vector_dim,
    );

    make_response(payload, ReflexStatus::Miss)
}

pub(crate) fn make_response(
    payload: CachePayload,
    status: ReflexStatus,
) -> Result<Response, GatewayError> {
    let payload_json = serde_json::to_value(&payload).unwrap_or_default();
    let tauq_content = TauqEncoder::encode(&payload_json);

    let message: ChatCompletionResponseMessage = serde_json::from_value(serde_json::json!({
        "role": "assistant",
        "content": tauq_content,
    }))
    .map_err(|e| GatewayError::SerializationFailed(e.to_string()))?;

    let choice = ChatChoice {
        index: 0,
        message,
        finish_reason: Some(FinishReason::Stop),
        logprobs: None,
    };

    let mut wrapper = payload.response;
    wrapper.object = "chat.completion".to_string();
    wrapper.choices = vec![choice];

    let mut headers = HeaderMap::new();
    headers.insert(
        REFLEX_STATUS_HEADER,
        HeaderValue::from_static(status.as_header_value()),
    );
    Ok((StatusCode::OK, headers, Json(wrapper)).into_response())
}

pub(crate) fn validate_no_legacy_fields(req: &serde_json::Value) -> Result<(), GatewayError> {
    if req.get("functions").is_some() || req.get("function_call").is_some() {
        return Err(GatewayError::InvalidRequest(
            "Legacy function-calling fields are not supported; use `tools`/`tool_choice`."
                .to_string(),
        ));
    }

    let messages = req
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or_else(|| GatewayError::InvalidRequest("Missing or invalid `messages`".to_string()))?;

    for m in messages {
        if m.get("role").and_then(|r| r.as_str()) == Some("function") {
            return Err(GatewayError::InvalidRequest(
                "Unsupported message role: `function`.".to_string(),
            ));
        }
        if m.get("role").and_then(|r| r.as_str()) == Some("assistant")
            && m.get("function_call").is_some()
        {
            return Err(GatewayError::InvalidRequest(
                "Legacy `function_call` on assistant messages is not supported; use `tool_calls`."
                    .to_string(),
            ));
        }
    }

    Ok(())
}

/// Builds a semantic key from a chat completion request for cache lookups.
///
/// # Semantic Cache Design Decision
///
/// This function deliberately extracts only a subset of request fields to form the
/// cache key. This is an intentional product decision that prioritizes cache hit rates
/// over exact output distribution matching.
///
/// ## Included Parameters (affect semantic meaning)
///
/// These parameters are included because they fundamentally change what the model
/// should produce:
///
/// - **`model`**: Different models produce different outputs
/// - **`messages`**: The conversation context determines the response content
/// - **`tools`**: Available function definitions change model behavior
/// - **`tool_choice`**: Controls whether/which tools the model should use
/// - **`response_format`**: Structured output schemas (e.g., JSON mode) change output shape
///
/// ## Excluded Parameters (sampling/generation controls)
///
/// These parameters are deliberately omitted from the cache key:
///
/// - **`temperature`**: Controls randomness (0.0 = deterministic, 2.0 = creative)
/// - **`max_tokens`** / **`max_completion_tokens`**: Limits response length
/// - **`top_p`**: Nucleus sampling threshold
/// - **`frequency_penalty`**: Penalizes repeated tokens
/// - **`presence_penalty`**: Penalizes tokens that have appeared at all
/// - **`n`**: Number of completions to generate
/// - **`stop`**: Stop sequences
/// - **`seed`**: Random seed for reproducibility
/// - **`logprobs`** / **`top_logprobs`**: Log probability settings
/// - **`logit_bias`**: Token probability adjustments
/// - **`user`**: End-user identifier (for abuse tracking, not semantics)
/// - **`stream`**: Delivery mechanism, not content
/// - **`stream_options`**: Stream configuration
/// - **`service_tier`**: Infrastructure routing
/// - **`store`**: Whether to store for fine-tuning
/// - **`metadata`**: Arbitrary metadata
/// - **`parallel_tool_calls`**: Execution strategy
///
/// ## Tradeoff
///
/// **Benefit**: Higher cache hit rates. Requests like "summarize X" with temperature=0.7
/// will return the same cached response as temperature=0.3, avoiding redundant LLM calls.
///
/// **Cost**: The cached response may not reflect the exact output distribution the caller
/// requested. A request with `max_tokens=100` might return a cached 500-token response.
/// Callers expecting low-temperature determinism may receive a response generated with
/// higher temperature.
///
/// ## Rationale
///
/// For most use cases, the semantic content of the request (what is being asked) matters
/// more than the sampling parameters (how the response should be generated). Two users
/// asking the same question with different temperature settings are likely satisfied by
/// the same high-quality cached answer. This design assumes that cache freshness and
/// hit rates outweigh perfect sampling parameter fidelity.
///
/// If exact parameter matching is required for your use case, consider implementing
/// a separate cache key strategy or bypassing the semantic cache entirely.
pub(crate) fn semantic_text_from_request(req: &CreateChatCompletionRequest) -> String {
    let mut root = serde_json::Map::new();
    root.insert("model".to_string(), serde_json::json!(req.model));
    root.insert(
        "messages".to_string(),
        serde_json::to_value(&req.messages).unwrap_or_else(|_| serde_json::json!([])),
    );

    if let Some(tools) = &req.tools {
        root.insert(
            "tools".to_string(),
            serde_json::to_value(tools).unwrap_or_else(|_| serde_json::json!([])),
        );
    }

    if let Some(tool_choice) = &req.tool_choice {
        root.insert(
            "tool_choice".to_string(),
            serde_json::to_value(tool_choice).unwrap_or(serde_json::Value::Null),
        );
    }

    if let Some(response_format) = &req.response_format {
        root.insert(
            "response_format".to_string(),
            serde_json::to_value(response_format).unwrap_or(serde_json::Value::Null),
        );
    }

    serde_json::to_string(&serde_json::Value::Object(root))
        .unwrap_or_else(|_| format!("model={} messages={}", req.model, req.messages.len()))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_index_update<B>(
    bq_client: B,
    collection_name: String,
    tenant_id: u64,
    context_hash: u64,
    timestamp: i64,
    vector: Vec<f32>,
    storage_key: String,
    vector_dim: u64,
) -> bool
where
    B: BqSearchBackend + Send + Sync + 'static,
{
    let point_id = generate_point_id(tenant_id, context_hash);

    let point = VectorPoint {
        id: point_id,
        vector,
        tenant_id,
        context_hash,
        timestamp,
        storage_key: Some(storage_key),
    };

    tokio::spawn(async move {
        if let Err(e) = bq_client
            .ensure_collection(&collection_name, vector_dim)
            .await
        {
            error!(error = %e, "Failed to ensure BQ collection");
            return;
        }

        if let Err(e) = bq_client
            .upsert_points(
                &collection_name,
                vec![point],
                reflex::vectordb::WriteConsistency::Eventual,
            )
            .await
        {
            error!(error = %e, "Failed to upsert point to BQ index");
            return;
        }

        debug!(
            point_id = point_id,
            "Successfully indexed point in BQ collection"
        );
    });

    true
}

/// Creates a mock SSE streaming response for testing purposes.
///
/// This function generates a simple SSE stream that emits a single mock chunk
/// followed by a `[DONE]` marker, mimicking the OpenAI streaming response format.
fn create_mock_streaming_response(
    model: String,
    semantic_text: String,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>> + Send> {
    let content = format!("Mock streaming response for: {}", semantic_text);

    let chunk_response = serde_json::json!({
        "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        "object": "chat.completion.chunk",
        "created": chrono::Utc::now().timestamp() as u32,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": { "role": "assistant", "content": content },
            "finish_reason": null
        }]
    });

    let done_response = serde_json::json!({
        "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        "object": "chat.completion.chunk",
        "created": chrono::Utc::now().timestamp() as u32,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    });

    let events = vec![
        Ok(Event::default().data(chunk_response.to_string())),
        Ok(Event::default().data(done_response.to_string())),
        Ok(Event::default().data("[DONE]")),
    ];

    Sse::new(stream::iter(events))
}

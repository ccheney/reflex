use async_openai::types::chat::{
    ChatChoiceStream, ChatCompletionStreamResponseDelta, CreateChatCompletionRequest,
    CreateChatCompletionStreamResponse,
};
use axum::response::sse::{Event, Sse};
use futures_util::stream::{Stream, StreamExt};
use genai::Client;
use genai::chat::ChatStreamEvent;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::error;

use crate::cache::BqSearchBackend;
use crate::gateway::adapter::adapt_openai_to_genai;
use crate::gateway::error::GatewayError;

/// Handles streaming chat completion requests, bypassing the semantic cache.
///
/// # Cache Bypass Rationale
///
/// Streaming requests deliberately bypass the cache for several reasons:
/// - **Incremental delivery**: Streaming responses are delivered chunk-by-chunk via SSE,
///   making traditional cache lookup/store semantics inefficient
/// - **Latency sensitivity**: Streaming is typically chosen for real-time feedback;
///   cache overhead would negate this benefit
/// - **Response variability**: Partial responses and timing are inherently variable,
///   making cache hit rates low and storage wasteful
///
/// # Type Parameter `B`
///
/// The generic parameter `B: BqSearchBackend` is retained for API consistency with
/// non-streaming handlers (e.g., `handle_chat_completion`) even though it is not used
/// in the function body. This allows callers to use a uniform handler signature and
/// simplifies router configuration where both streaming and non-streaming paths share
/// the same backend type.
///
/// # Future Work
///
/// Potential enhancements for streaming cache support:
/// - **Accumulated response caching**: Store the fully accumulated response after stream
///   completion (see `_accumulated_content` placeholder) for subsequent non-streaming lookups
/// - **Prefix caching**: Cache partial responses to enable "continuation" semantics
/// - **Semantic deduplication**: Detect duplicate streaming requests in-flight and fan-out
///   a single upstream stream to multiple clients
pub async fn handle_streaming_request<B>(
    client: Client,
    model: &str,
    request: CreateChatCompletionRequest,
    _tenant_id_hash: u64,
    _context_hash: u64,
    _semantic_text: String,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>> + Send + 'static>, GatewayError>
where
    B: BqSearchBackend + Clone + Send + Sync + 'static,
{
    let genai_req = adapt_openai_to_genai(request.clone());
    let model_owned = model.to_string();

    let chat_stream_resp = client
        .exec_chat_stream(&model_owned, genai_req, None)
        .await
        .map_err(|e| {
            error!("Provider stream init error: {}", e);
            GatewayError::ProviderError("Upstream service stream init failed".to_string())
        })?;

    let stream = chat_stream_resp.stream;

    let _accumulated_content = Arc::new(Mutex::new(String::new()));

    let event_stream = stream.map(move |result| match result {
        Ok(ChatStreamEvent::Start) => Ok(Event::default().comment("start")),
        Ok(ChatStreamEvent::Chunk(chunk)) => {
            let text = chunk.content;
            if !text.is_empty() {
                let delta: ChatCompletionStreamResponseDelta = match serde_json::from_value(
                    serde_json::json!({ "role": "assistant", "content": text }),
                ) {
                    Ok(d) => d,
                    Err(e) => {
                        error!("Failed to construct delta: {}", e);
                        return Ok(Event::default().comment("delta-error"));
                    }
                };

                let response: CreateChatCompletionStreamResponse =
                    match serde_json::from_value(serde_json::json!({
                        "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                        "object": "chat.completion.chunk",
                        "created": chrono::Utc::now().timestamp() as u32,
                        "model": model_owned.clone(),
                        "choices": vec![ChatChoiceStream {
                            index: 0,
                            delta,
                            finish_reason: None,
                            logprobs: None,
                        }],
                        "usage": serde_json::Value::Null,
                    })) {
                        Ok(r) => r,
                        Err(e) => {
                            error!("Failed to construct streaming response: {}", e);
                            return Ok(Event::default().comment("delta-error"));
                        }
                    };

                match serde_json::to_string(&response) {
                    Ok(json) => Ok(Event::default().data(json)),
                    Err(e) => {
                        error!("Failed to serialize response: {}", e);
                        Ok(Event::default().comment("serialization-error"))
                    }
                }
            } else {
                Ok(Event::default().comment("keep-alive"))
            }
        }
        Ok(ChatStreamEvent::End(_)) => Ok(Event::default().data("[DONE]")),
        Ok(_) => Ok(Event::default().comment("ignored-event")),
        Err(e) => {
            error!("Stream error: {}", e);
            Ok(Event::default()
                .event("error")
                .data("Stream interrupted by upstream error"))
        }
    });

    Ok(Sse::new(event_stream))
}

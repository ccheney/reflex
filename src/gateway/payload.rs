use async_openai::types::chat::CreateChatCompletionResponse;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CachePayload {
    pub semantic_request: String,
    pub response: CreateChatCompletionResponse,
}

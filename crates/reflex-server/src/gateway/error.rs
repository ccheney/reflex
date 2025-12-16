use axum::{
    Json,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use thiserror::Error;

use reflex::cache::REFLEX_STATUS_HEADER;
use reflex::scoring::ScoringError;

#[derive(Debug, Error)]
pub enum GatewayError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("cache lookup failed: {0}")]
    CacheLookupFailed(String),

    #[error("scoring failed: {0}")]
    ScoringFailed(#[from] ScoringError),

    #[error("provider error: {0}")]
    ProviderError(String),

    #[error("serialization failed: {0}")]
    SerializationFailed(String),

    #[error("storage error: {0}")]
    StorageError(String),

    #[error("embedding failed: {0}")]
    EmbeddingFailed(String),

    #[error("internal error: {0}")]
    InternalError(String),
}

#[derive(serde::Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: u16,
}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        let (status, error_message, reflex_status) = match &self {
            GatewayError::InvalidRequest(_) => {
                (StatusCode::BAD_REQUEST, self.to_string(), "invalid_request")
            }
            GatewayError::CacheLookupFailed(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                self.to_string(),
                "lookup_error",
            ),
            GatewayError::ScoringFailed(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                self.to_string(),
                "scoring_error",
            ),
            GatewayError::ProviderError(_) => {
                (StatusCode::BAD_GATEWAY, self.to_string(), "provider_error")
            }
            GatewayError::SerializationFailed(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                self.to_string(),
                "serialization_error",
            ),
            GatewayError::StorageError(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                self.to_string(),
                "storage_error",
            ),
            GatewayError::EmbeddingFailed(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                self.to_string(),
                "embedding_error",
            ),
            GatewayError::InternalError(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                self.to_string(),
                "internal_error",
            ),
        };

        let mut headers = HeaderMap::new();
        headers.insert(
            REFLEX_STATUS_HEADER,
            HeaderValue::from_str(reflex_status).unwrap_or(HeaderValue::from_static("error")),
        );

        let body = Json(ErrorResponse {
            error: error_message,
            code: status.as_u16(),
        });

        (status, headers, body).into_response()
    }
}

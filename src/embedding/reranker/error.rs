use std::path::PathBuf;
use thiserror::Error;

use crate::embedding::error::EmbeddingError;

#[derive(Debug, Error)]
pub enum RerankerError {
    #[error("reranker model not found at path: {path}")]
    ModelNotFound { path: PathBuf },

    #[error("failed to load reranker model: {reason}")]
    ModelLoadFailed { reason: String },

    #[error("{device} device unavailable: {reason}")]
    DeviceUnavailable { device: String, reason: String },

    #[error("reranker inference failed: {reason}")]
    InferenceFailed { reason: String },

    #[error("tokenization failed: {reason}")]
    TokenizationFailed { reason: String },

    #[error("invalid reranker configuration: {reason}")]
    InvalidConfig { reason: String },

    #[error("reranker not available: {reason}")]
    NotAvailable { reason: String },
}

impl From<candle_core::Error> for RerankerError {
    fn from(err: candle_core::Error) -> Self {
        RerankerError::InferenceFailed {
            reason: err.to_string(),
        }
    }
}

impl From<std::io::Error> for RerankerError {
    fn from(err: std::io::Error) -> Self {
        RerankerError::ModelLoadFailed {
            reason: err.to_string(),
        }
    }
}

impl From<EmbeddingError> for RerankerError {
    fn from(err: EmbeddingError) -> Self {
        match err {
            EmbeddingError::DeviceUnavailable { device, reason } => {
                RerankerError::DeviceUnavailable { device, reason }
            }
            _ => RerankerError::InferenceFailed {
                reason: err.to_string(),
            },
        }
    }
}

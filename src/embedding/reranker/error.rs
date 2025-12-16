use std::path::PathBuf;
use thiserror::Error;

use crate::embedding::error::EmbeddingError;

#[derive(Debug, Error)]
/// Errors returned by reranker load/scoring.
pub enum RerankerError {
    /// Model files were not found.
    #[error("reranker model not found at path: {path}")]
    ModelNotFound {
        /// Missing model path.
        path: PathBuf,
    },

    /// Model load failed.
    #[error("failed to load reranker model: {reason}")]
    ModelLoadFailed {
        /// Error message.
        reason: String,
    },

    /// Requested compute device is unavailable.
    #[error("{device} device unavailable: {reason}")]
    DeviceUnavailable {
        /// Device name (e.g. "cuda", "metal").
        device: String,
        /// Error message.
        reason: String,
    },

    /// Inference failed.
    #[error("reranker inference failed: {reason}")]
    InferenceFailed {
        /// Error message.
        reason: String,
    },

    /// Tokenization failed.
    #[error("tokenization failed: {reason}")]
    TokenizationFailed {
        /// Error message.
        reason: String,
    },

    /// Configuration is invalid.
    #[error("invalid reranker configuration: {reason}")]
    InvalidConfig {
        /// Error message.
        reason: String,
    },

    /// Operation requires a model but none is configured/loaded.
    #[error("reranker not available: {reason}")]
    NotAvailable {
        /// Error message.
        reason: String,
    },
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

use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
/// Errors returned by embedding generation and model loading.
pub enum EmbeddingError {
    /// Model files were not found.
    #[error("embedding model not found at path: {path}")]
    ModelNotFound {
        /// Missing model path.
        path: PathBuf,
    },

    /// Model load failed.
    #[error("failed to load embedding model: {reason}")]
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
    #[error("embedding inference failed: {reason}")]
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
    #[error("invalid model configuration: {reason}")]
    InvalidConfig {
        /// Error message.
        reason: String,
    },
}

impl From<candle_core::Error> for EmbeddingError {
    fn from(err: candle_core::Error) -> Self {
        EmbeddingError::InferenceFailed {
            reason: err.to_string(),
        }
    }
}

impl From<std::io::Error> for EmbeddingError {
    fn from(err: std::io::Error) -> Self {
        EmbeddingError::ModelLoadFailed {
            reason: err.to_string(),
        }
    }
}

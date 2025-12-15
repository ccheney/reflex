use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("embedding model not found at path: {path}")]
    ModelNotFound { path: PathBuf },

    #[error("failed to load embedding model: {reason}")]
    ModelLoadFailed { reason: String },

    #[error("{device} device unavailable: {reason}")]
    DeviceUnavailable { device: String, reason: String },

    #[error("embedding inference failed: {reason}")]
    InferenceFailed { reason: String },

    #[error("tokenization failed: {reason}")]
    TokenizationFailed { reason: String },

    #[error("invalid model configuration: {reason}")]
    InvalidConfig { reason: String },
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

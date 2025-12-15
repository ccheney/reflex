use crate::vectordb::VectorDbError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum L2CacheError {
    #[error("embedding generation failed: {reason}")]
    EmbeddingFailed { reason: String },

    #[error("vector database error: {0}")]
    VectorDb(#[from] VectorDbError),

    #[error("rescoring failed: {reason}")]
    RescoringFailed { reason: String },

    #[error("configuration error: {reason}")]
    ConfigError { reason: String },

    #[error("no candidates found for query")]
    NoCandidates,
}

pub type L2CacheResult<T> = Result<T, L2CacheError>;

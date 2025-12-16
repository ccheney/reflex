use crate::vectordb::VectorDbError;
use thiserror::Error;

#[derive(Debug, Error)]
/// Errors returned by the L2 semantic cache.
pub enum L2CacheError {
    /// Embedding generation failed.
    #[error("embedding generation failed: {reason}")]
    EmbeddingFailed {
        /// Error message.
        reason: String,
    },

    /// Vector database error (search/upsert/etc).
    #[error("vector database error: {0}")]
    VectorDb(#[from] VectorDbError),

    /// Full-precision rescoring failed.
    #[error("rescoring failed: {reason}")]
    RescoringFailed {
        /// Error message.
        reason: String,
    },

    /// Invalid configuration.
    #[error("configuration error: {reason}")]
    ConfigError {
        /// Error message.
        reason: String,
    },

    /// No candidates were returned (BQ stage empty or all candidates filtered).
    #[error("no candidates found for query")]
    NoCandidates,
}

/// Convenience result type for L2 operations.
pub type L2CacheResult<T> = Result<T, L2CacheError>;

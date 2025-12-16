use thiserror::Error;

use crate::embedding::RerankerError;

#[derive(Debug, Error)]
/// Errors returned by L3 scoring/verification.
pub enum ScoringError {
    /// Reranker load/inference/tokenization error.
    #[error("reranker error: {0}")]
    Reranker(#[from] RerankerError),

    /// Invalid input.
    #[error("invalid input: {reason}")]
    InvalidInput {
        /// Error message.
        reason: String,
    },

    /// Internal computation error.
    #[error("scoring computation failed: {reason}")]
    ComputationFailed {
        /// Error message.
        reason: String,
    },
}

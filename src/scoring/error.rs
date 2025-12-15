use thiserror::Error;

use crate::embedding::RerankerError;

#[derive(Debug, Error)]
pub enum ScoringError {
    #[error("reranker error: {0}")]
    Reranker(#[from] RerankerError),

    #[error("invalid input: {reason}")]
    InvalidInput { reason: String },

    #[error("scoring computation failed: {reason}")]
    ComputationFailed { reason: String },
}

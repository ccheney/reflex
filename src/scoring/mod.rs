//! L3 verification via cross-encoder reranking.
//!
//! Given a query and candidate `CacheEntry`s, score them with [`Reranker`](crate::embedding::Reranker)
//! and decide whether the best candidate is above the configured threshold.
//!
//! Note: [`CrossEncoderScorer`] treats `CacheEntry::payload_blob` as UTF-8 candidate text.

/// Scoring/verification errors.
pub mod error;
/// Cross-encoder scorer.
pub mod scorer;
/// Scoring result types.
pub mod types;

#[cfg(test)]
mod tests;

pub use error::ScoringError;
pub use scorer::CrossEncoderScorer;
pub use types::{VerificationResult, VerifiedCandidate};

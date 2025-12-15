//! L3 verification via cross-encoder reranking.
//!
//! Given a query and candidate `CacheEntry`s, score them with [`Reranker`](crate::embedding::Reranker)
//! and decide whether the best candidate is above the configured threshold.
//!
//! # Payload/Text Coupling
//!
//! **Important architectural note:** The [`CrossEncoderScorer`] uses `CacheEntry::payload_blob`
//! as the candidate text for cross-encoder comparison. This creates a coupling between the
//! scorer and how the gateway prepares candidates:
//!
//! - For L2 semantic hits, the gateway temporarily replaces `payload_blob` with the
//!   semantic request text (see `handler.rs` L2 hit handling)
//! - The scorer interprets `payload_blob` as UTF-8 text via `String::from_utf8_lossy`
//! - After scoring, the original `payload_blob` (containing the full cache payload)
//!   is restored for response generation
//!
//! This design allows the same `CacheEntry` type to be used throughout the pipeline
//! while ensuring the cross-encoder compares the correct semantic content. If you
//! modify either the gateway's candidate preparation or the scorer's text extraction,
//! ensure they remain coordinated.

pub mod error;
pub mod scorer;
pub mod types;

#[cfg(test)]
mod tests;

pub use error::ScoringError;
pub use scorer::CrossEncoderScorer;
pub use types::{VerificationResult, VerifiedCandidate};

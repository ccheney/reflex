//! Qdrant vector database integration.
//!
//! - [`QdrantClient`] is the direct client.
//! - [`bq`] adds binary-quantization utilities and a BQ-optimized client.

/// Binary quantization helpers and BQ client.
pub mod bq;
/// Direct Qdrant client and trait.
pub mod client;
/// Vector DB error types.
pub mod error;
#[cfg(any(test, feature = "mock"))]
/// In-memory vector DB mock (enabled with `mock` feature).
pub mod mock;
/// Shared vector DB model types.
pub mod model;
/// Full-precision rescoring utilities.
pub mod rescoring;

#[cfg(test)]
mod tests;

pub use client::{QdrantClient, VectorDbClient};
pub use error::VectorDbError;
#[cfg(any(test, feature = "mock"))]
pub use mock::{MockVectorDbClient, cosine_similarity};
pub use model::{
    SearchResult, VectorPoint, embedding_bytes_to_f32, f32_to_embedding_bytes, generate_point_id,
};

/// Default non-BQ collection name.
pub const DEFAULT_COLLECTION_NAME: &str = "reflex_cache";

/// Default vector size for non-BQ collections.
pub const DEFAULT_VECTOR_SIZE: u64 = crate::constants::DEFAULT_VECTOR_SIZE_U64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Write-consistency mode for Qdrant upserts/deletes.
pub enum WriteConsistency {
    /// Wait for the operation to be fully indexed and searchable.
    /// Slow, but ensures read-after-write consistency.
    /// Maps to `wait=true`.
    Strong,
    /// Return immediately after the server acknowledges receipt.
    /// Fast, but data may not be searchable immediately.
    /// Maps to `wait=false`.
    Eventual,
}

impl From<WriteConsistency> for bool {
    fn from(c: WriteConsistency) -> bool {
        matches!(c, WriteConsistency::Strong)
    }
}

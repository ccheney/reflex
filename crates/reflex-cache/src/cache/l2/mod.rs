//! L2 semantic cache (embedding + vector search + rescoring).
//!
//! The default implementation embeds with [`SinterEmbedder`](crate::embedding::sinter::SinterEmbedder),
//! searches a binary-quantized backend, then rescoring top candidates in full precision.

/// Backend trait used by L2 for vector search/upsert.
pub mod backend;
/// Core L2 cache implementation.
pub mod cache;
/// L2 configuration.
pub mod config;
/// L2 error types.
pub mod error;
/// Storage loader traits and implementations.
pub mod loader;
#[cfg(any(test, feature = "mock"))]
/// Mock L2 cache helpers (enabled with `mock` feature).
pub mod mock;
/// L2 result types.
pub mod types;

#[cfg(test)]
mod tests;

pub use backend::BqSearchBackend;
pub use cache::{L2SemanticCache, L2SemanticCacheHandle};
pub use config::{
    DEFAULT_TOP_K_BQ, DEFAULT_TOP_K_FINAL, L2_COLLECTION_NAME, L2_VECTOR_SIZE, L2Config,
};
pub use error::{L2CacheError, L2CacheResult};
#[cfg(any(test, feature = "mock"))]
pub use loader::MockStorageLoader;
pub use loader::{NvmeStorageLoader, StorageLoader};
#[cfg(any(test, feature = "mock"))]
pub use mock::MockL2SemanticCache;
pub use types::L2LookupResult;

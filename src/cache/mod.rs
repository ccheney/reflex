//! Tiered caching infrastructure.
//!
//! Reflex uses:
//! - **L1**: exact-match lookup (in-memory)
//! - **L2**: semantic search (vector DB)
//!
//! Start at [`TieredCache`] and [`TieredLookupResult`].

/// L1 exact-match cache.
pub mod l1;
/// L2 semantic cache.
pub mod l2;
/// L1+L2 tiered cache wrapper.
pub mod tiered;
/// Status/header types shared across the cache pipeline.
pub mod types;

#[cfg(test)]
mod l1_tests;
#[cfg(test)]
mod tiered_tests;

pub use l1::{L1Cache, L1CacheHandle, L1LookupResult};
pub use l2::{
    BqSearchBackend, DEFAULT_TOP_K_BQ, DEFAULT_TOP_K_FINAL, L2_COLLECTION_NAME, L2_VECTOR_SIZE,
    L2CacheError, L2CacheResult, L2Config, L2LookupResult, L2SemanticCache, L2SemanticCacheHandle,
    NvmeStorageLoader, StorageLoader,
};
#[cfg(any(test, feature = "mock"))]
pub use l2::{MockL2SemanticCache, MockStorageLoader};

#[cfg(any(test, feature = "mock"))]
pub use tiered::MockTieredCache;
pub use tiered::{TieredCache, TieredCacheHandle, TieredLookupResult};

pub use types::{
    REFLEX_STATUS_ERROR, REFLEX_STATUS_HEADER, REFLEX_STATUS_HEALTHY, REFLEX_STATUS_NOT_READY,
    REFLEX_STATUS_READY, REFLEX_STATUS_STORED, ReflexStatus,
};

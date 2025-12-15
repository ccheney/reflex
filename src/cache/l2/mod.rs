pub mod backend;
pub mod cache;
pub mod config;
pub mod error;
pub mod loader;
#[cfg(any(test, feature = "mock"))]
pub mod mock;
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

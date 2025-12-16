//! Mock L2 cache helpers (in-memory storage + mock vector DB).

use crate::embedding::sinter::{SinterConfig, SinterEmbedder};
use crate::vectordb::bq::MockBqClient;

use super::cache::L2SemanticCache;
use super::config::L2Config;
use super::error::{L2CacheError, L2CacheResult};
use super::loader::MockStorageLoader;

/// Type alias for an L2 cache backed by mocks.
pub type MockL2SemanticCache = L2SemanticCache<MockBqClient, MockStorageLoader>;

impl L2SemanticCache<MockBqClient, MockStorageLoader> {
    /// Creates a mock L2 cache and ensures the collection exists.
    pub async fn new_mock(config: L2Config) -> L2CacheResult<Self> {
        let embedder_config = SinterConfig::stub();
        let embedder =
            SinterEmbedder::load(embedder_config).map_err(|e| L2CacheError::EmbeddingFailed {
                reason: e.to_string(),
            })?;

        let bq_backend = MockBqClient::with_config(config.bq_config.clone());
        let storage = MockStorageLoader::new();

        let cache = Self::new(embedder, bq_backend, storage, config)?;

        cache.ensure_collection().await?;

        Ok(cache)
    }
}

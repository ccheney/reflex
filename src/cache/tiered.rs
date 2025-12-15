//! Tiered cache: L1 exact + L2 semantic.

use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::{debug, info, instrument};

#[cfg(any(test, feature = "mock"))]
use super::l2::L2Config;
#[cfg(any(test, feature = "mock"))]
use super::l2::MockStorageLoader;
use super::l2::{
    BqSearchBackend, L2CacheError, L2CacheResult, L2LookupResult, L2SemanticCache, StorageLoader,
};

use super::{L1CacheHandle, L1LookupResult, ReflexStatus};
use crate::storage::mmap::MmapFileHandle;
#[cfg(any(test, feature = "mock"))]
use crate::vectordb::bq::MockBqClient;

#[derive(Debug)]
pub enum TieredLookupResult {
    HitL1(L1LookupResult),
    HitL2(L2LookupResult),
    Miss,
}

impl TieredLookupResult {
    pub fn status(&self) -> ReflexStatus {
        match self {
            TieredLookupResult::HitL1(result) => result.status(),
            TieredLookupResult::HitL2(_) => ReflexStatus::HitL2Semantic,
            TieredLookupResult::Miss => ReflexStatus::Miss,
        }
    }

    pub fn is_hit(&self) -> bool {
        !matches!(self, TieredLookupResult::Miss)
    }

    pub fn is_l1_hit(&self) -> bool {
        matches!(self, TieredLookupResult::HitL1(_))
    }

    pub fn is_l2_hit(&self) -> bool {
        matches!(self, TieredLookupResult::HitL2(_))
    }
}

pub struct TieredCache<B: BqSearchBackend, S: StorageLoader> {
    l1: L1CacheHandle,
    l2: L2SemanticCache<B, S>,
}

impl<B: BqSearchBackend, S: StorageLoader> std::fmt::Debug for TieredCache<B, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TieredCache")
            .field("l1", &self.l1)
            .field("l2", &self.l2)
            .finish()
    }
}

impl<B: BqSearchBackend, S: StorageLoader> TieredCache<B, S> {
    pub fn new(l1: L1CacheHandle, l2: L2SemanticCache<B, S>) -> Self {
        Self { l1, l2 }
    }

    pub fn l1(&self) -> &L1CacheHandle {
        &self.l1
    }

    pub fn l2(&self) -> &L2SemanticCache<B, S> {
        &self.l2
    }

    #[instrument(skip(self, prompt), fields(prompt_len = prompt.len(), tenant_id = tenant_id))]
    pub async fn lookup(&self, prompt: &str, tenant_id: u64) -> L2CacheResult<TieredLookupResult> {
        self.lookup_with_semantic_query(prompt, prompt, tenant_id)
            .await
    }

    #[instrument(skip(self, exact_key, semantic_query), fields(key_len = exact_key.len(), query_len = semantic_query.len(), tenant_id = tenant_id))]
    pub async fn lookup_with_semantic_query(
        &self,
        exact_key: &str,
        semantic_query: &str,
        tenant_id: u64,
    ) -> L2CacheResult<TieredLookupResult> {
        debug!("Checking L1 cache");
        let l1_key = format!("{}:{}", tenant_id, exact_key);
        if let Some(result) = self.l1.lookup(&l1_key) {
            info!("L1 cache hit");
            return Ok(TieredLookupResult::HitL1(result));
        }

        debug!("L1 miss, checking L2 cache");

        match self.l2.search(semantic_query, tenant_id).await {
            Ok(result) => {
                if result.has_candidates() {
                    info!(
                        candidates = result.candidates().len(),
                        best_score = result.best_candidate().map(|c| c.score),
                        "L2 cache hit"
                    );
                    Ok(TieredLookupResult::HitL2(result))
                } else {
                    debug!("L2 returned no candidates");
                    Ok(TieredLookupResult::Miss)
                }
            }
            Err(L2CacheError::NoCandidates) => {
                debug!("L2 cache miss");
                Ok(TieredLookupResult::Miss)
            }
            Err(e) => Err(e),
        }
    }

    pub fn insert_l1(&self, prompt: &str, tenant_id: u64, handle: MmapFileHandle) -> [u8; 32] {
        let l1_key = format!("{}:{}", tenant_id, prompt);
        self.l1.insert(&l1_key, handle)
    }

    pub async fn index_l2(
        &self,
        prompt: &str,
        tenant_id: u64,
        context_hash: u64,
        storage_key: &str,
        timestamp: i64,
    ) -> L2CacheResult<u64> {
        self.l2
            .index(prompt, tenant_id, context_hash, storage_key, timestamp)
            .await
    }

    pub async fn insert_both(
        &self,
        prompt: &str,
        tenant_id: u64,
        context_hash: u64,
        storage_key: &str,
        timestamp: i64,
        handle: MmapFileHandle,
    ) -> L2CacheResult<([u8; 32], u64)> {
        let l1_hash = self.insert_l1(prompt, tenant_id, handle);

        let l2_point_id = self
            .index_l2(prompt, tenant_id, context_hash, storage_key, timestamp)
            .await?;

        Ok((l1_hash, l2_point_id))
    }

    pub fn remove_l1(&self, prompt: &str, tenant_id: u64) -> Option<MmapFileHandle> {
        let l1_key = format!("{}:{}", tenant_id, prompt);
        self.l1.remove_prompt(&l1_key)
    }

    pub fn contains_l1(&self, prompt: &str, tenant_id: u64) -> bool {
        let l1_key = format!("{}:{}", tenant_id, prompt);
        self.l1.contains_prompt(&l1_key)
    }

    pub fn l1_len(&self) -> usize {
        self.l1.len()
    }

    pub fn l1_is_empty(&self) -> bool {
        self.l1.is_empty()
    }

    pub fn clear_l1(&self) {
        self.l1.clear();
    }

    pub fn run_pending_tasks_l1(&self) {
        self.l1.run_pending_tasks();
    }

    pub async fn is_ready(&self) -> bool {
        self.l2.is_ready().await
    }
}

#[cfg(any(test, feature = "mock"))]
pub type MockTieredCache = TieredCache<MockBqClient, MockStorageLoader>;

#[cfg(any(test, feature = "mock"))]
impl TieredCache<MockBqClient, MockStorageLoader> {
    pub async fn new_mock() -> L2CacheResult<Self> {
        let l1 = L1CacheHandle::new();
        let l2 = L2SemanticCache::new_mock(L2Config::default()).await?;
        Ok(Self::new(l1, l2))
    }

    pub async fn new_mock_with_config(l2_config: L2Config) -> L2CacheResult<Self> {
        let l1 = L1CacheHandle::new();
        let l2 = L2SemanticCache::new_mock(l2_config).await?;
        Ok(Self::new(l1, l2))
    }

    pub fn mock_storage(&self) -> &MockStorageLoader {
        self.l2.storage()
    }

    pub fn mock_bq_backend(&self) -> &MockBqClient {
        self.l2.bq_backend()
    }
}

#[derive(Clone)]
pub struct TieredCacheHandle<B: BqSearchBackend, S: StorageLoader> {
    inner: Arc<RwLock<TieredCache<B, S>>>,
}

impl<B: BqSearchBackend, S: StorageLoader> TieredCacheHandle<B, S> {
    pub fn new(cache: TieredCache<B, S>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(cache)),
        }
    }

    pub async fn lookup(&self, prompt: &str, tenant_id: u64) -> L2CacheResult<TieredLookupResult> {
        self.inner.read().await.lookup(prompt, tenant_id).await
    }

    pub async fn lookup_with_semantic_query(
        &self,
        exact_key: &str,
        semantic_query: &str,
        tenant_id: u64,
    ) -> L2CacheResult<TieredLookupResult> {
        self.inner
            .read()
            .await
            .lookup_with_semantic_query(exact_key, semantic_query, tenant_id)
            .await
    }

    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }
}

impl<B: BqSearchBackend, S: StorageLoader> std::fmt::Debug for TieredCacheHandle<B, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TieredCacheHandle")
            .field("strong_count", &self.strong_count())
            .finish()
    }
}

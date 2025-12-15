use std::sync::Arc;

use futures_util::future::join_all;
use tokio::sync::RwLock;
use tracing::{debug, info, instrument, warn};

use crate::embedding::sinter::SinterEmbedder;
use crate::vectordb::VectorPoint;
use crate::vectordb::rescoring::{CandidateEntry, RescorerConfig, VectorRescorer};

use super::backend::BqSearchBackend;
use super::config::L2Config;
use super::error::{L2CacheError, L2CacheResult};
use super::loader::StorageLoader;
use super::types::L2LookupResult;

pub struct L2SemanticCache<B: BqSearchBackend, S: StorageLoader> {
    embedder: SinterEmbedder,
    bq_backend: B,
    storage: S,
    rescorer: VectorRescorer,
    config: L2Config,
}

impl<B: BqSearchBackend, S: StorageLoader> std::fmt::Debug for L2SemanticCache<B, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("L2SemanticCache")
            .field("embedder", &self.embedder)
            .field("rescorer", &self.rescorer)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<B: BqSearchBackend, S: StorageLoader> L2SemanticCache<B, S> {
    pub fn new(
        embedder: SinterEmbedder,
        bq_backend: B,
        storage: S,
        config: L2Config,
    ) -> L2CacheResult<Self> {
        config.validate()?;

        let rescorer_config = RescorerConfig {
            top_k: config.top_k_final,
            validate_dimensions: config.validate_dimensions,
        };

        Ok(Self {
            embedder,
            bq_backend,
            storage,
            rescorer: VectorRescorer::with_config(rescorer_config),
            config,
        })
    }

    pub fn config(&self) -> &L2Config {
        &self.config
    }

    pub fn embedder(&self) -> &SinterEmbedder {
        &self.embedder
    }

    pub fn is_embedder_stub(&self) -> bool {
        self.embedder.is_stub()
    }

    pub fn storage(&self) -> &S {
        &self.storage
    }

    pub fn bq_backend(&self) -> &B {
        &self.bq_backend
    }

    #[instrument(skip(self, prompt), fields(tenant_id = tenant_id, prompt_len = prompt.len()))]
    pub async fn search(&self, prompt: &str, tenant_id: u64) -> L2CacheResult<L2LookupResult> {
        debug!("Generating embedding for prompt");
        let embedding_f16 =
            self.embedder
                .embed(prompt)
                .map_err(|e| L2CacheError::EmbeddingFailed {
                    reason: e.to_string(),
                })?;

        let embedding_f32: Vec<f32> = embedding_f16.iter().map(|v| v.to_f32()).collect();

        debug!(
            embedding_dim = embedding_f32.len(),
            "Embedding generated, starting BQ search"
        );

        let bq_results = self
            .bq_backend
            .search_bq(
                &self.config.collection_name,
                embedding_f32,
                self.config.top_k_bq,
                Some(tenant_id),
            )
            .await?;

        let bq_candidates_count = bq_results.len();
        debug!(
            candidates = bq_candidates_count,
            "BQ search complete, loading storage entries"
        );

        if bq_results.is_empty() {
            return Err(L2CacheError::NoCandidates);
        }

        // Parallel storage loading: spawn all loads concurrently to reduce latency
        // from O(n) sequential to O(1) bounded by the slowest single load
        let load_futures: Vec<_> = bq_results
            .iter()
            .filter_map(|result| {
                result.storage_key.as_ref().map(|key| {
                    let key = key.clone();
                    let id = result.id;
                    let score = result.score;
                    async move {
                        let entry = self.storage.load(&key, tenant_id).await;
                        (id, score, key, entry)
                    }
                })
            })
            .collect();

        let load_results = join_all(load_futures).await;

        let mut candidate_entries = Vec::with_capacity(load_results.len());
        for (id, score, storage_key, entry) in load_results {
            if let Some(entry) = entry {
                candidate_entries.push(CandidateEntry::with_bq_score(id, entry, score));
            } else {
                warn!(
                    storage_key = storage_key,
                    "Storage entry not found or tenant mismatch, skipping candidate"
                );
            }
        }

        if candidate_entries.is_empty() {
            return Err(L2CacheError::NoCandidates);
        }

        debug!(
            loaded = candidate_entries.len(),
            "Storage entries loaded, starting rescore"
        );

        let scored_candidates = self
            .rescorer
            .rescore(&embedding_f16, candidate_entries)
            .map_err(|e| L2CacheError::RescoringFailed {
                reason: e.to_string(),
            })?;

        info!(
            tenant_id = tenant_id,
            bq_candidates = bq_candidates_count,
            rescored = scored_candidates.len(),
            best_score = scored_candidates.first().map(|c| c.score),
            "L2 search complete"
        );

        Ok(L2LookupResult::new(
            embedding_f16,
            scored_candidates,
            tenant_id,
            bq_candidates_count,
        ))
    }

    #[instrument(skip(self, prompt, storage_key), fields(tenant_id = tenant_id, context_hash = context_hash))]
    pub async fn index(
        &self,
        prompt: &str,
        tenant_id: u64,
        context_hash: u64,
        storage_key: &str,
        timestamp: i64,
    ) -> L2CacheResult<u64> {
        let embedding_f16 =
            self.embedder
                .embed(prompt)
                .map_err(|e| L2CacheError::EmbeddingFailed {
                    reason: e.to_string(),
                })?;

        let embedding_f32: Vec<f32> = embedding_f16.iter().map(|v| v.to_f32()).collect();

        let point_id = crate::vectordb::generate_point_id(tenant_id, context_hash);

        let point = VectorPoint {
            id: point_id,
            vector: embedding_f32,
            tenant_id,
            context_hash,
            timestamp,
            storage_key: Some(storage_key.to_string()),
        };

        self.bq_backend
            .upsert_points(
                &self.config.collection_name,
                vec![point],
                crate::vectordb::WriteConsistency::Eventual,
            )
            .await?;

        debug!(point_id = point_id, "Entry indexed in L2 cache");

        Ok(point_id)
    }

    pub async fn ensure_collection(&self) -> L2CacheResult<()> {
        self.bq_backend
            .ensure_collection(&self.config.collection_name, self.config.vector_size)
            .await?;
        Ok(())
    }

    pub async fn is_ready(&self) -> bool {
        self.bq_backend.is_ready().await
    }
}

#[derive(Clone)]
pub struct L2SemanticCacheHandle<B: BqSearchBackend, S: StorageLoader> {
    inner: Arc<RwLock<L2SemanticCache<B, S>>>,
}

impl<B: BqSearchBackend, S: StorageLoader> L2SemanticCacheHandle<B, S> {
    pub fn new(cache: L2SemanticCache<B, S>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(cache)),
        }
    }

    pub async fn search(&self, prompt: &str, tenant_id: u64) -> L2CacheResult<L2LookupResult> {
        self.inner.read().await.search(prompt, tenant_id).await
    }

    pub async fn index(
        &self,
        prompt: &str,
        tenant_id: u64,
        context_hash: u64,
        storage_key: &str,
        timestamp: i64,
    ) -> L2CacheResult<u64> {
        self.inner
            .read()
            .await
            .index(prompt, tenant_id, context_hash, storage_key, timestamp)
            .await
    }

    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }
}

impl<B: BqSearchBackend, S: StorageLoader> std::fmt::Debug for L2SemanticCacheHandle<B, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("L2SemanticCacheHandle")
            .field("strong_count", &self.strong_count())
            .finish()
    }
}

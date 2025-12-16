use std::collections::HashMap;

use super::config::BqConfig;
use super::utils::{hamming_distance, quantize_to_binary};
use crate::vectordb::{
    SearchResult, VectorDbError, VectorPoint, WriteConsistency, cosine_similarity,
};

#[derive(Default, Clone)]
/// In-memory mock implementation of a binary-quantized vector DB client.
pub struct MockBqClient {
    collections: std::sync::Arc<std::sync::RwLock<HashMap<String, MockBqCollection>>>,
    config: BqConfig,
}

#[derive(Default, Clone)]
struct MockBqCollection {
    vector_size: u64,
    points: HashMap<u64, MockBqPoint>,
}

#[derive(Clone)]
struct MockBqPoint {
    vector: Vec<f32>,
    binary: Vec<u8>,
    tenant_id: u64,
    context_hash: u64,
    timestamp: i64,
    storage_key: Option<String>,
}

impl MockBqClient {
    /// Creates a default mock client.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a mock client with an explicit config.
    pub fn with_config(config: BqConfig) -> Self {
        Self {
            collections: std::sync::Arc::new(std::sync::RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Returns the number of points currently stored in `collection`.
    pub fn point_count(&self, collection: &str) -> Option<usize> {
        self.collections
            .read()
            .ok()?
            .get(collection)
            .map(|c| c.points.len())
    }

    /// Poisons the internal RwLock for testing error handling paths.
    #[cfg(any(test, feature = "mock"))]
    pub fn poison_lock(&self) {
        use std::thread;

        let collections_clone = self.collections.clone();
        let handle = thread::spawn(move || {
            let _guard = collections_clone.write().unwrap();
            panic!("Intentional panic to poison lock for testing");
        });
        // Wait for the thread to panic, which poisons the lock
        let _ = handle.join();
    }

    /// Ensures a BQ collection exists.
    pub async fn ensure_bq_collection(
        &self,
        name: &str,
        vector_size: u64,
    ) -> Result<(), VectorDbError> {
        let mut collections =
            self.collections
                .write()
                .map_err(|_| VectorDbError::CreateCollectionFailed {
                    collection: name.to_string(),
                    message: "lock poisoned".to_string(),
                })?;

        collections
            .entry(name.to_string())
            .or_insert(MockBqCollection {
                vector_size,
                points: HashMap::new(),
            });

        Ok(())
    }

    /// Upserts points into a mock collection.
    pub async fn upsert_points(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
        _consistency: WriteConsistency,
    ) -> Result<(), VectorDbError> {
        let mut collections =
            self.collections
                .write()
                .map_err(|_| VectorDbError::UpsertFailed {
                    collection: collection.to_string(),
                    message: "lock poisoned".to_string(),
                })?;

        let coll =
            collections
                .get_mut(collection)
                .ok_or_else(|| VectorDbError::CollectionNotFound {
                    collection: collection.to_string(),
                })?;

        for point in points {
            if point.vector.len() as u64 != coll.vector_size {
                return Err(VectorDbError::InvalidDimension {
                    expected: coll.vector_size as usize,
                    actual: point.vector.len(),
                });
            }

            let binary = quantize_to_binary(&point.vector);

            coll.points.insert(
                point.id,
                MockBqPoint {
                    vector: point.vector,
                    binary,
                    tenant_id: point.tenant_id,
                    context_hash: point.context_hash,
                    timestamp: point.timestamp,
                    storage_key: point.storage_key,
                },
            );
        }

        Ok(())
    }

    /// Searches the mock BQ collection.
    pub async fn search_bq(
        &self,
        collection: &str,
        query: Vec<f32>,
        limit: u64,
        tenant_filter: Option<u64>,
    ) -> Result<Vec<SearchResult>, VectorDbError> {
        let collections = self
            .collections
            .read()
            .map_err(|_| VectorDbError::SearchFailed {
                collection: collection.to_string(),
                message: "lock poisoned".to_string(),
            })?;

        let coll =
            collections
                .get(collection)
                .ok_or_else(|| VectorDbError::CollectionNotFound {
                    collection: collection.to_string(),
                })?;

        let query_binary = quantize_to_binary(&query);

        let mut candidates: Vec<(u64, &MockBqPoint, u32)> = coll
            .points
            .iter()
            .filter(|(_, p)| tenant_filter.is_none() || tenant_filter == Some(p.tenant_id))
            .map(|(&id, p)| {
                let hamming = hamming_distance(&query_binary, &p.binary);
                (id, p, hamming)
            })
            .collect();

        candidates.sort_by_key(|(_, _, h)| *h);

        let rescore_limit = if self.config.rescore {
            self.config.rescore_limit as usize
        } else {
            limit as usize
        };
        candidates.truncate(rescore_limit);

        let mut results: Vec<SearchResult> = if self.config.rescore {
            candidates
                .into_iter()
                .map(|(id, p, _)| {
                    let score = cosine_similarity(&query, &p.vector);
                    SearchResult {
                        id,
                        score,
                        tenant_id: p.tenant_id,
                        context_hash: p.context_hash,
                        timestamp: p.timestamp,
                        storage_key: p.storage_key.clone(),
                    }
                })
                .collect()
        } else {
            // Max Hamming distance is based on the binary representation's bit count,
            // not the original float vector length. query_binary has length (dim / 8) bytes,
            // so max bits = query_binary.len() * 8 = dim (the original dimension).
            let max_hamming = (query_binary.len() as u32) * 8;
            candidates
                .into_iter()
                .map(|(id, p, hamming)| {
                    let score = 1.0 - (hamming as f32 / max_hamming as f32);
                    SearchResult {
                        id,
                        score,
                        tenant_id: p.tenant_id,
                        context_hash: p.context_hash,
                        timestamp: p.timestamp,
                        storage_key: p.storage_key.clone(),
                    }
                })
                .collect()
        };

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(limit as usize);
        Ok(results)
    }

    /// Deletes points by id.
    pub async fn delete_points(
        &self,
        collection: &str,
        ids: Vec<u64>,
    ) -> Result<(), VectorDbError> {
        let mut collections =
            self.collections
                .write()
                .map_err(|_| VectorDbError::DeleteFailed {
                    collection: collection.to_string(),
                    message: "lock poisoned".to_string(),
                })?;

        let coll =
            collections
                .get_mut(collection)
                .ok_or_else(|| VectorDbError::CollectionNotFound {
                    collection: collection.to_string(),
                })?;

        for id in ids {
            coll.points.remove(&id);
        }

        Ok(())
    }
}

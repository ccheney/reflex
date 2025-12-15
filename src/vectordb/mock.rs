use crate::vectordb::{SearchResult, VectorDbClient, VectorDbError, VectorPoint, WriteConsistency};
use std::collections::HashMap;

#[derive(Default)]
pub struct MockVectorDbClient {
    collections: std::sync::RwLock<HashMap<String, MockCollection>>,
}

#[derive(Default, Clone)]
struct MockCollection {
    vector_size: u64,
    points: HashMap<u64, MockStoredPoint>,
}

#[derive(Clone)]
struct MockStoredPoint {
    vector: Vec<f32>,
    tenant_id: u64,
    context_hash: u64,
    timestamp: i64,
    storage_key: Option<String>,
}

impl MockVectorDbClient {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn point_count(&self, collection: &str) -> Option<usize> {
        self.collections
            .read()
            .ok()?
            .get(collection)
            .map(|c| c.points.len())
    }

    /// Poisons the internal RwLock for testing error handling paths.
    /// This method is only available in test builds.
    #[cfg(test)]
    pub fn poison_lock(&self) {
        use std::thread;

        let collections_ptr = &self.collections as *const _ as usize;
        let handle = thread::spawn(move || {
            // SAFETY: We're in test code, the pointer is valid for the duration
            let collections: &std::sync::RwLock<HashMap<String, MockCollection>> =
                unsafe { &*(collections_ptr as *const _) };
            let _guard = collections.write().unwrap();
            panic!("Intentional panic to poison lock for testing");
        });
        // Wait for the thread to panic, which poisons the lock
        let _ = handle.join();
    }
}

impl VectorDbClient for MockVectorDbClient {
    async fn ensure_collection(&self, name: &str, vector_size: u64) -> Result<(), VectorDbError> {
        let mut collections =
            self.collections
                .write()
                .map_err(|_| VectorDbError::CreateCollectionFailed {
                    collection: name.to_string(),
                    message: "lock poisoned".to_string(),
                })?;

        collections
            .entry(name.to_string())
            .or_insert(MockCollection {
                vector_size,
                points: HashMap::new(),
            });

        Ok(())
    }

    async fn upsert_points(
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

            coll.points.insert(
                point.id,
                MockStoredPoint {
                    vector: point.vector,
                    tenant_id: point.tenant_id,
                    context_hash: point.context_hash,
                    timestamp: point.timestamp,
                    storage_key: point.storage_key,
                },
            );
        }

        Ok(())
    }

    async fn search(
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

        let mut results: Vec<SearchResult> = coll
            .points
            .iter()
            .filter(|(_, p)| tenant_filter.is_none() || tenant_filter == Some(p.tenant_id))
            .map(|(&id, p)| {
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
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(limit as usize);
        Ok(results)
    }

    async fn delete_points(&self, collection: &str, ids: Vec<u64>) -> Result<(), VectorDbError> {
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

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

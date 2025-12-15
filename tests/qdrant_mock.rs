//! Tests with a mock Qdrant-like client.

mod common;

use common::fixtures::{
    CacheEntryBuilder, EMBEDDING_SIZE_BYTES, create_batch_entries, generate_deterministic_embedding,
};
use reflex::storage::CacheEntry;
use rkyv::rancor::Error;
use rkyv::to_bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MockPointId {
    Uuid(String),
    Num(u64),
}

impl MockPointId {
    pub fn from_tenant_context(tenant_id: u64, context_hash: u64) -> Self {
        MockPointId::Num(tenant_id.wrapping_mul(1_000_000_000) + context_hash % 1_000_000_000)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockVector {
    pub data: Vec<f32>,
}

impl MockVector {
    pub fn from_embedding_bytes(bytes: &[u8]) -> Self {
        let data: Vec<f32> = bytes
            .chunks(2)
            .map(|chunk| {
                if chunk.len() == 2 {
                    let val = u16::from_le_bytes([chunk[0], chunk[1]]);
                    val as f32 / 65535.0
                } else {
                    0.0
                }
            })
            .collect();
        MockVector { data }
    }

    pub fn dimension(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockPayload {
    pub tenant_id: u64,
    pub context_hash: u64,
    pub timestamp: i64,
    pub cache_entry_ref: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockPoint {
    pub id: MockPointId,
    pub vector: MockVector,
    pub payload: MockPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockScoredPoint {
    pub id: MockPointId,
    pub score: f32,
    pub payload: Option<MockPayload>,
    pub vector: Option<MockVector>,
}

#[derive(Default)]
pub struct MockQdrantClient {
    collections: HashMap<String, Vec<MockPoint>>,
}

impl MockQdrantClient {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_collection(&mut self, name: &str) {
        self.collections.entry(name.to_string()).or_default();
    }

    pub fn upsert_points(
        &mut self,
        collection: &str,
        points: Vec<MockPoint>,
    ) -> Result<(), MockQdrantError> {
        let coll = self
            .collections
            .get_mut(collection)
            .ok_or(MockQdrantError::CollectionNotFound(collection.to_string()))?;

        for point in points {
            coll.retain(|p| p.id != point.id);
            coll.push(point);
        }

        Ok(())
    }

    pub fn search(
        &self,
        collection: &str,
        query_vector: &MockVector,
        limit: usize,
        filter_tenant: Option<u64>,
    ) -> Result<Vec<MockScoredPoint>, MockQdrantError> {
        let coll = self
            .collections
            .get(collection)
            .ok_or(MockQdrantError::CollectionNotFound(collection.to_string()))?;

        let mut results: Vec<MockScoredPoint> = coll
            .iter()
            .filter(|p| filter_tenant.is_none() || filter_tenant == Some(p.payload.tenant_id))
            .map(|p| {
                let score = cosine_similarity(&query_vector.data, &p.vector.data);
                MockScoredPoint {
                    id: p.id.clone(),
                    score,
                    payload: Some(p.payload.clone()),
                    vector: Some(p.vector.clone()),
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);

        Ok(results)
    }

    pub fn get_point(
        &self,
        collection: &str,
        id: &MockPointId,
    ) -> Result<Option<MockPoint>, MockQdrantError> {
        let coll = self
            .collections
            .get(collection)
            .ok_or(MockQdrantError::CollectionNotFound(collection.to_string()))?;

        Ok(coll.iter().find(|p| &p.id == id).cloned())
    }

    pub fn delete_points(
        &mut self,
        collection: &str,
        ids: &[MockPointId],
    ) -> Result<usize, MockQdrantError> {
        let coll = self
            .collections
            .get_mut(collection)
            .ok_or(MockQdrantError::CollectionNotFound(collection.to_string()))?;

        let initial_len = coll.len();
        coll.retain(|p| !ids.contains(&p.id));
        Ok(initial_len - coll.len())
    }

    pub fn point_count(&self, collection: &str) -> Result<usize, MockQdrantError> {
        let coll = self
            .collections
            .get(collection)
            .ok_or(MockQdrantError::CollectionNotFound(collection.to_string()))?;
        Ok(coll.len())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

#[derive(Debug, thiserror::Error)]
pub enum MockQdrantError {
    #[error("Collection not found: {0}")]
    CollectionNotFound(String),
    #[error("Point not found")]
    PointNotFound,
    #[error("Invalid vector dimension")]
    InvalidDimension,
}

fn cache_entry_to_point(entry: &CacheEntry) -> MockPoint {
    let id = MockPointId::from_tenant_context(entry.tenant_id, entry.context_hash);
    let vector = MockVector::from_embedding_bytes(&entry.embedding);
    let payload = MockPayload {
        tenant_id: entry.tenant_id,
        context_hash: entry.context_hash,
        timestamp: entry.timestamp,
        cache_entry_ref: None,
    };

    MockPoint {
        id,
        vector,
        payload,
    }
}

#[test]
fn test_upsert_single_point() {
    let mut client = MockQdrantClient::new();
    client.create_collection("cache");

    let entry = CacheEntryBuilder::new()
        .tenant_id(100)
        .context_hash(200)
        .with_realistic_embedding()
        .build();

    let point = cache_entry_to_point(&entry);
    client
        .upsert_points("cache", vec![point])
        .expect("Upsert should succeed");

    assert_eq!(client.point_count("cache").unwrap(), 1);
}

#[test]
fn test_upsert_batch() {
    let mut client = MockQdrantClient::new();
    client.create_collection("cache");

    let entries = create_batch_entries(50);
    let points: Vec<_> = entries.iter().map(cache_entry_to_point).collect();

    client
        .upsert_points("cache", points)
        .expect("Batch upsert should succeed");

    assert_eq!(client.point_count("cache").unwrap(), 50);
}

#[test]
fn test_upsert_replaces_existing() {
    let mut client = MockQdrantClient::new();
    client.create_collection("cache");

    let entry1 = CacheEntryBuilder::new()
        .tenant_id(100)
        .context_hash(200)
        .timestamp(1000)
        .with_realistic_embedding()
        .build();

    let entry2 = CacheEntryBuilder::new()
        .tenant_id(100)
        .context_hash(200)
        .timestamp(2000)
        .with_realistic_embedding()
        .build();

    client
        .upsert_points("cache", vec![cache_entry_to_point(&entry1)])
        .unwrap();
    client
        .upsert_points("cache", vec![cache_entry_to_point(&entry2)])
        .unwrap();

    assert_eq!(client.point_count("cache").unwrap(), 1);

    let id = MockPointId::from_tenant_context(100, 200);
    let point = client.get_point("cache", &id).unwrap().unwrap();
    assert_eq!(point.payload.timestamp, 2000);
}

#[test]
fn test_search_returns_results() {
    let mut client = MockQdrantClient::new();
    client.create_collection("cache");

    let entries = create_batch_entries(10);
    let points: Vec<_> = entries.iter().map(cache_entry_to_point).collect();
    client.upsert_points("cache", points).unwrap();

    let query = MockVector::from_embedding_bytes(&generate_deterministic_embedding(0));
    let results = client.search("cache", &query, 5, None).unwrap();

    assert!(!results.is_empty());
    assert!(results.len() <= 5);
}

#[test]
fn test_search_results_sorted_by_score() {
    let mut client = MockQdrantClient::new();
    client.create_collection("cache");

    let entries = create_batch_entries(20);
    let points: Vec<_> = entries.iter().map(cache_entry_to_point).collect();
    client.upsert_points("cache", points).unwrap();

    let query = MockVector::from_embedding_bytes(&generate_deterministic_embedding(0));
    let results = client.search("cache", &query, 10, None).unwrap();

    for i in 1..results.len() {
        assert!(
            results[i - 1].score >= results[i].score,
            "Results should be sorted by score descending"
        );
    }
}

#[test]
fn test_search_with_tenant_filter() {
    let mut client = MockQdrantClient::new();
    client.create_collection("cache");

    let tenant_a_entries = common::fixtures::create_tenant_entries(1000, 5);
    let tenant_b_entries = common::fixtures::create_tenant_entries(2000, 5);

    let all_points: Vec<_> = tenant_a_entries
        .iter()
        .chain(tenant_b_entries.iter())
        .map(cache_entry_to_point)
        .collect();

    client.upsert_points("cache", all_points).unwrap();

    let query = MockVector::from_embedding_bytes(&generate_deterministic_embedding(0));
    let results = client.search("cache", &query, 10, Some(1000)).unwrap();

    for result in &results {
        assert_eq!(result.payload.as_ref().unwrap().tenant_id, 1000);
    }
}

#[test]
fn test_search_respects_limit() {
    let mut client = MockQdrantClient::new();
    client.create_collection("cache");

    let entries = create_batch_entries(100);
    let points: Vec<_> = entries.iter().map(cache_entry_to_point).collect();
    client.upsert_points("cache", points).unwrap();

    let query = MockVector::from_embedding_bytes(&generate_deterministic_embedding(0));

    let results = client.search("cache", &query, 5, None).unwrap();
    assert_eq!(results.len(), 5);

    let results = client.search("cache", &query, 1, None).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_delete_points() {
    let mut client = MockQdrantClient::new();
    client.create_collection("cache");

    let entries = create_batch_entries(10);
    let points: Vec<_> = entries.iter().map(cache_entry_to_point).collect();
    let ids: Vec<_> = points.iter().map(|p| p.id.clone()).collect();

    client.upsert_points("cache", points).unwrap();
    assert_eq!(client.point_count("cache").unwrap(), 10);

    let deleted = client.delete_points("cache", &ids[0..5]).unwrap();
    assert_eq!(deleted, 5);
    assert_eq!(client.point_count("cache").unwrap(), 5);
}

#[test]
fn test_delete_nonexistent_points() {
    let mut client = MockQdrantClient::new();
    client.create_collection("cache");

    let fake_id = MockPointId::Num(99999);
    let deleted = client.delete_points("cache", &[fake_id]).unwrap();

    assert_eq!(deleted, 0);
}

#[test]
fn test_error_collection_not_found() {
    let client = MockQdrantClient::new();

    let result = client.point_count("nonexistent");
    assert!(matches!(
        result,
        Err(MockQdrantError::CollectionNotFound(_))
    ));
}

#[test]
fn test_search_empty_collection() {
    let mut client = MockQdrantClient::new();
    client.create_collection("empty");

    let query = MockVector::from_embedding_bytes(&generate_deterministic_embedding(0));
    let results = client.search("empty", &query, 10, None).unwrap();

    assert!(results.is_empty());
}

#[test]
fn test_vector_dimension_from_embedding() {
    let embedding = generate_deterministic_embedding(42);
    let vector = MockVector::from_embedding_bytes(&embedding);

    assert_eq!(vector.dimension(), EMBEDDING_SIZE_BYTES / 2);
}

#[test]
fn test_different_vector_dimensions() {
    let small = MockVector::from_embedding_bytes(&[0u8; 100]);
    let large = MockVector::from_embedding_bytes(&[0u8; 10000]);

    assert_eq!(small.dimension(), 50);
    assert_eq!(large.dimension(), 5000);
}

#[tokio::test]
async fn test_concurrent_reads() {
    use std::sync::Arc;

    let mut client = MockQdrantClient::new();
    client.create_collection("cache");

    let entries = create_batch_entries(100);
    let points: Vec<_> = entries.iter().map(cache_entry_to_point).collect();
    client.upsert_points("cache", points).unwrap();

    let client = Arc::new(client);

    let handles: Vec<_> = (0..20)
        .map(|i| {
            let client = Arc::clone(&client);
            tokio::spawn(async move {
                let query = MockVector::from_embedding_bytes(&generate_deterministic_embedding(i));
                client.search("cache", &query, 5, None)
            })
        })
        .collect();

    for handle in handles {
        let result = handle.await.expect("Task should complete");
        assert!(result.is_ok());
    }
}

#[test]
fn test_full_storage_qdrant_workflow() {
    let mut client = MockQdrantClient::new();
    client.create_collection("reflex_cache");

    let entries = create_batch_entries(5);
    for entry in &entries {
        let point = cache_entry_to_point(entry);

        let _serialized = to_bytes::<Error>(entry).expect("Serialization should succeed");

        client.upsert_points("reflex_cache", vec![point]).unwrap();
    }

    let query_embedding = generate_deterministic_embedding(0);
    let query_vector = MockVector::from_embedding_bytes(&query_embedding);

    let results = client
        .search("reflex_cache", &query_vector, 3, None)
        .unwrap();

    assert!(!results.is_empty());
    for result in results {
        assert!(result.payload.is_some());
    }
}

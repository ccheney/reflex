use super::client::VectorDbClient;
use super::error::VectorDbError;
use super::mock::{MockVectorDbClient, cosine_similarity};
use super::model::{
    VectorPoint, embedding_bytes_to_f32, f32_to_embedding_bytes, generate_point_id,
};

const TEST_COLLECTION: &str = "test_collection";
const TEST_VECTOR_SIZE: u64 = crate::constants::DEFAULT_VECTOR_SIZE_U64;

fn create_test_vector(seed: u64) -> Vec<f32> {
    (0..TEST_VECTOR_SIZE)
        .map(|i| {
            let mixed = (seed.wrapping_mul(31).wrapping_add(i)) % 1000;
            mixed as f32 / 1000.0
        })
        .collect()
}

fn create_test_point(id: u64, tenant_id: u64) -> VectorPoint {
    VectorPoint {
        id,
        vector: create_test_vector(id),
        tenant_id,
        context_hash: id * 100,
        timestamp: 1702512000 + id as i64,
        storage_key: Some(format!("key_{}", id)),
    }
}

#[tokio::test]
async fn test_ensure_collection_creates_new() {
    let client = MockVectorDbClient::new();

    client
        .ensure_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .expect("should create collection");

    assert_eq!(client.point_count(TEST_COLLECTION), Some(0));
}

#[tokio::test]
async fn test_ensure_collection_idempotent() {
    let client = MockVectorDbClient::new();

    client
        .ensure_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();
    client
        .ensure_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    assert_eq!(client.point_count(TEST_COLLECTION), Some(0));
}

#[tokio::test]
async fn test_upsert_single_point() {
    let client = MockVectorDbClient::new();
    client
        .ensure_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    let point = create_test_point(1, 1000);
    client
        .upsert_points(
            TEST_COLLECTION,
            vec![point],
            crate::vectordb::WriteConsistency::Strong,
        )
        .await
        .expect("should upsert point");

    assert_eq!(client.point_count(TEST_COLLECTION), Some(1));
}

#[tokio::test]
async fn test_upsert_batch() {
    let client = MockVectorDbClient::new();
    client
        .ensure_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    let points: Vec<_> = (0..50).map(|i| create_test_point(i, 1000)).collect();
    client
        .upsert_points(
            TEST_COLLECTION,
            points,
            crate::vectordb::WriteConsistency::Strong,
        )
        .await
        .expect("should upsert batch");

    assert_eq!(client.point_count(TEST_COLLECTION), Some(50));
}

#[tokio::test]
async fn test_upsert_replaces_existing() {
    let client = MockVectorDbClient::new();
    client
        .ensure_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    let point1 = create_test_point(1, 1000);
    client
        .upsert_points(
            TEST_COLLECTION,
            vec![point1],
            crate::vectordb::WriteConsistency::Strong,
        )
        .await
        .unwrap();

    let point2 = VectorPoint {
        id: 1,
        vector: create_test_vector(999),
        tenant_id: 2000,
        context_hash: 999,
        timestamp: 9999,
        storage_key: Some("updated".to_string()),
    };
    client
        .upsert_points(
            TEST_COLLECTION,
            vec![point2],
            crate::vectordb::WriteConsistency::Strong,
        )
        .await
        .unwrap();

    assert_eq!(client.point_count(TEST_COLLECTION), Some(1));

    let results = client
        .search(TEST_COLLECTION, create_test_vector(999), 1, None)
        .await
        .unwrap();
    assert_eq!(results[0].tenant_id, 2000);
}

#[tokio::test]
async fn test_upsert_empty_batch() {
    let client = MockVectorDbClient::new();
    client
        .ensure_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    client
        .upsert_points(
            TEST_COLLECTION,
            vec![],
            crate::vectordb::WriteConsistency::Strong,
        )
        .await
        .expect("empty upsert should succeed");

    assert_eq!(client.point_count(TEST_COLLECTION), Some(0));
}

#[tokio::test]
async fn test_upsert_to_nonexistent_collection() {
    let client = MockVectorDbClient::new();

    let point = create_test_point(1, 1000);
    let result = client
        .upsert_points(
            "nonexistent",
            vec![point],
            crate::vectordb::WriteConsistency::Strong,
        )
        .await;

    assert!(matches!(
        result,
        Err(VectorDbError::CollectionNotFound { .. })
    ));
}

#[tokio::test]
async fn test_upsert_wrong_dimension() {
    let client = MockVectorDbClient::new();
    client
        .ensure_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    let wrong_dim_point = VectorPoint {
        id: 1,
        vector: vec![0.1; 100],
        tenant_id: 1000,
        context_hash: 2000,
        timestamp: 0,
        storage_key: None,
    };

    let result = client
        .upsert_points(
            TEST_COLLECTION,
            vec![wrong_dim_point],
            crate::vectordb::WriteConsistency::Strong,
        )
        .await;

    assert!(matches!(
        result,
        Err(VectorDbError::InvalidDimension { .. })
    ));
}

#[tokio::test]
async fn test_search_returns_results() {
    let client = MockVectorDbClient::new();
    client
        .ensure_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    let points: Vec<_> = (0..10).map(|i| create_test_point(i, 1000)).collect();
    client
        .upsert_points(
            TEST_COLLECTION,
            points,
            crate::vectordb::WriteConsistency::Strong,
        )
        .await
        .unwrap();

    let query = create_test_vector(0);
    let results = client
        .search(TEST_COLLECTION, query, 5, None)
        .await
        .unwrap();

    assert!(!results.is_empty());
    assert!(results.len() <= 5);
}

#[tokio::test]
async fn test_search_results_sorted_by_score() {
    let client = MockVectorDbClient::new();
    client
        .ensure_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    let points: Vec<_> = (0..20).map(|i| create_test_point(i, 1000)).collect();
    client
        .upsert_points(
            TEST_COLLECTION,
            points,
            crate::vectordb::WriteConsistency::Strong,
        )
        .await
        .unwrap();

    let query = create_test_vector(0);
    let results = client
        .search(TEST_COLLECTION, query, 10, None)
        .await
        .unwrap();

    for i in 1..results.len() {
        assert!(
            results[i - 1].score >= results[i].score,
            "Results should be sorted by score descending"
        );
    }
}

#[tokio::test]
async fn test_search_with_tenant_filter() {
    let client = MockVectorDbClient::new();
    client
        .ensure_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    let tenant_a_points: Vec<_> = (0..5).map(|i| create_test_point(i, 1000)).collect();
    let tenant_b_points: Vec<_> = (5..10).map(|i| create_test_point(i, 2000)).collect();

    client
        .upsert_points(
            TEST_COLLECTION,
            tenant_a_points,
            crate::vectordb::WriteConsistency::Strong,
        )
        .await
        .unwrap();
    client
        .upsert_points(
            TEST_COLLECTION,
            tenant_b_points,
            crate::vectordb::WriteConsistency::Strong,
        )
        .await
        .unwrap();

    let query = create_test_vector(0);
    let results = client
        .search(TEST_COLLECTION, query, 10, Some(1000))
        .await
        .unwrap();

    for result in &results {
        assert_eq!(result.tenant_id, 1000);
    }
}

#[tokio::test]
async fn test_search_respects_limit() {
    let client = MockVectorDbClient::new();
    client
        .ensure_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    let points: Vec<_> = (0..100).map(|i| create_test_point(i, 1000)).collect();
    client
        .upsert_points(
            TEST_COLLECTION,
            points,
            crate::vectordb::WriteConsistency::Strong,
        )
        .await
        .unwrap();

    let query = create_test_vector(0);

    let results = client
        .search(TEST_COLLECTION, query.clone(), 5, None)
        .await
        .unwrap();
    assert_eq!(results.len(), 5);

    let results = client
        .search(TEST_COLLECTION, query, 1, None)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn test_search_empty_collection() {
    let client = MockVectorDbClient::new();
    client
        .ensure_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    let query = create_test_vector(0);
    let results = client
        .search(TEST_COLLECTION, query, 10, None)
        .await
        .unwrap();

    assert!(results.is_empty());
}

#[tokio::test]
async fn test_search_nonexistent_collection() {
    let client = MockVectorDbClient::new();

    let query = create_test_vector(0);
    let result = client.search("nonexistent", query, 10, None).await;

    assert!(matches!(
        result,
        Err(VectorDbError::CollectionNotFound { .. })
    ));
}

#[tokio::test]
async fn test_delete_points() {
    let client = MockVectorDbClient::new();
    client
        .ensure_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    let points: Vec<_> = (0..10).map(|i| create_test_point(i, 1000)).collect();
    client
        .upsert_points(
            TEST_COLLECTION,
            points,
            crate::vectordb::WriteConsistency::Strong,
        )
        .await
        .unwrap();

    assert_eq!(client.point_count(TEST_COLLECTION), Some(10));

    client
        .delete_points(TEST_COLLECTION, vec![0, 1, 2, 3, 4])
        .await
        .unwrap();

    assert_eq!(client.point_count(TEST_COLLECTION), Some(5));
}

#[tokio::test]
async fn test_delete_nonexistent_points() {
    let client = MockVectorDbClient::new();
    client
        .ensure_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    client
        .delete_points(TEST_COLLECTION, vec![99999])
        .await
        .expect("should succeed");
}

#[tokio::test]
async fn test_delete_empty_list() {
    let client = MockVectorDbClient::new();
    client
        .ensure_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    client
        .delete_points(TEST_COLLECTION, vec![])
        .await
        .expect("empty delete should succeed");
}

#[test]
fn test_embedding_bytes_to_f32() {
    let mut original = vec![0.0f32; crate::constants::DEFAULT_EMBEDDING_DIM];
    original[0] = 1.0;
    original[1] = 0.5;
    original[2] = -0.25;

    let bytes = f32_to_embedding_bytes(&original);
    let f32_values = embedding_bytes_to_f32(&bytes).unwrap();

    assert_eq!(f32_values.len(), crate::constants::DEFAULT_EMBEDDING_DIM);
    assert!((f32_values[0] - 1.0).abs() < 0.01);
    assert!((f32_values[1] - 0.5).abs() < 0.01);
    assert!((f32_values[2] - (-0.25)).abs() < 0.01);
}

#[test]
fn test_f32_to_embedding_bytes_roundtrip() {
    let mut original = vec![0.0f32; crate::constants::DEFAULT_EMBEDDING_DIM];
    original[0] = 1.0;
    original[1] = 0.5;
    original[2] = -0.25;
    original[3] = 100.5;
    let bytes = f32_to_embedding_bytes(&original);
    let roundtrip = embedding_bytes_to_f32(&bytes).unwrap();

    assert_eq!(original.len(), roundtrip.len());
    for (orig, rt) in original.iter().zip(roundtrip.iter()) {
        assert!((orig - rt).abs() < 0.1, "{} vs {}", orig, rt);
    }
}

#[test]
fn test_generate_point_id() {
    let id1 = generate_point_id(1000, 2000);
    let id2 = generate_point_id(1000, 2001);
    let id3 = generate_point_id(1001, 2000);

    assert_ne!(id1, id2);
    assert_ne!(id1, id3);
    assert_ne!(id2, id3);

    assert_eq!(id1, generate_point_id(1000, 2000));
}

#[test]
fn test_cosine_similarity_identical() {
    let v = vec![1.0, 2.0, 3.0];
    let similarity = cosine_similarity(&v, &v);
    assert!((similarity - 1.0).abs() < 0.0001);
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let v1 = vec![1.0, 0.0];
    let v2 = vec![0.0, 1.0];
    let similarity = cosine_similarity(&v1, &v2);
    assert!(similarity.abs() < 0.0001);
}

#[test]
fn test_cosine_similarity_opposite() {
    let v1 = vec![1.0, 0.0];
    let v2 = vec![-1.0, 0.0];
    let similarity = cosine_similarity(&v1, &v2);
    assert!((similarity - (-1.0)).abs() < 0.0001);
}

#[test]
fn test_cosine_similarity_different_lengths() {
    let v1 = vec![1.0, 2.0];
    let v2 = vec![1.0, 2.0, 3.0];
    let similarity = cosine_similarity(&v1, &v2);
    assert_eq!(similarity, 0.0);
}

#[test]
fn test_vector_point_new() {
    let point = VectorPoint::new(
        1,
        vec![0.1; crate::constants::DEFAULT_EMBEDDING_DIM],
        1000,
        2000,
    );

    assert_eq!(point.id, 1);
    assert_eq!(point.vector.len(), crate::constants::DEFAULT_EMBEDDING_DIM);
    assert_eq!(point.tenant_id, 1000);
    assert_eq!(point.context_hash, 2000);
    assert_eq!(point.timestamp, 0);
    assert!(point.storage_key.is_none());
}

#[test]
fn test_vector_point_builder_pattern() {
    let point = VectorPoint::new(
        1,
        vec![0.1; crate::constants::DEFAULT_EMBEDDING_DIM],
        1000,
        2000,
    )
    .with_timestamp(1702512000)
    .with_storage_key("my_key".to_string());

    assert_eq!(point.timestamp, 1702512000);
    assert_eq!(point.storage_key, Some("my_key".to_string()));
}

#[test]
fn test_vector_point_from_embedding_bytes() {
    let bytes: Vec<u8> = (0..crate::constants::EMBEDDING_F16_BYTES)
        .map(|i| (i % 256) as u8)
        .collect();

    let point = VectorPoint::from_embedding_bytes(1, &bytes, 1000, 2000).unwrap();

    assert_eq!(point.id, 1);
    assert_eq!(point.vector.len(), crate::constants::DEFAULT_EMBEDDING_DIM);
    assert_eq!(point.tenant_id, 1000);
    assert_eq!(point.context_hash, 2000);
}

#[test]
fn test_embedding_bytes_to_f32_rejects_invalid_lengths() {
    let bytes = vec![0u8; crate::constants::EMBEDDING_F16_BYTES - 1];
    let err = embedding_bytes_to_f32(&bytes).unwrap_err();
    assert!(matches!(
        err,
        VectorDbError::InvalidEmbeddingBytesLength { .. }
    ));
}

#[test]
fn test_error_messages() {
    let err = VectorDbError::ConnectionFailed {
        url: "http://localhost:6334".to_string(),
        message: "connection refused".to_string(),
    };
    assert!(err.to_string().contains("localhost:6334"));
    assert!(err.to_string().contains("connection refused"));

    let err = VectorDbError::CollectionNotFound {
        collection: "my_collection".to_string(),
    };
    assert!(err.to_string().contains("my_collection"));

    let err = VectorDbError::InvalidDimension {
        expected: crate::constants::DEFAULT_EMBEDDING_DIM,
        actual: 768,
    };
    assert!(
        err.to_string()
            .contains(&crate::constants::DEFAULT_EMBEDDING_DIM.to_string())
    );
    assert!(err.to_string().contains("768"));
}

#[test]
fn test_cosine_similarity_empty_vectors() {
    let v1: Vec<f32> = vec![];
    let v2: Vec<f32> = vec![];
    let similarity = cosine_similarity(&v1, &v2);
    assert_eq!(similarity, 0.0, "Empty vectors should return 0.0");
}

#[test]
fn test_cosine_similarity_zero_norm_a() {
    let v1 = vec![0.0, 0.0, 0.0];
    let v2 = vec![1.0, 2.0, 3.0];
    let similarity = cosine_similarity(&v1, &v2);
    assert_eq!(similarity, 0.0, "Zero vector should return 0.0");
}

#[test]
fn test_cosine_similarity_zero_norm_b() {
    let v1 = vec![1.0, 2.0, 3.0];
    let v2 = vec![0.0, 0.0, 0.0];
    let similarity = cosine_similarity(&v1, &v2);
    assert_eq!(similarity, 0.0, "Zero vector should return 0.0");
}

#[tokio::test]
async fn test_delete_from_nonexistent_collection() {
    let client = MockVectorDbClient::new();

    let result = client.delete_points("nonexistent", vec![1, 2, 3]).await;

    assert!(matches!(
        result,
        Err(VectorDbError::CollectionNotFound { .. })
    ));
}

#[test]
fn test_mock_client_new() {
    let client = MockVectorDbClient::new();
    // Verify that a freshly created client returns None for point_count
    assert!(client.point_count("any_collection").is_none());
}

#[tokio::test]
async fn test_search_result_includes_all_fields() {
    let client = MockVectorDbClient::new();
    client
        .ensure_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    let point = create_test_point(42, 1000);
    client
        .upsert_points(
            TEST_COLLECTION,
            vec![point],
            crate::vectordb::WriteConsistency::Strong,
        )
        .await
        .unwrap();

    let query = create_test_vector(42);
    let results = client
        .search(TEST_COLLECTION, query, 1, None)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 42);
    assert_eq!(results[0].tenant_id, 1000);
    assert_eq!(results[0].context_hash, 4200);
    assert_eq!(results[0].storage_key, Some("key_42".to_string()));
    assert!(results[0].timestamp > 0);
}

#[test]
fn test_write_consistency_from_strong() {
    let consistency = crate::vectordb::WriteConsistency::Strong;
    let wait: bool = consistency.into();
    assert!(wait);
}

#[test]
fn test_write_consistency_from_eventual() {
    let consistency = crate::vectordb::WriteConsistency::Eventual;
    let wait: bool = consistency.into();
    assert!(!wait);
}

mod lock_poison_tests {
    use super::*;

    fn create_poisoned_client() -> MockVectorDbClient {
        let client = MockVectorDbClient::new();
        client.poison_lock();
        client
    }

    #[tokio::test]
    async fn test_ensure_collection_lock_poisoned() {
        let client = create_poisoned_client();

        let result = client.ensure_collection("test", TEST_VECTOR_SIZE).await;

        assert!(matches!(
            result,
            Err(VectorDbError::CreateCollectionFailed { .. })
        ));
    }

    #[tokio::test]
    async fn test_upsert_points_lock_poisoned() {
        let client = create_poisoned_client();

        let point = create_test_point(1, 1000);
        let result = client
            .upsert_points(
                "test",
                vec![point],
                crate::vectordb::WriteConsistency::Strong,
            )
            .await;

        assert!(matches!(result, Err(VectorDbError::UpsertFailed { .. })));
    }

    #[tokio::test]
    async fn test_search_lock_poisoned() {
        let client = create_poisoned_client();

        let query = create_test_vector(1);
        let result = client.search("test", query, 10, None).await;

        assert!(matches!(result, Err(VectorDbError::SearchFailed { .. })));
    }

    #[tokio::test]
    async fn test_delete_points_lock_poisoned() {
        let client = create_poisoned_client();

        let result = client.delete_points("test", vec![1, 2, 3]).await;

        assert!(matches!(result, Err(VectorDbError::DeleteFailed { .. })));
    }

    #[test]
    fn test_point_count_lock_poisoned() {
        let client = create_poisoned_client();

        // point_count returns None when lock is poisoned (via .ok()?)
        let result = client.point_count("test");
        assert!(result.is_none());
    }
}

#[test]
fn test_mock_client_default() {
    let client = MockVectorDbClient::default();
    // Default client should have no collections
    assert!(client.point_count("any_collection").is_none());
}

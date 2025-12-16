use super::*;
use crate::vectordb::{VectorDbError, VectorPoint, WriteConsistency};

const TEST_COLLECTION: &str = "test_bq_collection";
const TEST_VECTOR_SIZE: u64 = crate::constants::DEFAULT_VECTOR_SIZE_U64;

fn create_test_vector(seed: u64) -> Vec<f32> {
    (0..TEST_VECTOR_SIZE)
        .map(|i| {
            let mixed = (seed.wrapping_mul(31).wrapping_add(i)) % 1000;
            (mixed as f32 / 500.0) - 1.0
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

#[test]
fn test_quantize_to_binary_basic() {
    let vector = vec![1.0, -0.5, 0.3, -0.1, 0.0, 0.8, -0.9, 0.2];
    let binary = quantize_to_binary(&vector);

    assert_eq!(binary.len(), 1);
    assert_eq!(binary[0], 0b10100101);
}

#[test]
fn test_quantize_to_binary_size() {
    let vector = create_test_vector(42);
    let binary = quantize_to_binary(&vector);

    assert_eq!(binary.len(), BQ_BYTES_PER_VECTOR);
    assert_eq!(binary.len(), crate::constants::EMBEDDING_BQ_BYTES);
}

#[test]
fn test_quantize_to_binary_all_positive() {
    let vector = vec![1.0; 16];
    let binary = quantize_to_binary(&vector);

    assert_eq!(binary.len(), 2);
    assert_eq!(binary[0], 0xFF);
    assert_eq!(binary[1], 0xFF);
}

#[test]
fn test_quantize_to_binary_all_negative() {
    let vector = vec![-1.0; 16];
    let binary = quantize_to_binary(&vector);

    assert_eq!(binary.len(), 2);
    assert_eq!(binary[0], 0x00);
    assert_eq!(binary[1], 0x00);
}

#[test]
fn test_hamming_distance_identical() {
    let a = vec![0xFF, 0x00, 0xAA];
    let b = vec![0xFF, 0x00, 0xAA];

    assert_eq!(hamming_distance(&a, &b), 0);
}

#[test]
fn test_hamming_distance_opposite() {
    let a = vec![0x00];
    let b = vec![0xFF];

    assert_eq!(hamming_distance(&a, &b), 8);
}

#[test]
fn test_hamming_distance_single_bit() {
    let a = vec![0b00000000];
    let b = vec![0b00000001];

    assert_eq!(hamming_distance(&a, &b), 1);
}

#[test]
fn test_hamming_distance_different_lengths() {
    let a = vec![0x00, 0x00];
    let b = vec![0xFF];

    assert_eq!(hamming_distance(&a, &b), u32::MAX);
}

#[test]
fn test_bq_config_defaults() {
    let config = BqConfig::default();

    assert!(config.always_ram);
    assert!(config.rescore);
    assert_eq!(config.rescore_limit, 50);
    assert!(config.on_disk_payload);
}

#[test]
fn test_bq_config_builder() {
    let config = BqConfig::new()
        .always_ram(false)
        .rescore(false)
        .rescore_limit(100);

    assert!(!config.always_ram);
    assert!(!config.rescore);
    assert_eq!(config.rescore_limit, 100);
}

#[test]
fn test_estimate_ram_bytes() {
    let config = BqConfig::default();

    let estimate = config.estimate_ram_bytes(1_000_000);
    assert_eq!(estimate, 192_000_000);
}

#[test]
fn test_estimate_savings() {
    let config = BqConfig::default();

    let savings = config.estimate_savings_bytes(1_000_000);
    let original = 1_000_000 * ORIGINAL_BYTES_PER_VECTOR as u64;
    let compressed = 1_000_000 * BQ_BYTES_PER_VECTOR as u64;

    assert_eq!(savings, original - compressed);
}

#[test]
fn test_compression_ratio() {
    assert_eq!(BQ_COMPRESSION_RATIO, 32);
}

#[tokio::test]
async fn test_mock_ensure_bq_collection() {
    let client = MockBqClient::new();

    client
        .ensure_bq_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .expect("should create collection");

    assert_eq!(client.point_count(TEST_COLLECTION), Some(0));
}

#[tokio::test]
async fn test_mock_ensure_bq_collection_idempotent() {
    let client = MockBqClient::new();

    client
        .ensure_bq_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();
    client
        .ensure_bq_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    assert_eq!(client.point_count(TEST_COLLECTION), Some(0));
}

#[tokio::test]
async fn test_mock_upsert_and_search() {
    let client = MockBqClient::new();
    client
        .ensure_bq_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    let points: Vec<_> = (0..10).map(|i| create_test_point(i, 1000)).collect();
    client
        .upsert_points(TEST_COLLECTION, points, WriteConsistency::Strong)
        .await
        .unwrap();

    assert_eq!(client.point_count(TEST_COLLECTION), Some(10));

    let query = create_test_vector(0);
    let results = client
        .search_bq(TEST_COLLECTION, query, 5, None)
        .await
        .unwrap();

    assert!(!results.is_empty());
    assert!(results.len() <= 5);

    for i in 1..results.len() {
        assert!(
            results[i - 1].score >= results[i].score,
            "Results should be sorted by score descending"
        );
    }
}

#[tokio::test]
async fn test_mock_search_with_tenant_filter() {
    let client = MockBqClient::new();
    client
        .ensure_bq_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    let tenant_a: Vec<_> = (0..5).map(|i| create_test_point(i, 1000)).collect();
    let tenant_b: Vec<_> = (5..10).map(|i| create_test_point(i, 2000)).collect();

    client
        .upsert_points(TEST_COLLECTION, tenant_a, WriteConsistency::Strong)
        .await
        .unwrap();
    client
        .upsert_points(TEST_COLLECTION, tenant_b, WriteConsistency::Strong)
        .await
        .unwrap();

    let query = create_test_vector(0);
    let results = client
        .search_bq(TEST_COLLECTION, query, 10, Some(1000))
        .await
        .unwrap();

    for result in &results {
        assert_eq!(result.tenant_id, 1000);
    }
}

#[tokio::test]
async fn test_mock_search_without_rescore() {
    let config = BqConfig::new().rescore(false);
    let client = MockBqClient::with_config(config);
    client
        .ensure_bq_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    let points: Vec<_> = (0..10).map(|i| create_test_point(i, 1000)).collect();
    client
        .upsert_points(TEST_COLLECTION, points, WriteConsistency::Strong)
        .await
        .unwrap();

    let query = create_test_vector(0);
    let results = client
        .search_bq(TEST_COLLECTION, query, 5, None)
        .await
        .unwrap();

    assert!(!results.is_empty());
    for result in &results {
        assert!(result.score >= 0.0 && result.score <= 1.0);
    }
}

#[tokio::test]
async fn test_mock_delete_points() {
    let client = MockBqClient::new();
    client
        .ensure_bq_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    let points: Vec<_> = (0..10).map(|i| create_test_point(i, 1000)).collect();
    client
        .upsert_points(TEST_COLLECTION, points, WriteConsistency::Strong)
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
async fn test_mock_search_empty_collection() {
    let client = MockBqClient::new();
    client
        .ensure_bq_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    let query = create_test_vector(0);
    let results = client
        .search_bq(TEST_COLLECTION, query, 10, None)
        .await
        .unwrap();

    assert!(results.is_empty());
}

#[tokio::test]
async fn test_mock_search_nonexistent_collection() {
    let client = MockBqClient::new();

    let query = create_test_vector(0);
    let result = client.search_bq("nonexistent", query, 10, None).await;

    assert!(matches!(
        result,
        Err(VectorDbError::CollectionNotFound { .. })
    ));
}

#[tokio::test]
async fn test_mock_upsert_wrong_dimension() {
    let client = MockBqClient::new();
    client
        .ensure_bq_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    let wrong_point = VectorPoint {
        id: 1,
        vector: vec![0.1; 100],
        tenant_id: 1000,
        context_hash: 2000,
        timestamp: 0,
        storage_key: None,
    };

    let result = client
        .upsert_points(TEST_COLLECTION, vec![wrong_point], WriteConsistency::Strong)
        .await;

    assert!(matches!(
        result,
        Err(VectorDbError::InvalidDimension { .. })
    ));
}

#[tokio::test]
async fn test_bq_search_finds_similar_vectors() {
    let client = MockBqClient::new();
    client
        .ensure_bq_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    let base = create_test_vector(42);

    let similar: Vec<f32> = base.iter().map(|&v| v + 0.01).collect();

    let different: Vec<f32> = base.iter().map(|&v| -v).collect();

    let points = vec![
        VectorPoint::new(1, base.clone(), 1000, 100),
        VectorPoint::new(2, similar, 1000, 200),
        VectorPoint::new(3, different, 1000, 300),
    ];

    client
        .upsert_points(TEST_COLLECTION, points, WriteConsistency::Strong)
        .await
        .unwrap();

    let results = client
        .search_bq(TEST_COLLECTION, base, 3, None)
        .await
        .unwrap();

    assert_eq!(results.len(), 3);

    assert_eq!(results[0].id, 1);

    assert_eq!(results[1].id, 2);

    assert_eq!(results[2].id, 3);
}

#[test]
fn test_mock_bq_client_new() {
    let client = MockBqClient::new();
    // Verify that a freshly created client has no collections
    assert!(client.point_count("any_collection").is_none());
}

#[tokio::test]
async fn test_mock_upsert_to_nonexistent_collection() {
    let client = MockBqClient::new();

    let point = create_test_point(1, 1000);
    let result = client
        .upsert_points("nonexistent", vec![point], WriteConsistency::Strong)
        .await;

    assert!(matches!(
        result,
        Err(VectorDbError::CollectionNotFound { .. })
    ));
}

#[tokio::test]
async fn test_mock_delete_from_nonexistent_collection() {
    let client = MockBqClient::new();

    let result = client.delete_points("nonexistent", vec![1, 2, 3]).await;

    assert!(matches!(
        result,
        Err(VectorDbError::CollectionNotFound { .. })
    ));
}

#[tokio::test]
async fn test_mock_point_count_nonexistent_collection() {
    let client = MockBqClient::new();
    assert!(client.point_count("nonexistent").is_none());
}

#[tokio::test]
async fn test_mock_search_result_includes_storage_key() {
    let client = MockBqClient::new();
    client
        .ensure_bq_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    let point = create_test_point(1, 1000);
    client
        .upsert_points(TEST_COLLECTION, vec![point], WriteConsistency::Strong)
        .await
        .unwrap();

    let query = create_test_vector(1);
    let results = client
        .search_bq(TEST_COLLECTION, query, 1, None)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].storage_key, Some("key_1".to_string()));
    assert_eq!(results[0].context_hash, 100);
    assert!(results[0].timestamp > 0);
}

#[tokio::test]
async fn test_mock_delete_nonexistent_points() {
    let client = MockBqClient::new();
    client
        .ensure_bq_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    // Deleting non-existent points should succeed without error
    client
        .delete_points(TEST_COLLECTION, vec![99999, 88888])
        .await
        .expect("Should succeed even with non-existent points");
}

#[tokio::test]
async fn test_mock_delete_empty_list() {
    let client = MockBqClient::new();
    client
        .ensure_bq_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
        .await
        .unwrap();

    // Deleting empty list should succeed
    client
        .delete_points(TEST_COLLECTION, vec![])
        .await
        .expect("Empty delete should succeed");
}

#[test]
fn test_mock_bq_client_with_config() {
    let config = BqConfig::new()
        .always_ram(false)
        .rescore(false)
        .rescore_limit(100);

    let client = MockBqClient::with_config(config);
    // Verify client was created with config (indirectly through behavior)
    assert!(client.point_count("any").is_none());
}

mod lock_poison_tests {
    use super::*;

    fn create_poisoned_bq_client() -> MockBqClient {
        let client = MockBqClient::new();
        client.poison_lock();
        client
    }

    #[tokio::test]
    async fn test_ensure_collection_lock_poisoned() {
        let client = create_poisoned_bq_client();

        let result = client.ensure_bq_collection("test", 1536).await;

        assert!(matches!(
            result,
            Err(VectorDbError::CreateCollectionFailed { .. })
        ));
    }

    #[tokio::test]
    async fn test_upsert_points_lock_poisoned() {
        let client = create_poisoned_bq_client();

        let point = VectorPoint::new(1, vec![0.1; 1536], 1000, 2000);
        let result = client
            .upsert_points("test", vec![point], WriteConsistency::Strong)
            .await;

        assert!(matches!(result, Err(VectorDbError::UpsertFailed { .. })));
    }

    #[tokio::test]
    async fn test_search_bq_lock_poisoned() {
        let client = create_poisoned_bq_client();

        let query = vec![0.1f32; 1536];
        let result = client.search_bq("test", query, 10, None).await;

        assert!(matches!(result, Err(VectorDbError::SearchFailed { .. })));
    }

    #[tokio::test]
    async fn test_delete_points_lock_poisoned() {
        let client = create_poisoned_bq_client();

        let result = client.delete_points("test", vec![1, 2, 3]).await;

        assert!(matches!(result, Err(VectorDbError::DeleteFailed { .. })));
    }

    #[test]
    fn test_point_count_lock_poisoned() {
        let client = create_poisoned_bq_client();

        // point_count returns None when lock is poisoned (via .ok()?)
        let result = client.point_count("test");
        assert!(result.is_none());
    }
}

#[test]
fn test_mock_bq_client_clone() {
    let client = MockBqClient::new();

    // Create a collection
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        client
            .ensure_bq_collection(TEST_COLLECTION, TEST_VECTOR_SIZE)
            .await
            .unwrap();
        let point = create_test_point(1, 1000);
        client
            .upsert_points(TEST_COLLECTION, vec![point], WriteConsistency::Strong)
            .await
            .unwrap();
    });

    // Clone the client - both should share the same Arc<RwLock<...>>
    let cloned = client.clone();

    // Verify both see the same data
    assert_eq!(client.point_count(TEST_COLLECTION), Some(1));
    assert_eq!(cloned.point_count(TEST_COLLECTION), Some(1));

    // Add data via clone, original should see it too (shared state)
    rt.block_on(async {
        let point = create_test_point(2, 1000);
        cloned
            .upsert_points(TEST_COLLECTION, vec![point], WriteConsistency::Strong)
            .await
            .unwrap();
    });

    assert_eq!(client.point_count(TEST_COLLECTION), Some(2));
    assert_eq!(cloned.point_count(TEST_COLLECTION), Some(2));
}

#[test]
fn test_mock_bq_client_default() {
    let client = MockBqClient::default();
    // Default client should have no collections
    assert!(client.point_count("any_collection").is_none());
}

use super::*;
use crate::storage::CacheEntry;
use crate::vectordb::VectorDbError;
use crate::vectordb::bq::BqConfig;
use crate::vectordb::rescoring::ScoredCandidate;
use half::f16;

#[test]
fn test_l2_config_default() {
    let config = L2Config::default();

    assert_eq!(config.top_k_bq, DEFAULT_TOP_K_BQ);
    assert_eq!(config.top_k_final, DEFAULT_TOP_K_FINAL);
    assert_eq!(config.collection_name, L2_COLLECTION_NAME);
    assert_eq!(config.vector_size, L2_VECTOR_SIZE);
    assert!(config.validate_dimensions);
}

#[test]
fn test_l2_config_with_top_k() {
    let config = L2Config::with_top_k(100, 10);

    assert_eq!(config.top_k_bq, 100);
    assert_eq!(config.top_k_final, 10);
}

#[test]
fn test_l2_config_builder() {
    let config = L2Config::default()
        .collection_name("my_collection")
        .bq_config(BqConfig::default().rescore(false));

    assert_eq!(config.collection_name, "my_collection");
    assert!(!config.bq_config.rescore);
}

#[test]
fn test_l2_config_validation_success() {
    let config = L2Config::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_l2_config_validation_zero_top_k_bq() {
    let config = L2Config {
        top_k_bq: 0,
        ..L2Config::default()
    };
    assert!(matches!(
        config.validate(),
        Err(L2CacheError::ConfigError { .. })
    ));
}

#[test]
fn test_l2_config_validation_zero_top_k_final() {
    let config = L2Config {
        top_k_final: 0,
        ..L2Config::default()
    };
    assert!(matches!(
        config.validate(),
        Err(L2CacheError::ConfigError { .. })
    ));
}

#[test]
fn test_l2_config_validation_top_k_final_greater_than_bq() {
    let config = L2Config {
        top_k_bq: 10,
        top_k_final: 20,
        ..L2Config::default()
    };
    assert!(matches!(
        config.validate(),
        Err(L2CacheError::ConfigError { .. })
    ));
}

#[test]
fn test_mock_storage_loader_new() {
    let loader = MockStorageLoader::new();
    assert!(loader.is_empty());
    assert_eq!(loader.len(), 0);
}

#[tokio::test]
async fn test_mock_storage_loader_insert_and_load() {
    let loader = MockStorageLoader::new();

    let entry = CacheEntry {
        tenant_id: 1000,
        context_hash: 2000,
        timestamp: 1702500000,
        embedding: vec![0u8; crate::constants::EMBEDDING_F16_BYTES],
        payload_blob: vec![0xDE, 0xAD, 0xBE, 0xEF],
    };

    loader.insert("key_1", entry.clone());

    assert_eq!(loader.len(), 1);
    assert!(!loader.is_empty());

    let loaded = loader.load("key_1", 1000).await.expect("should load entry");
    assert_eq!(loaded.tenant_id, 1000);
    assert_eq!(loaded.context_hash, 2000);
}

#[tokio::test]
async fn test_mock_storage_loader_load_missing() {
    let loader = MockStorageLoader::new();
    let result = loader.load("nonexistent", 0).await;
    assert!(result.is_none());
}

#[test]
fn test_l2_lookup_result_methods() {
    let entry = CacheEntry {
        tenant_id: 1,
        context_hash: 2,
        timestamp: 3,
        embedding: vec![],
        payload_blob: vec![],
    };

    let scored = ScoredCandidate {
        id: 42,
        entry,
        score: 0.95,
        bq_score: Some(0.80),
    };

    let result = L2LookupResult {
        query_embedding: vec![f16::from_f32(0.1); 10],
        candidates: vec![scored],
        tenant_id: 1000,
        bq_candidates_count: 50,
    };

    assert_eq!(result.query_embedding().len(), 10);
    assert_eq!(result.candidates().len(), 1);
    assert_eq!(result.tenant_id(), 1000);
    assert_eq!(result.bq_candidates_count(), 50);
    assert!(result.has_candidates());
    assert!(result.best_candidate().is_some());
    assert_eq!(result.best_candidate().unwrap().id, 42);
}

#[test]
fn test_l2_lookup_result_empty() {
    let result = L2LookupResult {
        query_embedding: vec![],
        candidates: vec![],
        tenant_id: 1000,
        bq_candidates_count: 0,
    };

    assert!(!result.has_candidates());
    assert!(result.best_candidate().is_none());
}

#[test]
fn test_l2_lookup_result_into_candidates() {
    let entry = CacheEntry {
        tenant_id: 1,
        context_hash: 2,
        timestamp: 3,
        embedding: vec![],
        payload_blob: vec![],
    };

    let scored = ScoredCandidate {
        id: 42,
        entry,
        score: 0.95,
        bq_score: None,
    };

    let result = L2LookupResult {
        query_embedding: vec![],
        candidates: vec![scored],
        tenant_id: 1000,
        bq_candidates_count: 1,
    };

    let candidates = result.into_candidates();
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].id, 42);
}

#[tokio::test]
async fn test_mock_l2_cache_creation() {
    let config = L2Config::default();
    let cache = L2SemanticCache::new_mock(config)
        .await
        .expect("should create mock cache");

    assert!(cache.is_embedder_stub());
}

#[tokio::test]
async fn test_mock_l2_cache_index_and_search() {
    let config = L2Config::default();
    let cache = L2SemanticCache::new_mock(config)
        .await
        .expect("should create cache");

    let entry = CacheEntry {
        tenant_id: 1000,
        context_hash: 2000,
        timestamp: 1702500000,
        embedding: vec![0u8; crate::constants::EMBEDDING_F16_BYTES],
        payload_blob: vec![0xDE, 0xAD, 0xBE, 0xEF],
    };

    cache.storage().insert("storage_key_1", entry);

    let point_id = cache
        .index(
            "What is the capital of France?",
            1000,
            2000,
            "storage_key_1",
            1702500000,
        )
        .await
        .expect("should index entry");

    assert!(point_id > 0);

    let result = cache
        .search("What is the capital of France?", 1000)
        .await
        .expect("should find result");

    assert!(result.has_candidates());
    assert!(!result.query_embedding().is_empty());
}

#[tokio::test]
async fn test_mock_l2_cache_search_empty() {
    let config = L2Config::default();
    let cache = L2SemanticCache::new_mock(config)
        .await
        .expect("should create cache");

    let result = cache.search("Test prompt", 1000).await;

    assert!(matches!(result, Err(L2CacheError::NoCandidates)));
}

#[tokio::test]
async fn test_mock_l2_cache_tenant_isolation() {
    let config = L2Config::default();
    let cache = L2SemanticCache::new_mock(config)
        .await
        .expect("should create cache");

    let entry_tenant_a = CacheEntry {
        tenant_id: 1000,
        context_hash: 2000,
        timestamp: 1702500000,
        embedding: vec![0u8; crate::constants::EMBEDDING_F16_BYTES],
        payload_blob: vec![0x01],
    };

    let entry_tenant_b = CacheEntry {
        tenant_id: 2000,
        context_hash: 3000,
        timestamp: 1702500000,
        embedding: vec![0u8; crate::constants::EMBEDDING_F16_BYTES],
        payload_blob: vec![0x02],
    };

    cache.storage().insert("key_tenant_a", entry_tenant_a);
    cache.storage().insert("key_tenant_b", entry_tenant_b);

    cache
        .index("Query for tenant A", 1000, 2000, "key_tenant_a", 1702500000)
        .await
        .expect("should index tenant A");

    cache
        .index("Query for tenant B", 2000, 3000, "key_tenant_b", 1702500000)
        .await
        .expect("should index tenant B");

    let result_a = cache
        .search("Query for tenant A", 1000)
        .await
        .expect("should find result");

    for candidate in result_a.candidates() {
        assert_eq!(candidate.entry.tenant_id, 1000);
    }

    let result_b = cache
        .search("Query for tenant B", 2000)
        .await
        .expect("should find result");

    for candidate in result_b.candidates() {
        assert_eq!(candidate.entry.tenant_id, 2000);
    }
}

#[tokio::test]
async fn test_mock_l2_cache_multiple_entries() {
    let config = L2Config::with_top_k(50, 3);
    let cache = L2SemanticCache::new_mock(config)
        .await
        .expect("should create cache");

    for i in 0..10 {
        let entry = CacheEntry {
            tenant_id: 1000,
            context_hash: i as u64,
            timestamp: 1702500000 + i as i64,
            embedding: vec![((i * 7) % 256) as u8; crate::constants::EMBEDDING_F16_BYTES],
            payload_blob: vec![i as u8],
        };

        let key = format!("key_{}", i);
        cache.storage().insert(&key, entry);

        cache
            .index(
                &format!("Query number {}", i),
                1000,
                i as u64,
                &key,
                1702500000,
            )
            .await
            .expect("should index");
    }

    let result = cache
        .search("Query number 0", 1000)
        .await
        .expect("should find results");

    assert!(result.candidates().len() <= 3);
    assert!(result.bq_candidates_count() <= 10);

    let candidates = result.candidates();
    for i in 1..candidates.len() {
        assert!(
            candidates[i - 1].score >= candidates[i].score,
            "Candidates should be sorted by score descending"
        );
    }
}

#[tokio::test]
async fn test_l2_cache_handle_clone() {
    let config = L2Config::default();
    let cache = L2SemanticCache::new_mock(config)
        .await
        .expect("should create cache");

    let handle = L2SemanticCacheHandle::new(cache);
    assert_eq!(handle.strong_count(), 1);

    let clone = handle.clone();
    assert_eq!(handle.strong_count(), 2);
    assert_eq!(clone.strong_count(), 2);
}

#[tokio::test]
async fn test_l2_cache_handle_concurrent_search() {
    let config = L2Config::default();
    let cache = L2SemanticCache::new_mock(config)
        .await
        .expect("should create cache");

    let entry = CacheEntry {
        tenant_id: 1000,
        context_hash: 2000,
        timestamp: 1702500000,
        embedding: vec![0u8; crate::constants::EMBEDDING_F16_BYTES],
        payload_blob: vec![],
    };
    cache.storage().insert("key_1", entry);

    cache
        .index("Test query", 1000, 2000, "key_1", 1702500000)
        .await
        .expect("should index");

    let handle = L2SemanticCacheHandle::new(cache);

    let mut handles = Vec::new();
    for _ in 0..5 {
        let cache_clone = handle.clone();
        handles.push(tokio::spawn(async move {
            cache_clone.search("Test query", 1000).await
        }));
    }

    for h in handles {
        let result: Result<L2LookupResult, L2CacheError> = h.await.expect("task should complete");
        assert!(result.is_ok());
    }
}

#[test]
fn test_error_messages() {
    let err = L2CacheError::EmbeddingFailed {
        reason: "model not loaded".to_string(),
    };
    assert!(err.to_string().contains("embedding"));
    assert!(err.to_string().contains("model not loaded"));

    let err = L2CacheError::NoCandidates;
    assert!(err.to_string().contains("no candidates"));

    let err = L2CacheError::ConfigError {
        reason: "invalid top_k".to_string(),
    };
    assert!(err.to_string().contains("configuration"));
}

#[test]
fn test_error_from_vectordb() {
    let vdb_err = VectorDbError::CollectionNotFound {
        collection: "test".to_string(),
    };
    let l2_err: L2CacheError = vdb_err.into();

    assert!(matches!(l2_err, L2CacheError::VectorDb(_)));
}

#[tokio::test]
async fn test_debug_impl() {
    let config = L2Config::default();
    let cache = L2SemanticCache::new_mock(config)
        .await
        .expect("should create cache");

    let debug_str = format!("{:?}", cache);
    assert!(debug_str.contains("L2SemanticCache"));
    assert!(debug_str.contains("embedder"));
    assert!(debug_str.contains("rescorer"));
    assert!(debug_str.contains("config"));
}

#[tokio::test]
async fn test_handle_debug_impl() {
    let config = L2Config::default();
    let cache = L2SemanticCache::new_mock(config)
        .await
        .expect("should create cache");

    let handle = L2SemanticCacheHandle::new(cache);
    let debug_str = format!("{:?}", handle);

    assert!(debug_str.contains("L2SemanticCacheHandle"));
    assert!(debug_str.contains("strong_count"));
}

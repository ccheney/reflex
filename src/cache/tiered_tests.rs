use super::tiered::{TieredCache, TieredCacheHandle, TieredLookupResult};
use super::types::ReflexStatus;
use crate::storage::CacheEntry;
use crate::storage::mmap::MmapFileHandle;

#[test]
fn test_tiered_lookup_result_miss() {
    let result = TieredLookupResult::Miss;

    assert!(!result.is_hit());
    assert!(!result.is_l1_hit());
    assert!(!result.is_l2_hit());
    assert_eq!(result.status(), ReflexStatus::Miss);
}

#[tokio::test]
async fn test_mock_tiered_cache_creation() {
    let cache = TieredCache::new_mock().await.expect("should create cache");

    assert!(cache.l1_is_empty());
}

#[tokio::test]
async fn test_mock_tiered_cache_l1_operations() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let cache = TieredCache::new_mock().await.expect("should create cache");
    let mut file = NamedTempFile::new().expect("create temp file");
    file.write_all(b"test payload").expect("write");
    file.flush().expect("flush");

    let handle = MmapFileHandle::open(file.path()).expect("open mmap");

    let hash = cache.insert_l1("test prompt", 1000, handle);
    assert_eq!(hash.len(), 32);

    cache.run_pending_tasks_l1();
    assert!(!cache.l1_is_empty());
    assert_eq!(cache.l1_len(), 1);
    assert!(cache.contains_l1("test prompt", 1000));

    let removed = cache.remove_l1("test prompt", 1000);
    assert!(removed.is_some());

    cache.run_pending_tasks_l1();
    assert!(cache.l1_is_empty());
}

#[tokio::test]
async fn test_mock_tiered_cache_l1_hit() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let cache = TieredCache::new_mock().await.expect("should create cache");

    let mut file = NamedTempFile::new().expect("create temp file");
    file.write_all(b"test payload").expect("write");
    file.flush().expect("flush");

    let handle = MmapFileHandle::open(file.path()).expect("open mmap");

    cache.insert_l1("What is the capital of France?", 1000, handle);

    let result = cache
        .lookup("What is the capital of France?", 1000)
        .await
        .expect("lookup should succeed");

    assert!(result.is_hit());
    assert!(result.is_l1_hit());
    assert!(!result.is_l2_hit());
    assert_eq!(result.status(), ReflexStatus::HitL1Exact);
}

#[tokio::test]
async fn test_mock_tiered_cache_l2_hit() {
    let cache = TieredCache::new_mock().await.expect("should create cache");

    let entry = CacheEntry {
        tenant_id: 1000,
        context_hash: 2000,
        timestamp: 1702500000,
        embedding: vec![0u8; crate::constants::EMBEDDING_F16_BYTES],
        payload_blob: vec![0xDE, 0xAD],
    };

    cache.mock_storage().insert("storage_key_1", entry);

    cache
        .index_l2(
            "What is the capital of France?",
            1000,
            2000,
            "storage_key_1",
            1702500000,
        )
        .await
        .expect("should index");

    let result = cache
        .lookup("What is the capital of France?", 1000)
        .await
        .expect("lookup should succeed");

    assert!(result.is_hit());
    assert!(!result.is_l1_hit());
    assert!(result.is_l2_hit());
    assert_eq!(result.status(), ReflexStatus::HitL2Semantic);
}

#[tokio::test]
async fn test_mock_tiered_cache_miss() {
    let cache = TieredCache::new_mock().await.expect("should create cache");

    let result = cache
        .lookup("Unknown query", 1000)
        .await
        .expect("lookup should succeed");

    assert!(!result.is_hit());
    assert_eq!(result.status(), ReflexStatus::Miss);
}

#[tokio::test]
async fn test_mock_tiered_cache_l1_takes_priority() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let cache = TieredCache::new_mock().await.expect("should create cache");

    let entry = CacheEntry {
        tenant_id: 1000,
        context_hash: 2000,
        timestamp: 1702500000,
        embedding: vec![0u8; crate::constants::EMBEDDING_F16_BYTES],
        payload_blob: vec![],
    };
    cache.mock_storage().insert("storage_key", entry);

    cache
        .index_l2("shared prompt", 1000, 2000, "storage_key", 1702500000)
        .await
        .expect("should index");

    let mut file = NamedTempFile::new().expect("create temp file");
    file.write_all(b"l1 payload").expect("write");
    file.flush().expect("flush");

    let handle = MmapFileHandle::open(file.path()).expect("open mmap");
    cache.insert_l1("shared prompt", 1000, handle);

    let result = cache
        .lookup("shared prompt", 1000)
        .await
        .expect("lookup should succeed");

    assert!(result.is_l1_hit());
    assert!(!result.is_l2_hit());
}

#[tokio::test]
async fn test_mock_tiered_cache_clear_l1() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let cache = TieredCache::new_mock().await.expect("should create cache");

    for i in 0..5 {
        let mut file = NamedTempFile::new().expect("create temp file");
        file.write_all(format!("payload {}", i).as_bytes())
            .expect("write");
        file.flush().expect("flush");

        let handle = MmapFileHandle::open(file.path()).expect("open mmap");
        cache.insert_l1(&format!("prompt {}", i), 1000, handle);
    }

    cache.run_pending_tasks_l1();
    assert_eq!(cache.l1_len(), 5);

    cache.clear_l1();

    cache.run_pending_tasks_l1();
    assert!(cache.l1_is_empty());
}

#[tokio::test]
async fn test_tiered_cache_handle_clone() {
    let cache = TieredCache::new_mock().await.expect("should create cache");
    let handle = TieredCacheHandle::new(cache);

    assert_eq!(handle.strong_count(), 1);

    let clone = handle.clone();
    assert_eq!(handle.strong_count(), 2);
    assert_eq!(clone.strong_count(), 2);
}

#[tokio::test]
async fn test_tiered_cache_handle_debug() {
    let cache = TieredCache::new_mock().await.expect("should create cache");
    let handle = TieredCacheHandle::new(cache);

    let debug_str = format!("{:?}", handle);
    assert!(debug_str.contains("TieredCacheHandle"));
    assert!(debug_str.contains("strong_count"));
}

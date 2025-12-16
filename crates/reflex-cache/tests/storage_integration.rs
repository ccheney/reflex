//! Integration tests for storage.

mod common;

use common::fixtures::{
    CacheEntryBuilder, DEFAULT_TENANT_ID, EMBEDDING_SIZE_BYTES, assert_entries_equal,
    create_batch_entries, create_tenant_entries, create_time_series_entries,
    generate_deterministic_embedding,
};
use reflex::storage::CacheEntry;
use rkyv::rancor::Error;
use rkyv::{access, from_bytes, to_bytes};

#[test]
fn test_varying_payload_sizes() {
    let sizes = [0, 1, 100, 1024, 10_000, 100_000];

    for size in sizes {
        let payload = vec![0xAB_u8; size];
        let entry = CacheEntryBuilder::new()
            .payload_blob(payload.clone())
            .build();

        let serialized = to_bytes::<Error>(&entry).expect("Serialization should succeed");
        let deserialized: CacheEntry =
            from_bytes::<CacheEntry, Error>(&serialized).expect("Deserialization should succeed");

        assert_eq!(
            deserialized.payload_blob.len(),
            size,
            "Payload size mismatch for size {}",
            size
        );
        assert_eq!(deserialized.payload_blob, payload);
    }
}

#[test]
fn test_zero_copy_archive_access() {
    let original = CacheEntryBuilder::new()
        .tenant_id(12345)
        .context_hash(67890)
        .with_realistic_embedding()
        .build();

    let serialized = to_bytes::<Error>(&original).expect("Serialization should succeed");

    let archived =
        access::<ArchivedCacheEntry, Error>(&serialized).expect("Archive access should succeed");

    assert_eq!(archived.tenant_id, original.tenant_id);
    assert_eq!(archived.context_hash, original.context_hash);
    assert_eq!(archived.timestamp, original.timestamp);
    assert_eq!(archived.embedding.len(), EMBEDDING_SIZE_BYTES);
}

#[test]
fn test_archived_embedding_byte_access() {
    let embedding = generate_deterministic_embedding(42);
    let entry = CacheEntryBuilder::new()
        .embedding(embedding.clone())
        .build();

    let serialized = to_bytes::<Error>(&entry).expect("Serialization should succeed");
    let archived =
        access::<ArchivedCacheEntry, Error>(&serialized).expect("Archive access should succeed");

    for (i, byte) in embedding.iter().enumerate() {
        assert_eq!(archived.embedding[i], *byte, "Byte mismatch at index {}", i);
    }
}

#[test]
fn test_batch_archive_access() {
    let entries = create_batch_entries(50);
    let serialized: Vec<_> = entries
        .iter()
        .map(|e| to_bytes::<Error>(e).expect("Serialization should succeed"))
        .collect();

    let tenant_ids: Vec<u64> = serialized
        .iter()
        .map(|bytes| {
            let archived =
                access::<ArchivedCacheEntry, Error>(bytes).expect("Archive access should succeed");
            u64::from(archived.tenant_id)
        })
        .collect();

    for (i, &id) in tenant_ids.iter().enumerate() {
        assert_eq!(id, DEFAULT_TENANT_ID + i as u64);
    }
}

#[test]
fn test_multi_tenant_isolation() {
    let tenant_a_id = 1000;
    let tenant_b_id = 2000;

    let tenant_a_entries = create_tenant_entries(tenant_a_id, 5);
    let tenant_b_entries = create_tenant_entries(tenant_b_id, 5);

    for entry in &tenant_a_entries {
        assert_eq!(entry.tenant_id, tenant_a_id);
    }

    for entry in &tenant_b_entries {
        assert_eq!(entry.tenant_id, tenant_b_id);
    }

    assert_ne!(
        tenant_a_entries[0].embedding, tenant_b_entries[0].embedding,
        "Different tenants should have different embeddings"
    );
}

#[test]
fn test_tenant_filtering_via_archive() {
    let all_entries: Vec<_> = [1000_u64, 1000, 2000, 1000, 2000, 3000]
        .iter()
        .enumerate()
        .map(|(i, &tenant_id)| {
            CacheEntryBuilder::new()
                .tenant_id(tenant_id)
                .context_hash(i as u64)
                .build()
        })
        .collect();

    let serialized: Vec<_> = all_entries
        .iter()
        .map(|e| to_bytes::<Error>(e).expect("Serialization should succeed"))
        .collect();

    let target_tenant = 1000_u64;
    let filtered_count = serialized
        .iter()
        .filter(|bytes| {
            let archived =
                access::<ArchivedCacheEntry, Error>(bytes).expect("Archive access should succeed");
            archived.tenant_id == target_tenant
        })
        .count();

    assert_eq!(filtered_count, 3, "Should find 3 entries for tenant 1000");
}

#[test]
fn test_time_series_entries() {
    let start_time = 1702500000_i64;
    let interval = 3600_i64;
    let entries = create_time_series_entries(start_time, interval, 24);

    assert_eq!(entries.len(), 24);

    for (i, entry) in entries.iter().enumerate() {
        let expected_timestamp = start_time + (i as i64 * interval);
        assert_eq!(entry.timestamp, expected_timestamp);
    }
}

#[test]
fn test_timestamp_range_filtering() {
    let entries = create_time_series_entries(1000, 100, 20);
    let serialized: Vec<_> = entries
        .iter()
        .map(|e| to_bytes::<Error>(e).expect("Serialization should succeed"))
        .collect();

    let min_ts = 500_i64;
    let max_ts = 1500_i64;

    let in_range: Vec<_> = serialized
        .iter()
        .filter_map(|bytes| {
            let archived =
                access::<ArchivedCacheEntry, Error>(bytes).expect("Archive access should succeed");
            if archived.timestamp >= min_ts && archived.timestamp <= max_ts {
                Some(archived.timestamp)
            } else {
                None
            }
        })
        .collect();

    assert_eq!(in_range.len(), 6);
}

#[tokio::test]
async fn test_concurrent_serialization() {
    let entries = create_batch_entries(100);

    let handles: Vec<_> = entries
        .into_iter()
        .map(|entry| {
            tokio::spawn(async move {
                let serialized = to_bytes::<Error>(&entry).expect("Serialization should succeed");
                let deserialized: CacheEntry = from_bytes::<CacheEntry, Error>(&serialized)
                    .expect("Deserialization should succeed");
                (entry, deserialized)
            })
        })
        .collect();

    for handle in handles {
        let (original, deserialized) = handle.await.expect("Task should complete");
        assert_entries_equal(&original, &deserialized);
    }
}

#[tokio::test]
async fn test_concurrent_archive_reads() {
    let entry = CacheEntryBuilder::new()
        .tenant_id(999)
        .with_realistic_embedding()
        .build();
    let serialized = to_bytes::<Error>(&entry).expect("Serialization should succeed");
    let shared_bytes = std::sync::Arc::new(serialized.to_vec());

    let reader_count = 50;
    let handles: Vec<_> = (0..reader_count)
        .map(|_| {
            let bytes = std::sync::Arc::clone(&shared_bytes);
            tokio::spawn(async move {
                let archived = access::<ArchivedCacheEntry, Error>(&bytes)
                    .expect("Archive access should succeed");
                archived.tenant_id
            })
        })
        .collect();

    for handle in handles {
        let tenant_id = handle.await.expect("Task should complete");
        assert_eq!(tenant_id, 999);
    }
}

#[test]
fn test_memory_efficient_batch_processing() {
    let entry_count = 1000;

    for i in 0..entry_count {
        let entry = CacheEntryBuilder::new()
            .tenant_id(i as u64)
            .with_realistic_embedding()
            .build();

        let serialized = to_bytes::<Error>(&entry).expect("Serialization should succeed");
        let _deserialized: CacheEntry =
            from_bytes::<CacheEntry, Error>(&serialized).expect("Deserialization should succeed");
    }
}

#[test]
fn test_serialized_size_scaling() {
    let small_entry = CacheEntryBuilder::new()
        .embedding(vec![])
        .payload_blob(vec![])
        .build();

    let medium_entry = CacheEntryBuilder::new()
        .with_realistic_embedding()
        .payload_blob(vec![0; 1024])
        .build();

    let large_entry = CacheEntryBuilder::new()
        .embedding(vec![0; 10_000])
        .payload_blob(vec![0; 100_000])
        .build();

    let small_size = to_bytes::<Error>(&small_entry)
        .expect("Serialization should succeed")
        .len();
    let medium_size = to_bytes::<Error>(&medium_entry)
        .expect("Serialization should succeed")
        .len();
    let large_size = to_bytes::<Error>(&large_entry)
        .expect("Serialization should succeed")
        .len();

    assert!(
        small_size < medium_size,
        "Medium entry should be larger than small"
    );
    assert!(
        medium_size < large_size,
        "Large entry should be larger than medium"
    );

    assert!(large_size >= 100_000);
}

#[test]
fn test_extreme_field_values() {
    let entry = CacheEntry {
        tenant_id: u64::MAX,
        context_hash: u64::MAX,
        timestamp: i64::MAX,
        embedding: vec![0xFF; 100],
        payload_blob: vec![0xFF; 100],
    };

    let serialized = to_bytes::<Error>(&entry).expect("Serialization should succeed");
    let deserialized: CacheEntry =
        from_bytes::<CacheEntry, Error>(&serialized).expect("Deserialization should succeed");

    assert_eq!(deserialized.tenant_id, u64::MAX);
    assert_eq!(deserialized.context_hash, u64::MAX);
    assert_eq!(deserialized.timestamp, i64::MAX);
}

#[test]
fn test_minimum_field_values() {
    let entry = CacheEntry {
        tenant_id: 0,
        context_hash: 0,
        timestamp: i64::MIN,
        embedding: vec![],
        payload_blob: vec![],
    };

    let serialized = to_bytes::<Error>(&entry).expect("Serialization should succeed");
    let deserialized: CacheEntry =
        from_bytes::<CacheEntry, Error>(&serialized).expect("Deserialization should succeed");

    assert_eq!(deserialized.tenant_id, 0);
    assert_eq!(deserialized.timestamp, i64::MIN);
}

use reflex::storage::ArchivedCacheEntry;

use reflex::{MockStorageLoader, NvmeStorageLoader, StorageLoader};
use std::path::PathBuf;
use tempfile::TempDir;

#[tokio::test]
async fn test_nvme_storage_loader_reads_aligned_mmap_files() {
    use reflex::storage::mmap::AlignedMmapBuilder;

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_path = temp_dir.path().join("test_entry.rkyv");

    let entry = CacheEntryBuilder::new()
        .tenant_id(12345)
        .context_hash(67890)
        .timestamp(1702500000)
        .with_realistic_embedding()
        .with_sample_payload()
        .build();

    let serialized = to_bytes::<Error>(&entry)
        .expect("Serialization should succeed")
        .into_vec();

    let builder = AlignedMmapBuilder::new(&file_path);
    builder
        .write(&serialized)
        .expect("Failed to write via AlignedMmapBuilder");

    let loader = NvmeStorageLoader::new(temp_dir.path().to_path_buf());
    let loaded = loader
        .load("test_entry.rkyv", 12345)
        .await
        .expect("NvmeStorageLoader should find the file");

    assert_entries_equal(&entry, &loaded);
}

#[tokio::test]
async fn test_nvme_storage_loader_missing_file_returns_none() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let loader = NvmeStorageLoader::new(temp_dir.path().to_path_buf());

    let result = loader.load("nonexistent.rkyv", 0).await;
    assert!(result.is_none());
}

#[tokio::test]
async fn test_nvme_storage_loader_nested_tenant_paths() {
    use reflex::storage::mmap::AlignedMmapBuilder;

    let temp_dir = TempDir::new().expect("Failed to create temp dir");

    let tenant_id = 42_u64;
    let tenant_dir = temp_dir.path().join(tenant_id.to_string());
    std::fs::create_dir_all(&tenant_dir).expect("Failed to create tenant dir");

    let entry_id = "abc123";
    let file_path = tenant_dir.join(format!("{}.rkyv", entry_id));

    let entry = CacheEntryBuilder::new()
        .tenant_id(tenant_id)
        .context_hash(999)
        .build();

    let serialized = to_bytes::<Error>(&entry)
        .expect("Serialization should succeed")
        .into_vec();
    let builder = AlignedMmapBuilder::new(&file_path);
    builder.write(&serialized).expect("Write should succeed");

    let loader = NvmeStorageLoader::new(temp_dir.path().to_path_buf());
    let storage_key = format!("{}/{}.rkyv", tenant_id, entry_id);
    let loaded = loader
        .load(&storage_key, tenant_id)
        .await
        .expect("Should load nested file");

    assert_eq!(loaded.tenant_id, tenant_id);
}

#[tokio::test]
async fn test_storage_loader_trait_polymorphism() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");

    let mock = MockStorageLoader::new();
    let entry = CacheEntryBuilder::new()
        .tenant_id(100)
        .context_hash(200)
        .build();
    mock.insert("mock_key", entry.clone());

    let mock_result = mock.load("mock_key", 100).await;
    assert!(mock_result.is_some());
    assert_eq!(mock_result.unwrap().tenant_id, 100);

    let nvme = NvmeStorageLoader::new(temp_dir.path().to_path_buf());
    let nvme_result = nvme.load("mock_key", 100).await;
    assert!(nvme_result.is_none());
}

#[tokio::test]
async fn test_nvme_storage_loader_batch_operations() {
    use reflex::storage::mmap::AlignedMmapBuilder;

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let loader = NvmeStorageLoader::new(temp_dir.path().to_path_buf());

    let entry_count = 10;
    let mut entries = Vec::new();

    for i in 0..entry_count {
        let entry = CacheEntryBuilder::new()
            .tenant_id(i as u64)
            .context_hash(i as u64 * 100)
            .timestamp(1702500000 + i as i64)
            .build();

        let serialized = to_bytes::<Error>(&entry)
            .expect("Serialization should succeed")
            .into_vec();
        let file_path = temp_dir.path().join(format!("entry_{}.rkyv", i));
        let builder = AlignedMmapBuilder::new(&file_path);
        builder.write(&serialized).expect("Write should succeed");

        entries.push(entry);
    }

    for (i, entry) in entries.iter().enumerate() {
        let storage_key = format!("entry_{}.rkyv", i);
        let loaded = loader
            .load(&storage_key, i as u64)
            .await
            .expect("Should load entry");
        assert_entries_equal(entry, &loaded);
    }
}

#[tokio::test]
async fn test_nvme_storage_loader_concurrent_reads() {
    use reflex::storage::mmap::AlignedMmapBuilder;
    use std::sync::Arc;

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_path = temp_dir.path().join("concurrent_test.rkyv");

    let entry = CacheEntryBuilder::new()
        .tenant_id(999)
        .with_realistic_embedding()
        .build();

    let serialized = to_bytes::<Error>(&entry)
        .expect("Serialization should succeed")
        .into_vec();
    let builder = AlignedMmapBuilder::new(&file_path);
    builder.write(&serialized).expect("Write should succeed");

    let loader = Arc::new(NvmeStorageLoader::new(temp_dir.path().to_path_buf()));

    let reader_count = 20;
    let handles: Vec<_> = (0..reader_count)
        .map(|_| {
            let loader = Arc::clone(&loader);
            tokio::spawn(async move { loader.load("concurrent_test.rkyv", 999).await })
        })
        .collect();

    for handle in handles {
        let result = handle.await.expect("Task should complete");
        let loaded = result.expect("Load should succeed");
        assert_eq!(loaded.tenant_id, 999);
    }
}

#[test]
fn test_nvme_storage_loader_path_accessor() {
    let path = PathBuf::from("/mnt/nvme/reflex_data");
    let loader = NvmeStorageLoader::new(path.clone());

    assert_eq!(loader.storage_path(), path.as_path());
}

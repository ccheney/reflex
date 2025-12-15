use super::*;
use crate::storage::ArchivedCacheEntry;
use tempfile::TempDir;

fn create_test_storage() -> (NvmeStorage, TempDir) {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let storage = NvmeStorage::new(dir.path().to_path_buf());
    (storage, dir)
}

fn create_test_entry(tenant_id: u64) -> CacheEntry {
    CacheEntry {
        tenant_id,
        context_hash: 67890,
        timestamp: 1702500000,
        embedding: (0..crate::constants::EMBEDDING_F16_BYTES)
            .map(|i| (i % 256) as u8)
            .collect(),
        payload_blob: b"test payload data".to_vec(),
    }
}

#[test]
fn test_store_and_load() {
    let (storage, _dir) = create_test_storage();

    let entry = create_test_entry(12345);
    let entry_id = 1;

    let handle = storage
        .store(entry_id, &entry)
        .expect("Failed to store entry");

    assert!(storage.exists(entry_id, entry.tenant_id));

    let archived = handle
        .access_archived::<ArchivedCacheEntry>()
        .expect("Failed to access archived");

    assert_eq!(archived.tenant_id, entry.tenant_id);
    assert_eq!(archived.context_hash, entry.context_hash);
    assert_eq!(archived.timestamp, entry.timestamp);
    assert_eq!(
        archived.embedding.len(),
        crate::constants::EMBEDDING_F16_BYTES
    );
    assert_eq!(archived.payload_blob.as_slice(), b"test payload data");
}

#[test]
fn test_load_existing() {
    let (storage, _dir) = create_test_storage();

    let entry = create_test_entry(12345);
    let entry_id = 42;

    storage.store(entry_id, &entry).expect("Failed to store");

    let handle = storage
        .load(entry_id, entry.tenant_id)
        .expect("Failed to load");

    let archived = handle
        .access_archived::<ArchivedCacheEntry>()
        .expect("Failed to access archived");

    assert_eq!(archived.tenant_id, entry.tenant_id);
}

#[test]
fn test_load_nonexistent() {
    let (storage, _dir) = create_test_storage();

    let result = storage.load(999, 12345);

    assert!(matches!(result, Err(NvmeError::NotFound { .. })));
}

#[test]
fn test_delete() {
    let (storage, _dir) = create_test_storage();

    let entry = create_test_entry(12345);
    let entry_id = 1;

    storage.store(entry_id, &entry).expect("Failed to store");
    assert!(storage.exists(entry_id, entry.tenant_id));

    storage
        .delete(entry_id, entry.tenant_id)
        .expect("Failed to delete");
    assert!(!storage.exists(entry_id, entry.tenant_id));

    let result = storage.load(entry_id, entry.tenant_id);
    assert!(matches!(result, Err(NvmeError::NotFound { .. })));
}

#[test]
fn test_delete_nonexistent() {
    let (storage, _dir) = create_test_storage();

    let result = storage.delete(999, 12345);

    assert!(matches!(result, Err(NvmeError::NotFound { .. })));
}

#[test]
fn test_multiple_tenants() {
    let (storage, _dir) = create_test_storage();

    let tenant1 = 11111;
    let tenant2 = 22222;
    let tenant3 = 33333;

    let entry1 = create_test_entry(tenant1);
    let entry2 = create_test_entry(tenant2);
    let entry3 = create_test_entry(tenant3);

    storage.store(1, &entry1).expect("Failed to store t1e1");
    storage.store(2, &entry1).expect("Failed to store t1e2");
    storage.store(1, &entry2).expect("Failed to store t2e1");
    storage.store(1, &entry3).expect("Failed to store t3e1");

    assert!(storage.exists(1, tenant1));
    assert!(storage.exists(2, tenant1));
    assert!(storage.exists(1, tenant2));
    assert!(storage.exists(1, tenant3));

    assert!(!storage.exists(1, 99999));
    assert!(!storage.exists(2, tenant2));
}

#[test]
fn test_list_entries() {
    let (storage, _dir) = create_test_storage();

    let tenant_id = 12345;
    let entry = create_test_entry(tenant_id);

    storage.store(1, &entry).expect("Failed to store");
    storage.store(5, &entry).expect("Failed to store");
    storage.store(10, &entry).expect("Failed to store");

    let entries = storage.list_entries(tenant_id).expect("Failed to list");

    assert_eq!(entries.len(), 3);
    assert!(entries.contains(&1));
    assert!(entries.contains(&5));
    assert!(entries.contains(&10));
}

#[test]
fn test_list_entries_empty_tenant() {
    let (storage, _dir) = create_test_storage();

    let entries = storage.list_entries(99999).expect("Failed to list");

    assert!(entries.is_empty());
}

#[test]
fn test_list_tenants() {
    let (storage, _dir) = create_test_storage();

    let entry1 = create_test_entry(11111);
    let entry2 = create_test_entry(22222);

    storage.store(1, &entry1).expect("Failed to store");
    storage.store(1, &entry2).expect("Failed to store");

    let tenants = storage.list_tenants().expect("Failed to list");

    assert_eq!(tenants.len(), 2);
    assert!(tenants.contains(&11111));
    assert!(tenants.contains(&22222));
}

#[test]
fn test_ensure_storage_path() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let nested_path = dir.path().join("level1").join("level2").join("storage");

    let storage = NvmeStorage::new(nested_path.clone());

    assert!(!nested_path.exists());

    storage
        .ensure_storage_path()
        .expect("Failed to ensure path");

    assert!(nested_path.exists());
}

#[test]
fn test_cleanup_empty_tenant_dirs() {
    let (storage, _dir) = create_test_storage();

    let entry1 = create_test_entry(11111);
    let entry2 = create_test_entry(22222);

    storage.store(1, &entry1).expect("Failed to store");
    storage.store(1, &entry2).expect("Failed to store");

    storage.delete(1, 11111).expect("Failed to delete");

    let removed = storage
        .cleanup_empty_tenant_dirs()
        .expect("Failed to cleanup");

    assert_eq!(removed, 1);

    let tenants = storage.list_tenants().expect("Failed to list");
    assert_eq!(tenants.len(), 1);
    assert!(tenants.contains(&22222));
}

#[test]
fn test_stats() {
    let (storage, _dir) = create_test_storage();

    let stats = storage.stats().expect("Failed to get stats");
    assert_eq!(stats.tenant_count, 0);
    assert_eq!(stats.entry_count, 0);
    assert_eq!(stats.total_bytes, 0);

    let entry1 = create_test_entry(11111);
    let entry2 = create_test_entry(22222);

    storage.store(1, &entry1).expect("Failed to store");
    storage.store(2, &entry1).expect("Failed to store");
    storage.store(1, &entry2).expect("Failed to store");

    let stats = storage.stats().expect("Failed to get stats");
    assert_eq!(stats.tenant_count, 2);
    assert_eq!(stats.entry_count, 3);
    assert!(stats.total_bytes > 0);
}

#[test]
fn test_overwrite_existing_entry() {
    let (storage, _dir) = create_test_storage();

    let tenant_id = 12345;
    let entry_id = 1;

    let entry1 = CacheEntry {
        tenant_id,
        context_hash: 111,
        timestamp: 1000,
        embedding: vec![0x01; 100],
        payload_blob: b"first".to_vec(),
    };

    let entry2 = CacheEntry {
        tenant_id,
        context_hash: 222,
        timestamp: 2000,
        embedding: vec![0x02; 100],
        payload_blob: b"second".to_vec(),
    };

    storage.store(entry_id, &entry1).expect("Failed to store");

    let handle = storage.store(entry_id, &entry2).expect("Failed to store");

    let archived = handle
        .access_archived::<ArchivedCacheEntry>()
        .expect("Failed to access");

    assert_eq!(archived.context_hash, 222);
    assert_eq!(archived.timestamp, 2000);
    assert_eq!(archived.payload_blob.as_slice(), b"second");
}

#[test]
fn test_entry_with_empty_vectors() {
    let (storage, _dir) = create_test_storage();

    let entry = CacheEntry {
        tenant_id: 12345,
        context_hash: 67890,
        timestamp: 1702500000,
        embedding: vec![],
        payload_blob: vec![],
    };

    let handle = storage.store(1, &entry).expect("Failed to store");

    let archived = handle
        .access_archived::<ArchivedCacheEntry>()
        .expect("Failed to access");

    assert!(archived.embedding.is_empty());
    assert!(archived.payload_blob.is_empty());
}

#[test]
fn test_entry_with_large_payload() {
    let (storage, _dir) = create_test_storage();

    let large_payload: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();

    let entry = CacheEntry {
        tenant_id: 12345,
        context_hash: 67890,
        timestamp: 1702500000,
        embedding: vec![0u8; crate::constants::EMBEDDING_F16_BYTES],
        payload_blob: large_payload.clone(),
    };

    let handle = storage.store(1, &entry).expect("Failed to store");

    let archived = handle
        .access_archived::<ArchivedCacheEntry>()
        .expect("Failed to access");

    assert_eq!(archived.payload_blob.len(), 1_000_000);
}

#[test]
fn test_boundary_entry_ids() {
    let (storage, _dir) = create_test_storage();

    let entry = create_test_entry(12345);

    storage.store(0, &entry).expect("Failed to store 0");
    storage
        .store(u64::MAX, &entry)
        .expect("Failed to store u64::MAX");

    assert!(storage.exists(0, entry.tenant_id));
    assert!(storage.exists(u64::MAX, entry.tenant_id));

    let handle0 = storage.load(0, entry.tenant_id).expect("Failed to load 0");
    let handle_max = storage
        .load(u64::MAX, entry.tenant_id)
        .expect("Failed to load max");

    assert!(handle0.access_archived::<ArchivedCacheEntry>().is_ok());
    assert!(handle_max.access_archived::<ArchivedCacheEntry>().is_ok());
}

#[test]
fn test_boundary_tenant_ids() {
    let (storage, _dir) = create_test_storage();

    let entry0 = create_test_entry(0);
    let entry_max = create_test_entry(u64::MAX);

    storage.store(1, &entry0).expect("Failed to store tenant 0");
    storage
        .store(1, &entry_max)
        .expect("Failed to store tenant max");

    assert!(storage.exists(1, 0));
    assert!(storage.exists(1, u64::MAX));
}

#[test]
fn test_multiple_handles_same_file() {
    let (storage, _dir) = create_test_storage();

    let entry = create_test_entry(12345);
    storage.store(1, &entry).expect("Failed to store");

    let handle1 = storage.load(1, entry.tenant_id).expect("Failed to load");
    let handle2 = storage.load(1, entry.tenant_id).expect("Failed to load");
    let handle3 = storage.load(1, entry.tenant_id).expect("Failed to load");

    let arch1 = handle1
        .access_archived::<ArchivedCacheEntry>()
        .expect("Failed");
    let arch2 = handle2
        .access_archived::<ArchivedCacheEntry>()
        .expect("Failed");
    let arch3 = handle3
        .access_archived::<ArchivedCacheEntry>()
        .expect("Failed");

    assert_eq!(arch1.tenant_id, arch2.tenant_id);
    assert_eq!(arch2.tenant_id, arch3.tenant_id);
}

#[test]
fn test_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let (storage, dir) = create_test_storage();
    let storage = Arc::new(storage);

    for i in 0..10 {
        let entry = create_test_entry(12345);
        storage.store(i, &entry).expect("Failed to store");
    }

    let mut handles = vec![];

    for _ in 0..4 {
        let storage_clone = Arc::clone(&storage);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                for i in 0..10 {
                    let handle = storage_clone.load(i, 12345).expect("Failed to load");
                    let _ = handle.access_archived::<ArchivedCacheEntry>();
                }
            }
        }));
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    drop(dir);
}

#[test]
fn test_list_entries_ignores_non_rkyv_files() {
    let (storage, _dir) = create_test_storage();

    let tenant_id = 12345;
    let entry = create_test_entry(tenant_id);

    storage.store(1, &entry).expect("Failed to store");

    let tenant_path = storage.storage_path().join(tenant_id.to_string());

    std::fs::write(tenant_path.join("2.txt"), b"ignored").expect("Failed to write txt");

    std::fs::write(tenant_path.join("notanumber.rkyv"), b"ignored").expect("Failed to write");

    std::fs::write(tenant_path.join("noext"), b"ignored").expect("Failed to write");

    std::fs::create_dir(tenant_path.join("subdir")).expect("Failed to create dir");

    let entries = storage.list_entries(tenant_id).expect("Failed to list");

    assert_eq!(entries.len(), 1);
    assert!(entries.contains(&1));
}

#[test]
fn test_list_tenants_ignores_non_numeric_dirs() {
    let (storage, _dir) = create_test_storage();

    let entry = create_test_entry(12345);
    storage.store(1, &entry).expect("Failed to store");

    let storage_path = storage.storage_path();
    std::fs::create_dir(storage_path.join("not_a_number")).expect("Failed to create dir");
    std::fs::create_dir(storage_path.join("abc123")).expect("Failed to create dir");

    std::fs::write(storage_path.join("99999"), b"not a dir").expect("Failed to write");

    let tenants = storage.list_tenants().expect("Failed to list");

    assert_eq!(tenants.len(), 1);
    assert!(tenants.contains(&12345));
}

#[test]
fn test_list_tenants_nonexistent_storage() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let nonexistent = dir.path().join("does_not_exist");
    let storage = NvmeStorage::new(nonexistent);

    let tenants = storage.list_tenants().expect("Failed to list");
    assert!(tenants.is_empty());
}

#[test]
fn test_cleanup_empty_dirs_nonexistent_storage() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let nonexistent = dir.path().join("does_not_exist");
    let storage = NvmeStorage::new(nonexistent);

    let removed = storage
        .cleanup_empty_tenant_dirs()
        .expect("Failed to cleanup");
    assert_eq!(removed, 0);
}

#[test]
fn test_cleanup_skips_non_empty_dirs() {
    let (storage, _dir) = create_test_storage();

    let entry = create_test_entry(12345);
    storage.store(1, &entry).expect("Failed to store");

    let storage_path = storage.storage_path();
    std::fs::create_dir(storage_path.join("99999")).expect("Failed to create dir");

    let removed = storage
        .cleanup_empty_tenant_dirs()
        .expect("Failed to cleanup");

    assert_eq!(removed, 1);

    let tenants = storage.list_tenants().expect("Failed to list");
    assert!(tenants.contains(&12345));
}

#[test]
fn test_storage_path_accessor() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let path = dir.path().to_path_buf();
    let storage = NvmeStorage::new(path.clone());

    assert_eq!(storage.storage_path(), path);
}

#[test]
fn test_storage_stats_equality() {
    let stats1 = StorageStats {
        tenant_count: 1,
        entry_count: 2,
        total_bytes: 100,
    };
    let stats2 = StorageStats {
        tenant_count: 1,
        entry_count: 2,
        total_bytes: 100,
    };
    let stats3 = StorageStats {
        tenant_count: 2,
        entry_count: 2,
        total_bytes: 100,
    };

    assert_eq!(stats1, stats2);
    assert_ne!(stats1, stats3);
}

#[test]
fn test_storage_stats_clone() {
    let stats = StorageStats {
        tenant_count: 5,
        entry_count: 10,
        total_bytes: 1000,
    };
    let cloned = stats;

    assert_eq!(stats.tenant_count, cloned.tenant_count);
    assert_eq!(stats.entry_count, cloned.entry_count);
    assert_eq!(stats.total_bytes, cloned.total_bytes);
}

#[test]
fn test_storage_stats_debug() {
    let stats = StorageStats {
        tenant_count: 1,
        entry_count: 2,
        total_bytes: 100,
    };
    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("StorageStats"));
    assert!(debug_str.contains("tenant_count"));
}

#[test]
fn test_nvme_storage_clone() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let storage = NvmeStorage::new(dir.path().to_path_buf());
    let cloned = storage.clone();

    assert_eq!(storage.storage_path(), cloned.storage_path());
}

#[test]
fn test_nvme_storage_debug() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let storage = NvmeStorage::new(dir.path().to_path_buf());
    let debug_str = format!("{:?}", storage);

    assert!(debug_str.contains("NvmeStorage"));
    assert!(debug_str.contains("storage_path"));
}

#[test]
fn test_nvme_error_display_not_found() {
    let err = NvmeError::NotFound {
        tenant_id: 123,
        entry_id: 456,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("123"));
    assert!(msg.contains("456"));
    assert!(msg.contains("not found"));
}

#[test]
fn test_nvme_error_display_storage_unavailable() {
    let err = NvmeError::StorageUnavailable {
        path: std::path::PathBuf::from("/test/path"),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("/test/path"));
    assert!(msg.contains("unavailable"));
}

#[test]
fn test_nvme_error_display_tenant_dir_creation_failed() {
    let err = NvmeError::TenantDirCreationFailed {
        path: std::path::PathBuf::from("/tenant/path"),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("/tenant/path"));
    assert!(msg.contains("tenant directory"));
}

#[test]
fn test_nvme_error_display_serialization() {
    let err = NvmeError::Serialization("test serialization error".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("test serialization error"));
}

#[test]
fn test_nvme_error_display_io() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
    let err = NvmeError::Io(io_err);
    let msg = format!("{}", err);
    assert!(msg.contains("I/O error"));
}

#[test]
fn test_nvme_error_display_mmap() {
    let mmap_err = crate::storage::mmap::MmapError::EmptyFile;
    let err = NvmeError::Mmap(mmap_err);
    let msg = format!("{}", err);
    assert!(msg.contains("mmap error"));
}

#[test]
fn test_nvme_error_debug() {
    let err = NvmeError::NotFound {
        tenant_id: 1,
        entry_id: 2,
    };
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("NotFound"));

    let err = NvmeError::StorageUnavailable {
        path: std::path::PathBuf::from("/path"),
    };
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("StorageUnavailable"));

    let err = NvmeError::TenantDirCreationFailed {
        path: std::path::PathBuf::from("/path"),
    };
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("TenantDirCreationFailed"));
}

#[test]
fn test_nvme_error_from_io() {
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "permission denied");
    let nvme_err: NvmeError = io_err.into();
    assert!(matches!(nvme_err, NvmeError::Io(_)));
}

#[test]
fn test_nvme_error_from_mmap() {
    let mmap_err = crate::storage::mmap::MmapError::EmptyFile;
    let nvme_err: NvmeError = mmap_err.into();
    assert!(matches!(nvme_err, NvmeError::Mmap(_)));
}

#[cfg(unix)]
#[test]
fn test_ensure_storage_path_permission_denied() {
    use std::os::unix::fs::PermissionsExt;

    let dir = TempDir::new().expect("Failed to create temp dir");
    let restricted_dir = dir.path().join("restricted");
    std::fs::create_dir(&restricted_dir).expect("Failed to create restricted dir");

    std::fs::set_permissions(&restricted_dir, std::fs::Permissions::from_mode(0o000))
        .expect("Failed to set permissions");

    let storage = NvmeStorage::new(restricted_dir.join("nested").join("storage"));

    let result = storage.ensure_storage_path();

    std::fs::set_permissions(&restricted_dir, std::fs::Permissions::from_mode(0o755))
        .expect("Failed to restore permissions");

    assert!(matches!(result, Err(NvmeError::StorageUnavailable { .. })));
}

#[cfg(unix)]
#[test]
fn test_store_tenant_dir_creation_fails() {
    use std::os::unix::fs::PermissionsExt;

    let dir = TempDir::new().expect("Failed to create temp dir");
    let storage_path = dir.path().join("storage");
    std::fs::create_dir(&storage_path).expect("Failed to create storage dir");

    std::fs::set_permissions(&storage_path, std::fs::Permissions::from_mode(0o444))
        .expect("Failed to set permissions");

    let storage = NvmeStorage::new(storage_path.clone());
    let entry = create_test_entry(12345);

    let result = storage.store(1, &entry);

    std::fs::set_permissions(&storage_path, std::fs::Permissions::from_mode(0o755))
        .expect("Failed to restore permissions");

    assert!(matches!(
        result,
        Err(NvmeError::TenantDirCreationFailed { .. })
    ));
}

use crate::storage::mmap::{AlignedMmapBuilder, MmapFileHandle};
use crate::storage::{CacheEntry, StorageError, StorageWriter};

pub trait StorageLoader: Send + Sync {
    fn load(
        &self,
        storage_key: &str,
        tenant_id: u64,
    ) -> impl std::future::Future<Output = Option<CacheEntry>> + Send;
}

#[cfg(any(test, feature = "mock"))]
#[derive(Default, Clone)]
pub struct MockStorageLoader {
    entries: std::sync::Arc<std::sync::RwLock<std::collections::HashMap<String, CacheEntry>>>,
}

#[cfg(any(test, feature = "mock"))]
impl MockStorageLoader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, key: &str, entry: CacheEntry) {
        self.entries
            .write()
            .expect("lock poisoned")
            .insert(key.to_string(), entry);
    }

    pub fn len(&self) -> usize {
        self.entries.read().expect("lock poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.read().expect("lock poisoned").is_empty()
    }
}

#[cfg(any(test, feature = "mock"))]
impl StorageLoader for MockStorageLoader {
    async fn load(&self, storage_key: &str, _tenant_id: u64) -> Option<CacheEntry> {
        self.entries
            .read()
            .expect("lock poisoned")
            .get(storage_key)
            .cloned()
    }
}

#[cfg(any(test, feature = "mock"))]
impl StorageWriter for MockStorageLoader {
    fn write(
        &self,
        key: &str,
        data: &[u8],
    ) -> Result<MmapFileHandle, crate::storage::StorageError> {
        use std::io::Write;

        // Create a temp file and write data to it
        let mut temp_file = tempfile::NamedTempFile::new()
            .map_err(|e| crate::storage::StorageError::WriteFailed(e.to_string()))?;
        temp_file
            .write_all(data)
            .map_err(|e| crate::storage::StorageError::WriteFailed(e.to_string()))?;
        temp_file
            .flush()
            .map_err(|e| crate::storage::StorageError::WriteFailed(e.to_string()))?;

        // Also store the entry in our mock storage for later retrieval
        if let Ok(entry) = rkyv::from_bytes::<CacheEntry, rkyv::rancor::Error>(data) {
            self.insert(key, entry);
        }

        // Open as mmap handle
        MmapFileHandle::open(temp_file.path())
            .map_err(|e| crate::storage::StorageError::WriteFailed(e.to_string()))
    }
}

#[derive(Debug, Clone)]
pub struct NvmeStorageLoader {
    storage_path: std::path::PathBuf,
}

impl NvmeStorageLoader {
    pub fn new(storage_path: std::path::PathBuf) -> Self {
        Self { storage_path }
    }

    pub fn storage_path(&self) -> &std::path::Path {
        &self.storage_path
    }
}

fn sanitize_storage_key(storage_key: &str) -> Option<std::path::PathBuf> {
    use std::path::{Component, Path};

    if storage_key.is_empty() {
        return None;
    }

    let p = Path::new(storage_key);
    let mut out = std::path::PathBuf::new();

    for c in p.components() {
        match c {
            Component::Normal(seg) => out.push(seg),
            Component::CurDir => continue,
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return None,
        }
    }

    if out.as_os_str().is_empty() {
        None
    } else {
        Some(out)
    }
}

impl StorageWriter for NvmeStorageLoader {
    fn write(&self, key: &str, data: &[u8]) -> Result<MmapFileHandle, StorageError> {
        let rel = sanitize_storage_key(key).ok_or_else(|| {
            StorageError::Io(format!("Invalid storage key (path traversal?): {}", key))
        })?;
        let file_path = self.storage_path.join(rel);

        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| StorageError::Io(format!("Failed to create directory: {}", e)))?;
        }

        let builder = AlignedMmapBuilder::new(file_path);
        builder
            .write_readonly(data)
            .map_err(|e| StorageError::WriteFailed(format!("Failed to write file: {}", e)))
    }
}

impl StorageLoader for NvmeStorageLoader {
    async fn load(&self, storage_key: &str, tenant_id: u64) -> Option<CacheEntry> {
        use crate::storage::mmap::MmapFileHandle;
        use rkyv::from_bytes;
        use rkyv::rancor::Error;

        let storage_path = self.storage_path.clone();
        let storage_key = storage_key.to_string();

        tokio::task::spawn_blocking(move || {
            let rel = match sanitize_storage_key(&storage_key) {
                Some(r) => r,
                None => {
                    tracing::warn!(
                        storage_key = %storage_key,
                        "Rejected invalid storage_key (path traversal?)"
                    );
                    return None;
                }
            };

            let file_path = storage_path.join(rel);
            let handle = match MmapFileHandle::open(&file_path) {
                Ok(h) => h,
                Err(_) => return None,
            };
            let bytes = handle.as_slice();

            let entry: CacheEntry = match from_bytes::<CacheEntry, Error>(bytes) {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!(
                        "Failed to deserialize cache entry at {:?}: {}",
                        file_path,
                        e
                    );
                    return None;
                }
            };

            if entry.tenant_id != tenant_id {
                tracing::warn!(
                    "Tenant ID mismatch for key {}: expected {}, found {}",
                    storage_key,
                    tenant_id,
                    entry.tenant_id
                );
                return None;
            }

            Some(entry)
        })
        .await
        .ok()
        .flatten()
    }
}

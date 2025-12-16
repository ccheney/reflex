//! NVMe-backed storage (simple file-per-entry layout).

/// NVMe backend error types.
pub mod error;

#[cfg(test)]
mod tests;

pub use error::{NvmeError, NvmeResult};

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use rkyv::rancor::Error as RkyvError;
use rkyv::to_bytes;

use crate::storage::CacheEntry;
use crate::storage::mmap::MmapFileHandle;

const RKYV_EXTENSION: &str = "rkyv";

const TEMP_EXTENSION: &str = "rkyv.tmp";

#[derive(Debug, Clone)]
/// Stores and retrieves [`CacheEntry`] records on disk.
pub struct NvmeStorage {
    storage_path: PathBuf,
}

impl NvmeStorage {
    /// Creates a storage rooted at `storage_path`.
    pub fn new(storage_path: PathBuf) -> Self {
        Self { storage_path }
    }

    /// Returns the root storage directory.
    pub fn storage_path(&self) -> &Path {
        &self.storage_path
    }

    /// Ensures the root storage directory exists.
    pub fn ensure_storage_path(&self) -> NvmeResult<()> {
        if !self.storage_path.exists() {
            fs::create_dir_all(&self.storage_path).map_err(|_| NvmeError::StorageUnavailable {
                path: self.storage_path.clone(),
            })?;
        }
        Ok(())
    }

    fn tenant_path(&self, tenant_id: u64) -> PathBuf {
        self.storage_path.join(tenant_id.to_string())
    }

    fn entry_path(&self, tenant_id: u64, entry_id: u64) -> PathBuf {
        self.tenant_path(tenant_id)
            .join(format!("{}.{}", entry_id, RKYV_EXTENSION))
    }

    fn temp_entry_path(&self, tenant_id: u64, entry_id: u64) -> PathBuf {
        self.tenant_path(tenant_id)
            .join(format!("{}.{}", entry_id, TEMP_EXTENSION))
    }

    fn ensure_tenant_dir(&self, tenant_id: u64) -> NvmeResult<()> {
        let tenant_path = self.tenant_path(tenant_id);
        if !tenant_path.exists() {
            fs::create_dir_all(&tenant_path)
                .map_err(|_| NvmeError::TenantDirCreationFailed { path: tenant_path })?;
        }
        Ok(())
    }

    /// Writes `entry` under `(tenant_id, id)` and returns an mmap handle to the bytes.
    pub fn store(&self, id: u64, entry: &CacheEntry) -> NvmeResult<MmapFileHandle> {
        let tenant_id = entry.tenant_id;

        self.ensure_storage_path()?;
        self.ensure_tenant_dir(tenant_id)?;

        let bytes = to_bytes::<RkyvError>(entry)
            .map_err(|e| NvmeError::Serialization(format!("{:?}", e)))?;

        let temp_path = self.temp_entry_path(tenant_id, id);
        let final_path = self.entry_path(tenant_id, id);

        {
            let mut file = File::create(&temp_path)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }

        fs::rename(&temp_path, &final_path)?;

        let handle = MmapFileHandle::open(&final_path)?;
        Ok(handle)
    }

    /// Loads the `(tenant_id, id)` entry as an mmap handle.
    pub fn load(&self, id: u64, tenant_id: u64) -> NvmeResult<MmapFileHandle> {
        let path = self.entry_path(tenant_id, id);

        if !path.exists() {
            return Err(NvmeError::NotFound {
                tenant_id,
                entry_id: id,
            });
        }

        let handle = MmapFileHandle::open(&path)?;
        Ok(handle)
    }

    /// Deletes the `(tenant_id, id)` entry.
    pub fn delete(&self, id: u64, tenant_id: u64) -> NvmeResult<()> {
        let path = self.entry_path(tenant_id, id);

        if !path.exists() {
            return Err(NvmeError::NotFound {
                tenant_id,
                entry_id: id,
            });
        }

        fs::remove_file(&path)?;
        Ok(())
    }

    /// Returns `true` if `(tenant_id, id)` exists.
    pub fn exists(&self, id: u64, tenant_id: u64) -> bool {
        self.entry_path(tenant_id, id).exists()
    }

    /// Lists entry ids for `tenant_id`.
    pub fn list_entries(&self, tenant_id: u64) -> NvmeResult<Vec<u64>> {
        let tenant_path = self.tenant_path(tenant_id);

        if !tenant_path.exists() {
            return Ok(Vec::new());
        }

        let mut entries = Vec::new();

        for entry in fs::read_dir(&tenant_path)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(ext) = path.extension()
                && ext == RKYV_EXTENSION
                && let Some(stem) = path.file_stem()
                && let Some(stem_str) = stem.to_str()
                && let Ok(id) = stem_str.parse::<u64>()
            {
                entries.push(id);
            }
        }

        Ok(entries)
    }

    /// Lists tenant ids currently present under the root directory.
    pub fn list_tenants(&self) -> NvmeResult<Vec<u64>> {
        if !self.storage_path.exists() {
            return Ok(Vec::new());
        }

        let mut tenants = Vec::new();

        for entry in fs::read_dir(&self.storage_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir()
                && let Some(name) = path.file_name()
                && let Some(name_str) = name.to_str()
                && let Ok(id) = name_str.parse::<u64>()
            {
                tenants.push(id);
            }
        }

        Ok(tenants)
    }

    /// Removes empty tenant directories and returns the count removed.
    pub fn cleanup_empty_tenant_dirs(&self) -> NvmeResult<usize> {
        if !self.storage_path.exists() {
            return Ok(0);
        }

        let mut removed = 0;

        for entry in fs::read_dir(&self.storage_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                let is_empty = fs::read_dir(&path)?.next().is_none();

                if is_empty {
                    fs::remove_dir(&path)?;
                    removed += 1;
                }
            }
        }

        Ok(removed)
    }

    /// Returns basic storage stats by scanning the directory tree.
    pub fn stats(&self) -> NvmeResult<StorageStats> {
        let tenants = self.list_tenants()?;
        let mut total_entries = 0;
        let mut total_bytes = 0;

        for tenant_id in &tenants {
            let entries = self.list_entries(*tenant_id)?;
            total_entries += entries.len();

            for entry_id in entries {
                let path = self.entry_path(*tenant_id, entry_id);
                if let Ok(metadata) = fs::metadata(&path) {
                    total_bytes += metadata.len();
                }
            }
        }

        Ok(StorageStats {
            tenant_count: tenants.len(),
            entry_count: total_entries,
            total_bytes,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Aggregate stats for the NVMe storage directory.
pub struct StorageStats {
    /// Number of tenant directories.
    pub tenant_count: usize,
    /// Total number of entry files.
    pub entry_count: usize,
    /// Total bytes across all entry files.
    pub total_bytes: u64,
}

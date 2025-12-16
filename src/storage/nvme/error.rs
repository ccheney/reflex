use std::path::PathBuf;
use thiserror::Error;

use crate::storage::mmap::MmapError;

#[derive(Error, Debug)]
/// Errors returned by the NVMe storage backend.
pub enum NvmeError {
    /// IO error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Mmap error.
    #[error("mmap error: {0}")]
    Mmap(#[from] MmapError),

    /// Serialization/deserialization error.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Entry was not found.
    #[error("entry not found: tenant={tenant_id}, id={entry_id}")]
    NotFound {
        /// Tenant id.
        tenant_id: u64,
        /// Entry id.
        entry_id: u64,
    },

    /// Storage root path is missing/unavailable.
    #[error("storage path unavailable: {path}")]
    StorageUnavailable {
        /// Path that was unavailable.
        path: PathBuf,
    },

    /// Failed to create the tenant directory.
    #[error("failed to create tenant directory: {path}")]
    TenantDirCreationFailed {
        /// Directory path.
        path: PathBuf,
    },
}

/// Convenience result type for NVMe storage operations.
pub type NvmeResult<T> = Result<T, NvmeError>;

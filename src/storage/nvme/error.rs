use std::path::PathBuf;
use thiserror::Error;

use crate::storage::mmap::MmapError;

#[derive(Error, Debug)]
pub enum NvmeError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("mmap error: {0}")]
    Mmap(#[from] MmapError),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("entry not found: tenant={tenant_id}, id={entry_id}")]
    NotFound { tenant_id: u64, entry_id: u64 },

    #[error("storage path unavailable: {path}")]
    StorageUnavailable { path: PathBuf },

    #[error("failed to create tenant directory: {path}")]
    TenantDirCreationFailed { path: PathBuf },
}

pub type NvmeResult<T> = Result<T, NvmeError>;

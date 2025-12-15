use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LifecycleError {
    #[error("Cloud operation failed: {0}")]
    CloudError(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("snapshot not found at {path}")]
    SnapshotNotFound { path: PathBuf },

    #[error("GCS bucket not configured (set REFLEX_GCS_BUCKET)")]
    GcsBucketNotConfigured,
}

pub type LifecycleResult<T> = Result<T, LifecycleError>;

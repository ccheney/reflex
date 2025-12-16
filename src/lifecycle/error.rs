use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
/// Errors returned by lifecycle operations.
pub enum LifecycleError {
    /// Cloud CLI/API error.
    #[error("Cloud operation failed: {0}")]
    CloudError(String),

    /// IO error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration error.
    #[error("configuration error: {0}")]
    Config(String),

    /// Snapshot file was not found.
    #[error("snapshot not found at {path}")]
    SnapshotNotFound {
        /// Missing path.
        path: PathBuf,
    },

    /// GCS bucket is required for this operation.
    #[error("GCS bucket not configured (set REFLEX_GCS_BUCKET)")]
    GcsBucketNotConfigured,
}

/// Convenience result type for lifecycle operations.
pub type LifecycleResult<T> = Result<T, LifecycleError>;

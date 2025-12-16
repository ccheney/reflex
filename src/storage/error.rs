use thiserror::Error;

#[derive(Error, Debug)]
/// Errors returned by storage operations.
pub enum StorageError {
    /// Generic IO failure.
    #[error("I/O error: {0}")]
    Io(String),

    /// Write failed.
    #[error("write failed: {0}")]
    WriteFailed(String),
}

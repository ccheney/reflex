use thiserror::Error;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("I/O error: {0}")]
    Io(String),

    #[error("write failed: {0}")]
    WriteFailed(String),
}

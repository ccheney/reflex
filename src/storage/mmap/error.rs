use std::io;
use thiserror::Error;

#[derive(Error, Debug)]
/// Errors returned by mmap operations.
pub enum MmapError {
    /// Underlying IO error.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// File was smaller than required.
    #[error("File size {actual} is smaller than minimum {expected}")]
    FileTooSmall {
        /// Expected minimum size (bytes).
        expected: usize,
        /// Actual size (bytes).
        actual: usize,
    },

    /// Cannot map an empty file.
    #[error("Cannot mmap empty file")]
    EmptyFile,

    /// `rkyv` validation failed.
    #[error("rkyv validation failed: {0}")]
    ValidationFailed(String),

    /// The requested offset was not aligned for `rkyv` access.
    #[error("Data at offset {offset} is not aligned to {alignment} bytes")]
    AlignmentError {
        /// Requested offset (bytes).
        offset: usize,
        /// Required alignment (bytes).
        alignment: usize,
    },

    /// File resize failed (read-only map, invalid size, or OS error).
    #[error("Failed to resize file: {0}")]
    ResizeFailed(String),
}

/// Convenience result type for mmap operations.
pub type MmapResult<T> = Result<T, MmapError>;

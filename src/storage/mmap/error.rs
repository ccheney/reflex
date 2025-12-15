use std::io;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MmapError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("File size {actual} is smaller than minimum {expected}")]
    FileTooSmall { expected: usize, actual: usize },

    #[error("Cannot mmap empty file")]
    EmptyFile,

    #[error("rkyv validation failed: {0}")]
    ValidationFailed(String),

    #[error("Data at offset {offset} is not aligned to {alignment} bytes")]
    AlignmentError { offset: usize, alignment: usize },

    #[error("Failed to resize file: {0}")]
    ResizeFailed(String),
}

pub type MmapResult<T> = Result<T, MmapError>;

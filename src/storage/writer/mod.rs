use crate::storage::error::StorageError;
use crate::storage::mmap::MmapFileHandle;

/// Writes opaque bytes to storage and returns a readable mmap handle.
pub trait StorageWriter: Send + Sync {
    /// Writes `data` under `key`.
    fn write(&self, key: &str, data: &[u8]) -> Result<MmapFileHandle, StorageError>;
}

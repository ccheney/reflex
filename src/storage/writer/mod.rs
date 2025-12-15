use crate::storage::error::StorageError;
use crate::storage::mmap::MmapFileHandle;

pub trait StorageWriter: Send + Sync {
    fn write(&self, key: &str, data: &[u8]) -> Result<MmapFileHandle, StorageError>;
}

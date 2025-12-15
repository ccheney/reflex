//! Storage primitives (models and mmap helpers).

pub mod error;
pub mod mmap;
mod model;
pub mod nvme;
pub mod writer;

pub use error::StorageError;
pub use model::ArchivedCacheEntry;
pub use model::CacheEntry;
pub use writer::StorageWriter;

//! Storage primitives.
//!
//! - [`CacheEntry`] is the on-disk record.
//! - [`mmap`] provides aligned memory-mapped IO helpers used by caches.

/// Storage error types.
pub mod error;
/// Memory-mapped IO helpers.
pub mod mmap;
mod model;
/// NVMe-backed storage implementation.
pub mod nvme;
/// Storage writer trait.
pub mod writer;

pub use error::StorageError;
pub use model::ArchivedCacheEntry;
pub use model::CacheEntry;
pub use writer::StorageWriter;

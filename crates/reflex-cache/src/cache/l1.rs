//! L1 exact-match cache (in-memory).
//!
//! L1 uses a BLAKE3 hash of the prompt (plus tenant prefix) as the key and stores an
//! [`crate::storage::mmap::MmapFileHandle`] to the persisted entry payload.

use moka::sync::Cache;
use std::sync::Arc;

use super::types::ReflexStatus;
use crate::hashing::hash_prompt;
use crate::storage::mmap::MmapFileHandle;

/// Result of an L1 lookup (exact hash match).
#[derive(Debug, Clone)]
pub struct L1LookupResult {
    handle: MmapFileHandle,
    hash: [u8; 32],
}

impl L1LookupResult {
    /// Returns the [`ReflexStatus`] for this lookup result.
    #[inline]
    pub fn status(&self) -> ReflexStatus {
        ReflexStatus::HitL1Exact
    }

    /// Returns the underlying mmap handle.
    #[inline]
    pub fn handle(&self) -> &MmapFileHandle {
        &self.handle
    }

    /// Consumes the result and returns the underlying mmap handle.
    #[inline]
    pub fn into_handle(self) -> MmapFileHandle {
        self.handle
    }

    /// Returns the 32-byte BLAKE3 prompt hash used as the key.
    #[inline]
    pub fn hash(&self) -> &[u8; 32] {
        &self.hash
    }

    /// Returns the raw bytes of the mmap'd payload.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.handle.as_slice()
    }
}

/// In-memory exact-match cache keyed by prompt hash.
pub struct L1Cache {
    entries: Cache<[u8; 32], MmapFileHandle>,
}

impl L1Cache {
    const DEFAULT_CAPACITY: u64 = 10_000;

    /// Creates a cache with the default capacity.
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }

    /// Creates a cache with a max entry capacity (LRU eviction).
    #[inline]
    pub fn with_capacity(capacity: u64) -> Self {
        Self {
            entries: Cache::builder().max_capacity(capacity).build(),
        }
    }

    /// Looks up a prompt by hashing it with [`hash_prompt`].
    #[inline]
    pub fn lookup(&self, prompt: &str) -> Option<L1LookupResult> {
        let hash = hash_prompt(prompt);
        self.lookup_by_hash(&hash)
    }

    /// Looks up an entry by a precomputed 32-byte hash.
    #[inline]
    pub fn lookup_by_hash(&self, hash: &[u8; 32]) -> Option<L1LookupResult> {
        self.entries.get(hash).map(|handle| L1LookupResult {
            handle,
            hash: *hash,
        })
    }

    /// Inserts a prompt → handle mapping and returns the computed hash.
    #[inline]
    pub fn insert(&self, prompt: &str, handle: MmapFileHandle) -> [u8; 32] {
        let hash = hash_prompt(prompt);
        self.entries.insert(hash, handle);
        hash
    }

    /// Inserts a precomputed hash → handle mapping.
    #[inline]
    pub fn insert_by_hash(&self, hash: [u8; 32], handle: MmapFileHandle) {
        self.entries.insert(hash, handle);
    }

    /// Removes an entry by hash.
    #[inline]
    pub fn remove(&self, hash: &[u8; 32]) -> Option<MmapFileHandle> {
        self.entries.remove(hash)
    }

    /// Removes an entry by prompt (hashing it first).
    #[inline]
    pub fn remove_prompt(&self, prompt: &str) -> Option<MmapFileHandle> {
        let hash = hash_prompt(prompt);
        self.remove(&hash)
    }

    /// Returns the number of cached entries.
    #[inline]
    pub fn len(&self) -> u64 {
        self.entries.entry_count()
    }

    /// Returns `true` if the cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.entry_count() == 0
    }

    /// Clears all entries.
    #[inline]
    pub fn clear(&self) {
        self.entries.invalidate_all();
    }

    /// Returns `true` if the cache contains the given hash.
    #[inline]
    pub fn contains_hash(&self, hash: &[u8; 32]) -> bool {
        self.entries.contains_key(hash)
    }

    /// Returns `true` if the cache contains the given prompt.
    #[inline]
    pub fn contains_prompt(&self, prompt: &str) -> bool {
        let hash = hash_prompt(prompt);
        self.contains_hash(&hash)
    }

    /// Runs any pending maintenance tasks in the underlying cache.
    #[inline]
    pub fn run_pending_tasks(&self) {
        self.entries.run_pending_tasks();
    }

    /// Returns an iterator of currently stored hashes.
    pub fn hashes(&self) -> impl Iterator<Item = [u8; 32]> {
        self.entries.iter().map(|(k, _)| *k)
    }
}

impl Default for L1Cache {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for L1Cache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("L1Cache")
            .field("entries", &self.entries.entry_count())
            .finish()
    }
}

#[derive(Clone)]
/// Shared handle to an [`L1Cache`].
pub struct L1CacheHandle {
    inner: Arc<L1Cache>,
}

impl L1CacheHandle {
    /// Creates a new handle with default capacity.
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(L1Cache::new()),
        }
    }

    /// Creates a new handle with a specific capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Arc::new(L1Cache::with_capacity(capacity as u64)),
        }
    }

    /// Looks up a prompt.
    #[inline]
    pub fn lookup(&self, prompt: &str) -> Option<L1LookupResult> {
        self.inner.lookup(prompt)
    }

    /// Looks up by precomputed hash.
    #[inline]
    pub fn lookup_by_hash(&self, hash: &[u8; 32]) -> Option<L1LookupResult> {
        self.inner.lookup_by_hash(hash)
    }

    /// Inserts a prompt → handle mapping and returns the computed hash.
    #[inline]
    pub fn insert(&self, prompt: &str, handle: MmapFileHandle) -> [u8; 32] {
        self.inner.insert(prompt, handle)
    }

    /// Inserts a hash → handle mapping.
    #[inline]
    pub fn insert_by_hash(&self, hash: [u8; 32], handle: MmapFileHandle) {
        self.inner.insert_by_hash(hash, handle)
    }

    /// Removes an entry by hash.
    #[inline]
    pub fn remove(&self, hash: &[u8; 32]) -> Option<MmapFileHandle> {
        self.inner.remove(hash)
    }

    /// Removes an entry by prompt.
    #[inline]
    pub fn remove_prompt(&self, prompt: &str) -> Option<MmapFileHandle> {
        self.inner.remove_prompt(prompt)
    }

    /// Returns the number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len() as usize
    }

    /// Returns `true` if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clears all entries.
    #[inline]
    pub fn clear(&self) {
        self.inner.clear();
    }

    /// Returns `true` if the cache contains the given hash.
    #[inline]
    pub fn contains_hash(&self, hash: &[u8; 32]) -> bool {
        self.inner.contains_hash(hash)
    }

    /// Returns `true` if the cache contains the given prompt.
    #[inline]
    pub fn contains_prompt(&self, prompt: &str) -> bool {
        self.inner.contains_prompt(prompt)
    }

    /// Runs any pending maintenance tasks.
    #[inline]
    pub fn run_pending_tasks(&self) {
        self.inner.run_pending_tasks();
    }

    /// Returns the number of strong references to the underlying cache.
    #[inline]
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }
}

impl Default for L1CacheHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for L1CacheHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("L1CacheHandle")
            .field("strong_count", &self.strong_count())
            .finish()
    }
}

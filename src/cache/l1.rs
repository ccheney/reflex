use moka::sync::Cache;
use std::sync::Arc;

use super::types::ReflexStatus;
use crate::hashing::hash_prompt;
use crate::storage::mmap::MmapFileHandle;

#[derive(Debug, Clone)]
pub struct L1LookupResult {
    handle: MmapFileHandle,
    hash: [u8; 32],
}

impl L1LookupResult {
    #[inline]
    pub fn status(&self) -> ReflexStatus {
        ReflexStatus::HitL1Exact
    }

    #[inline]
    pub fn handle(&self) -> &MmapFileHandle {
        &self.handle
    }

    #[inline]
    pub fn into_handle(self) -> MmapFileHandle {
        self.handle
    }

    #[inline]
    pub fn hash(&self) -> &[u8; 32] {
        &self.hash
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.handle.as_slice()
    }
}

pub struct L1Cache {
    entries: Cache<[u8; 32], MmapFileHandle>,
}

impl L1Cache {
    const DEFAULT_CAPACITY: u64 = 10_000;

    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }

    #[inline]
    pub fn with_capacity(capacity: u64) -> Self {
        Self {
            entries: Cache::builder().max_capacity(capacity).build(),
        }
    }

    #[inline]
    pub fn lookup(&self, prompt: &str) -> Option<L1LookupResult> {
        let hash = hash_prompt(prompt);
        self.lookup_by_hash(&hash)
    }

    #[inline]
    pub fn lookup_by_hash(&self, hash: &[u8; 32]) -> Option<L1LookupResult> {
        self.entries.get(hash).map(|handle| L1LookupResult {
            handle,
            hash: *hash,
        })
    }

    #[inline]
    pub fn insert(&self, prompt: &str, handle: MmapFileHandle) -> [u8; 32] {
        let hash = hash_prompt(prompt);
        self.entries.insert(hash, handle);
        hash
    }

    #[inline]
    pub fn insert_by_hash(&self, hash: [u8; 32], handle: MmapFileHandle) {
        self.entries.insert(hash, handle);
    }

    #[inline]
    pub fn remove(&self, hash: &[u8; 32]) -> Option<MmapFileHandle> {
        self.entries.remove(hash)
    }

    #[inline]
    pub fn remove_prompt(&self, prompt: &str) -> Option<MmapFileHandle> {
        let hash = hash_prompt(prompt);
        self.remove(&hash)
    }

    #[inline]
    pub fn len(&self) -> u64 {
        self.entries.entry_count()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.entry_count() == 0
    }

    #[inline]
    pub fn clear(&self) {
        self.entries.invalidate_all();
    }

    #[inline]
    pub fn contains_hash(&self, hash: &[u8; 32]) -> bool {
        self.entries.contains_key(hash)
    }

    #[inline]
    pub fn contains_prompt(&self, prompt: &str) -> bool {
        let hash = hash_prompt(prompt);
        self.contains_hash(&hash)
    }

    #[inline]
    pub fn run_pending_tasks(&self) {
        self.entries.run_pending_tasks();
    }

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
pub struct L1CacheHandle {
    inner: Arc<L1Cache>,
}

impl L1CacheHandle {
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(L1Cache::new()),
        }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Arc::new(L1Cache::with_capacity(capacity as u64)),
        }
    }

    #[inline]
    pub fn lookup(&self, prompt: &str) -> Option<L1LookupResult> {
        self.inner.lookup(prompt)
    }

    #[inline]
    pub fn lookup_by_hash(&self, hash: &[u8; 32]) -> Option<L1LookupResult> {
        self.inner.lookup_by_hash(hash)
    }

    #[inline]
    pub fn insert(&self, prompt: &str, handle: MmapFileHandle) -> [u8; 32] {
        self.inner.insert(prompt, handle)
    }

    #[inline]
    pub fn insert_by_hash(&self, hash: [u8; 32], handle: MmapFileHandle) {
        self.inner.insert_by_hash(hash, handle)
    }

    #[inline]
    pub fn remove(&self, hash: &[u8; 32]) -> Option<MmapFileHandle> {
        self.inner.remove(hash)
    }

    #[inline]
    pub fn remove_prompt(&self, prompt: &str) -> Option<MmapFileHandle> {
        self.inner.remove_prompt(prompt)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len() as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[inline]
    pub fn clear(&self) {
        self.inner.clear();
    }

    #[inline]
    pub fn contains_hash(&self, hash: &[u8; 32]) -> bool {
        self.inner.contains_hash(hash)
    }

    #[inline]
    pub fn contains_prompt(&self, prompt: &str) -> bool {
        self.inner.contains_prompt(prompt)
    }

    #[inline]
    pub fn run_pending_tasks(&self) {
        self.inner.run_pending_tasks();
    }

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

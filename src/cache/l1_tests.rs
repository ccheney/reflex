use super::l1::{L1Cache, L1CacheHandle};
use super::types::ReflexStatus;
use crate::hashing::hash_prompt;
use crate::storage::mmap::MmapFileHandle;
use std::io::Write;
use tempfile::NamedTempFile;

fn create_test_handle(content: &[u8]) -> (NamedTempFile, MmapFileHandle) {
    let mut file = NamedTempFile::new().expect("Failed to create temp file");
    file.write_all(content)
        .expect("Failed to write to temp file");
    file.flush().expect("Failed to flush temp file");

    let handle = MmapFileHandle::open(file.path()).expect("Failed to create MmapFileHandle");

    (file, handle)
}

#[test]
fn test_reflex_status_header_values() {
    assert_eq!(ReflexStatus::HitL1Exact.as_header_value(), "HIT_L1_EXACT");
    assert_eq!(
        ReflexStatus::HitL2Semantic.as_header_value(),
        "HIT_L2_SEMANTIC"
    );
    assert_eq!(
        ReflexStatus::HitL3Verified.as_header_value(),
        "HIT_L3_VERIFIED"
    );
    assert_eq!(ReflexStatus::Miss.as_header_value(), "MISS");
}

#[test]
fn test_reflex_status_is_hit() {
    assert!(ReflexStatus::HitL1Exact.is_hit());
    assert!(ReflexStatus::HitL2Semantic.is_hit());
    assert!(ReflexStatus::HitL3Verified.is_hit());
    assert!(!ReflexStatus::Miss.is_hit());
}

#[test]
fn test_reflex_status_display() {
    assert_eq!(format!("{}", ReflexStatus::HitL1Exact), "HIT_L1_EXACT");
    assert_eq!(format!("{}", ReflexStatus::Miss), "MISS");
}

#[test]
fn test_reflex_status_clone_and_eq() {
    let status = ReflexStatus::HitL1Exact;
    let cloned = status;
    assert_eq!(status, cloned);
}

#[test]
fn test_l1_cache_new() {
    let cache = L1Cache::new();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_l1_cache_with_capacity() {
    let cache = L1Cache::with_capacity(1000);
    assert!(cache.is_empty());
}

#[test]
fn test_l1_cache_insert_and_lookup() {
    let cache = L1Cache::new();
    let content = b"test payload data";
    let (_file, handle) = create_test_handle(content);

    let prompt = "What is the capital of France?";
    let hash = cache.insert(prompt, handle);

    cache.run_pending_tasks();
    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());

    let result = cache.lookup(prompt).expect("Should find entry");
    assert_eq!(result.hash(), &hash);
    assert_eq!(result.status(), ReflexStatus::HitL1Exact);
    assert_eq!(result.as_slice(), content);
}

#[test]
fn test_l1_cache_lookup_miss() {
    let cache = L1Cache::new();
    let result = cache.lookup("nonexistent prompt");
    assert!(result.is_none());
}

#[test]
fn test_l1_cache_lookup_by_hash() {
    let cache = L1Cache::new();
    let content = b"test data";
    let (_file, handle) = create_test_handle(content);

    let prompt = "test prompt";
    let hash = cache.insert(prompt, handle);

    let result = cache.lookup_by_hash(&hash).expect("Should find by hash");
    assert_eq!(result.hash(), &hash);
}

#[test]
fn test_l1_cache_insert_replaces_existing() {
    let cache = L1Cache::new();
    let content1 = b"first payload";
    let content2 = b"second payload";

    let (_file1, handle1) = create_test_handle(content1);
    let (_file2, handle2) = create_test_handle(content2);

    let prompt = "duplicate prompt";
    cache.insert(prompt, handle1);
    cache.insert(prompt, handle2);

    cache.run_pending_tasks();
    assert_eq!(cache.len(), 1);

    let result = cache.lookup(prompt).expect("Should find entry");
    assert_eq!(result.as_slice(), content2);
}

#[test]
fn test_l1_cache_insert_by_hash() {
    let cache = L1Cache::new();
    let content = b"test data";
    let (_file, handle) = create_test_handle(content);

    let hash = hash_prompt("test");
    cache.insert_by_hash(hash, handle);

    let (_file2, handle2) = create_test_handle(b"new data");
    cache.insert_by_hash(hash, handle2);

    let result = cache.lookup_by_hash(&hash).expect("Found");
    assert_eq!(result.as_slice(), b"new data");
}

#[test]
fn test_l1_cache_remove() {
    let cache = L1Cache::new();
    let content = b"test data";
    let (_file, handle) = create_test_handle(content);

    let prompt = "test prompt";
    let hash = cache.insert(prompt, handle);

    cache.run_pending_tasks();
    assert_eq!(cache.len(), 1);

    cache.remove(&hash);
    assert!(cache.lookup(prompt).is_none());
}

#[test]
fn test_l1_cache_remove_nonexistent() {
    let cache = L1Cache::new();
    let hash = hash_prompt("nonexistent");
    cache.remove(&hash);
}

#[test]
fn test_l1_cache_remove_prompt() {
    let cache = L1Cache::new();
    let content = b"test data";
    let (_file, handle) = create_test_handle(content);

    let prompt = "test prompt";
    cache.insert(prompt, handle);

    cache.remove_prompt(prompt);
    assert!(cache.lookup(prompt).is_none());
}

#[test]
fn test_l1_cache_contains_hash() {
    let cache = L1Cache::new();
    let content = b"test data";
    let (_file, handle) = create_test_handle(content);

    let prompt = "test prompt";
    let hash = cache.insert(prompt, handle);

    assert!(cache.contains_hash(&hash));

    let other_hash = hash_prompt("other prompt");
    assert!(!cache.contains_hash(&other_hash));
}

#[test]
fn test_l1_cache_contains_prompt() {
    let cache = L1Cache::new();
    let content = b"test data";
    let (_file, handle) = create_test_handle(content);

    let prompt = "test prompt";
    cache.insert(prompt, handle);

    assert!(cache.contains_prompt(prompt));
    assert!(!cache.contains_prompt("other prompt"));
}

#[test]
fn test_l1_cache_clear() {
    let cache = L1Cache::new();

    for i in 0..5 {
        let content = format!("content {}", i);
        let (_file, handle) = create_test_handle(content.as_bytes());
        cache.insert(&format!("prompt {}", i), handle);
    }

    cache.clear();
    assert!(cache.is_empty());
}

#[test]
fn test_l1_cache_hashes_iterator() {
    let cache = L1Cache::new();
    let mut expected_hashes = Vec::new();

    for i in 0..3 {
        let content = format!("content {}", i);
        let (_file, handle) = create_test_handle(content.as_bytes());
        let hash = cache.insert(&format!("prompt {}", i), handle);
        expected_hashes.push(hash);
    }

    let collected: Vec<_> = cache.hashes().collect();
    assert_eq!(collected.len(), 3);

    for hash in &expected_hashes {
        assert!(collected.contains(hash));
    }
}

#[test]
fn test_l1_cache_handle_concurrent_reads() {
    let cache = L1CacheHandle::new();
    let content = b"concurrent read data";
    let (_file, handle) = create_test_handle(content);

    cache.insert("test prompt", handle);

    let mut threads = vec![];

    for _ in 0..10 {
        let cache_clone = cache.clone();
        threads.push(std::thread::spawn(move || {
            for _ in 0..100 {
                let result = cache_clone.lookup("test prompt");
                assert!(result.is_some());
            }
        }));
    }

    for t in threads {
        t.join().expect("Thread panicked");
    }
}

#[test]
fn test_l1_cache_handle_concurrent_writes() {
    let cache = L1CacheHandle::new();

    let mut threads = vec![];

    for i in 0..10 {
        let cache_clone = cache.clone();
        threads.push(std::thread::spawn(move || {
            for j in 0..10 {
                let content = format!("data {} {}", i, j);
                let mut file = NamedTempFile::new().expect("Failed to create temp file");
                file.write_all(content.as_bytes()).expect("Failed to write");
                file.flush().expect("Failed to flush");

                let handle = MmapFileHandle::open(file.path()).expect("Failed to create handle");

                cache_clone.insert(&format!("prompt {} {}", i, j), handle);

                drop(file);
            }
        }));
    }

    for t in threads {
        t.join().expect("Thread panicked");
    }

    cache.run_pending_tasks();
    assert_eq!(cache.len(), 100);
}

#[test]
fn test_l1_lookup_result_handle() {
    let cache = L1Cache::new();
    let content = b"test data for handle";
    let (_file, handle) = create_test_handle(content);

    let prompt = "test prompt for handle";
    cache.insert(prompt, handle);

    let result = cache.lookup(prompt).expect("Should find entry");

    // Test handle() method - returns borrowed MmapFileHandle
    let borrowed_handle = result.handle();
    assert_eq!(borrowed_handle.as_slice(), content);
    assert!(!borrowed_handle.is_empty());
}

#[test]
fn test_l1_lookup_result_into_handle() {
    let cache = L1Cache::new();
    let content = b"test data for into_handle";
    let (_file, handle) = create_test_handle(content);

    let prompt = "test prompt for into_handle";
    cache.insert(prompt, handle);

    let result = cache.lookup(prompt).expect("Should find entry");

    // Test into_handle() method - consumes self and returns owned MmapFileHandle
    let owned_handle = result.into_handle();
    assert_eq!(owned_handle.as_slice(), content);
    assert!(!owned_handle.is_empty());
}

#[test]
fn test_l1_cache_default() {
    let cache = L1Cache::default();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_l1_cache_debug() {
    let cache = L1Cache::new();
    let content = b"debug test data";
    let (_file, handle) = create_test_handle(content);

    cache.insert("debug prompt", handle);
    cache.run_pending_tasks();

    let debug_str = format!("{:?}", cache);
    assert!(debug_str.contains("L1Cache"));
    assert!(debug_str.contains("entries"));
}

#[test]
fn test_l1_cache_remove_returns_handle() {
    let cache = L1Cache::new();
    let content = b"removable data";
    let (_file, handle) = create_test_handle(content);

    let prompt = "removable prompt";
    let hash = cache.insert(prompt, handle);

    // Remove should return the handle
    let removed = cache.remove(&hash);
    assert!(removed.is_some());
    assert_eq!(removed.unwrap().as_slice(), content);

    // Subsequent remove should return None
    let removed_again = cache.remove(&hash);
    assert!(removed_again.is_none());
}

#[test]
fn test_l1_cache_remove_prompt_returns_handle() {
    let cache = L1Cache::new();
    let content = b"removable prompt data";
    let (_file, handle) = create_test_handle(content);

    let prompt = "prompt to remove";
    cache.insert(prompt, handle);

    // remove_prompt should return the handle
    let removed = cache.remove_prompt(prompt);
    assert!(removed.is_some());
    assert_eq!(removed.unwrap().as_slice(), content);

    // Subsequent remove should return None
    let removed_again = cache.remove_prompt(prompt);
    assert!(removed_again.is_none());
}

#[test]
fn test_l1_cache_handle_with_capacity() {
    let cache = L1CacheHandle::with_capacity(500);
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_l1_cache_handle_lookup_by_hash() {
    let cache = L1CacheHandle::new();
    let content = b"hash lookup data";
    let (_file, handle) = create_test_handle(content);

    let prompt = "hash lookup prompt";
    let hash = cache.insert(prompt, handle);

    // lookup_by_hash should find the entry
    let result = cache.lookup_by_hash(&hash).expect("Should find by hash");
    assert_eq!(result.hash(), &hash);
    assert_eq!(result.as_slice(), content);

    // Non-existent hash should return None
    let other_hash = hash_prompt("other prompt");
    assert!(cache.lookup_by_hash(&other_hash).is_none());
}

#[test]
fn test_l1_cache_handle_insert_by_hash() {
    let cache = L1CacheHandle::new();
    let content = b"insert by hash data";
    let (_file, handle) = create_test_handle(content);

    let hash = hash_prompt("custom hash");
    cache.insert_by_hash(hash, handle);

    let result = cache.lookup_by_hash(&hash).expect("Should find entry");
    assert_eq!(result.as_slice(), content);
}

#[test]
fn test_l1_cache_handle_remove() {
    let cache = L1CacheHandle::new();
    let content = b"handle remove data";
    let (_file, handle) = create_test_handle(content);

    let prompt = "handle remove prompt";
    let hash = cache.insert(prompt, handle);

    // Remove should return the handle
    let removed = cache.remove(&hash);
    assert!(removed.is_some());
    assert_eq!(removed.unwrap().as_slice(), content);

    // Entry should be gone
    assert!(cache.lookup(prompt).is_none());
}

#[test]
fn test_l1_cache_handle_remove_prompt() {
    let cache = L1CacheHandle::new();
    let content = b"handle remove prompt data";
    let (_file, handle) = create_test_handle(content);

    let prompt = "handle prompt to remove";
    cache.insert(prompt, handle);

    // remove_prompt should return the handle
    let removed = cache.remove_prompt(prompt);
    assert!(removed.is_some());
    assert_eq!(removed.unwrap().as_slice(), content);

    // Entry should be gone
    assert!(cache.lookup(prompt).is_none());
}

#[test]
fn test_l1_cache_handle_is_empty() {
    let cache = L1CacheHandle::new();
    assert!(cache.is_empty());

    let content = b"empty test data";
    let (_file, handle) = create_test_handle(content);
    cache.insert("empty test prompt", handle);

    cache.run_pending_tasks();
    assert!(!cache.is_empty());
}

#[test]
fn test_l1_cache_handle_clear() {
    let cache = L1CacheHandle::new();

    for i in 0..5 {
        let content = format!("handle content {}", i);
        let (_file, handle) = create_test_handle(content.as_bytes());
        cache.insert(&format!("handle prompt {}", i), handle);
    }

    cache.run_pending_tasks();
    assert!(!cache.is_empty());

    cache.clear();
    // After clear(), we need to run pending tasks and then check
    // that lookups fail (clear invalidates all entries)
    cache.run_pending_tasks();

    // Verify all entries are invalidated by checking lookups fail
    for i in 0..5 {
        assert!(cache.lookup(&format!("handle prompt {}", i)).is_none());
    }
}

#[test]
fn test_l1_cache_handle_contains_hash() {
    let cache = L1CacheHandle::new();
    let content = b"contains hash data";
    let (_file, handle) = create_test_handle(content);

    let prompt = "contains hash prompt";
    let hash = cache.insert(prompt, handle);

    assert!(cache.contains_hash(&hash));

    let other_hash = hash_prompt("nonexistent");
    assert!(!cache.contains_hash(&other_hash));
}

#[test]
fn test_l1_cache_handle_contains_prompt() {
    let cache = L1CacheHandle::new();
    let content = b"contains prompt data";
    let (_file, handle) = create_test_handle(content);

    let prompt = "contains prompt test";
    cache.insert(prompt, handle);

    assert!(cache.contains_prompt(prompt));
    assert!(!cache.contains_prompt("nonexistent prompt"));
}

#[test]
fn test_l1_cache_handle_default() {
    let cache = L1CacheHandle::default();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_l1_cache_handle_debug() {
    let cache = L1CacheHandle::new();
    let debug_str = format!("{:?}", cache);
    assert!(debug_str.contains("L1CacheHandle"));
    assert!(debug_str.contains("strong_count"));
}

#[test]
fn test_l1_cache_handle_strong_count() {
    let cache1 = L1CacheHandle::new();
    assert_eq!(cache1.strong_count(), 1);

    let cache2 = cache1.clone();
    assert_eq!(cache1.strong_count(), 2);
    assert_eq!(cache2.strong_count(), 2);

    drop(cache2);
    assert_eq!(cache1.strong_count(), 1);
}

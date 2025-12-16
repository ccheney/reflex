//! Integration tests for hashing.

mod common;

use blake3::Hasher;
use common::fixtures::{CacheEntryBuilder, EMBEDDING_SIZE_BYTES};
use reflex::storage::CacheEntry;
use rkyv::rancor::Error;
use rkyv::to_bytes;
use std::collections::HashSet;

#[test]
fn test_blake3_determinism() {
    let input = b"test-tenant-id-12345";

    let hash1 = blake3::hash(input);
    let hash2 = blake3::hash(input);
    let hash3 = blake3::hash(input);

    assert_eq!(hash1, hash2);
    assert_eq!(hash2, hash3);
}

#[test]
fn test_blake3_uniqueness() {
    let inputs = [
        b"tenant-001".as_slice(),
        b"tenant-002".as_slice(),
        b"TENANT-001".as_slice(),
        b"tenant-001 ".as_slice(),
    ];

    let hashes: Vec<_> = inputs.iter().map(|i| blake3::hash(i)).collect();

    let unique_hashes: HashSet<_> = hashes.iter().collect();
    assert_eq!(unique_hashes.len(), inputs.len());
}

#[test]
fn test_blake3_empty_input() {
    let hash = blake3::hash(b"");

    assert!(!hash.as_bytes().iter().all(|&b| b == 0));
}

fn hash_tenant_id(tenant: &str) -> u64 {
    let hash = blake3::hash(tenant.as_bytes());
    let bytes: [u8; 8] = hash.as_bytes()[0..8].try_into().unwrap();
    u64::from_le_bytes(bytes)
}

#[test]
fn test_tenant_id_generation_consistency() {
    let tenant = "acme-corp-production";

    let id1 = hash_tenant_id(tenant);
    let id2 = hash_tenant_id(tenant);
    let id3 = hash_tenant_id(tenant);

    assert_eq!(id1, id2);
    assert_eq!(id2, id3);
}

#[test]
fn test_tenant_id_uniqueness() {
    let tenants = [
        "tenant-alpha",
        "tenant-beta",
        "tenant-gamma",
        "Tenant-Alpha",
        "tenant-alpha-v2",
    ];

    let ids: Vec<_> = tenants.iter().map(|t| hash_tenant_id(t)).collect();
    let unique_ids: HashSet<_> = ids.iter().collect();

    assert_eq!(unique_ids.len(), tenants.len());
}

#[test]
fn test_tenant_id_various_formats() {
    let formats = [
        "simple",
        "with-dashes-123",
        "with_underscores",
        "CamelCase",
        "UPPERCASE",
        "lowercase",
        "with.dots.and-dashes",
        "uuid-550e8400-e29b-41d4-a716-446655440000",
        "very-long-tenant-name-that-exceeds-typical-length-limits-for-testing",
    ];

    for format in formats {
        let id = hash_tenant_id(format);
        assert!(id > 0 || format.is_empty(), "Failed for: {}", format);
    }
}

fn hash_context(role: &str, plan: &str, extra: Option<&str>) -> u64 {
    let mut hasher = Hasher::new();
    hasher.update(role.as_bytes());
    hasher.update(b"|");
    hasher.update(plan.as_bytes());
    if let Some(e) = extra {
        hasher.update(b"|");
        hasher.update(e.as_bytes());
    }

    let hash = hasher.finalize();
    let bytes: [u8; 8] = hash.as_bytes()[0..8].try_into().unwrap();
    u64::from_le_bytes(bytes)
}

#[test]
fn test_context_hash_determinism() {
    let hash1 = hash_context("admin", "enterprise", Some("feature-flags"));
    let hash2 = hash_context("admin", "enterprise", Some("feature-flags"));

    assert_eq!(hash1, hash2);
}

#[test]
fn test_context_hash_role_sensitivity() {
    let admin_hash = hash_context("admin", "basic", None);
    let user_hash = hash_context("user", "basic", None);
    let guest_hash = hash_context("guest", "basic", None);

    assert_ne!(admin_hash, user_hash);
    assert_ne!(user_hash, guest_hash);
    assert_ne!(admin_hash, guest_hash);
}

#[test]
fn test_context_hash_plan_sensitivity() {
    let free_hash = hash_context("user", "free", None);
    let basic_hash = hash_context("user", "basic", None);
    let premium_hash = hash_context("user", "premium", None);

    assert_ne!(free_hash, basic_hash);
    assert_ne!(basic_hash, premium_hash);
}

fn hash_cache_entry_content(entry: &CacheEntry) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(&entry.tenant_id.to_le_bytes());
    hasher.update(&entry.context_hash.to_le_bytes());
    hasher.update(&entry.embedding);
    hasher.update(&entry.payload_blob);
    *hasher.finalize().as_bytes()
}

#[test]
fn test_content_hash_identical_entries() {
    let entry1 = CacheEntryBuilder::new()
        .tenant_id(100)
        .context_hash(200)
        .with_realistic_embedding()
        .with_sample_payload()
        .build();

    let entry2 = CacheEntryBuilder::new()
        .tenant_id(100)
        .context_hash(200)
        .with_realistic_embedding()
        .with_sample_payload()
        .build();

    let hash1 = hash_cache_entry_content(&entry1);
    let hash2 = hash_cache_entry_content(&entry2);

    assert_eq!(hash1, hash2);
}

#[test]
fn test_content_hash_different_entries() {
    let entry1 = CacheEntryBuilder::new()
        .tenant_id(100)
        .context_hash(200)
        .with_seeded_embedding(1)
        .build();

    let entry2 = CacheEntryBuilder::new()
        .tenant_id(100)
        .context_hash(200)
        .with_seeded_embedding(2)
        .build();

    let hash1 = hash_cache_entry_content(&entry1);
    let hash2 = hash_cache_entry_content(&entry2);

    assert_ne!(hash1, hash2);
}

#[test]
fn test_content_hash_ignores_timestamp() {
    let entry1 = CacheEntryBuilder::new()
        .tenant_id(100)
        .timestamp(1000)
        .with_realistic_embedding()
        .build();

    let entry2 = CacheEntryBuilder::new()
        .tenant_id(100)
        .timestamp(2000)
        .with_realistic_embedding()
        .build();

    let hash1 = hash_cache_entry_content(&entry1);
    let hash2 = hash_cache_entry_content(&entry2);

    assert_eq!(hash1, hash2, "Timestamp should not affect content hash");
}

fn hash_serialized_entry(entry: &CacheEntry) -> [u8; 32] {
    let serialized = to_bytes::<Error>(entry).expect("Serialization should succeed");
    *blake3::hash(&serialized).as_bytes()
}

#[test]
fn test_serialized_entry_hashing() {
    let entry = CacheEntryBuilder::new()
        .tenant_id(42)
        .with_realistic_embedding()
        .build();

    let hash1 = hash_serialized_entry(&entry);
    let hash2 = hash_serialized_entry(&entry);

    assert_eq!(hash1, hash2);
}

#[test]
fn test_serialized_hash_sensitivity() {
    let base_entry = CacheEntryBuilder::new()
        .tenant_id(42)
        .context_hash(84)
        .build();

    let modified_entry = CacheEntryBuilder::new()
        .tenant_id(42)
        .context_hash(85)
        .build();

    let hash1 = hash_serialized_entry(&base_entry);
    let hash2 = hash_serialized_entry(&modified_entry);

    assert_ne!(hash1, hash2);
}

#[tokio::test]
async fn test_concurrent_hashing_consistency() {
    let tenant = "concurrent-test-tenant";
    let iterations = 100;

    let handles: Vec<_> = (0..iterations)
        .map(|_| {
            let tenant = tenant.to_string();
            tokio::spawn(async move { hash_tenant_id(&tenant) })
        })
        .collect();

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.expect("Task should complete"));
    }

    let first = results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(*result, first, "Hash mismatch at iteration {}", i);
    }
}

#[tokio::test]
async fn test_concurrent_different_inputs() {
    let handle_count = 50;

    let handles: Vec<_> = (0..handle_count)
        .map(|i| {
            tokio::spawn(async move {
                let tenant = format!("tenant-{}", i);
                hash_tenant_id(&tenant)
            })
        })
        .collect();

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.expect("Task should complete"));
    }

    let unique: HashSet<_> = results.iter().collect();
    assert_eq!(unique.len(), handle_count);
}

fn hash_embedding(embedding: &[u8]) -> [u8; 32] {
    *blake3::hash(embedding).as_bytes()
}

#[test]
fn test_embedding_hash_for_dedup() {
    let emb1 = common::fixtures::generate_deterministic_embedding(42);
    let emb2 = common::fixtures::generate_deterministic_embedding(42);
    let emb3 = common::fixtures::generate_deterministic_embedding(43);

    let hash1 = hash_embedding(&emb1);
    let hash2 = hash_embedding(&emb2);
    let hash3 = hash_embedding(&emb3);

    assert_eq!(hash1, hash2, "Same embeddings should have same hash");
    assert_ne!(
        hash1, hash3,
        "Different embeddings should have different hash"
    );
}

#[test]
fn test_embedding_size_validation() {
    let valid_embedding = common::fixtures::generate_deterministic_embedding(0);
    assert_eq!(valid_embedding.len(), EMBEDDING_SIZE_BYTES);

    let small_embedding = vec![0u8; 100];
    let large_embedding = vec![0u8; 10_000];

    let small_hash = hash_embedding(&small_embedding);
    let large_hash = hash_embedding(&large_embedding);

    assert_eq!(small_hash.len(), 32);
    assert_eq!(large_hash.len(), 32);
    assert_ne!(small_hash, large_hash);
}

#[test]
fn test_keyed_hashing_per_tenant() {
    let content = b"same content for all tenants";

    let key1 = blake3::hash(b"tenant-1").as_bytes()[0..32]
        .try_into()
        .unwrap();
    let key2 = blake3::hash(b"tenant-2").as_bytes()[0..32]
        .try_into()
        .unwrap();

    let hash1 = blake3::keyed_hash(&key1, content);
    let hash2 = blake3::keyed_hash(&key2, content);

    assert_ne!(
        hash1, hash2,
        "Different keys should produce different hashes"
    );
}

#[test]
fn test_keyed_hashing_consistency() {
    let content = b"test content";
    let key: [u8; 32] = blake3::hash(b"tenant-key").as_bytes()[0..32]
        .try_into()
        .unwrap();

    let hash1 = blake3::keyed_hash(&key, content);
    let hash2 = blake3::keyed_hash(&key, content);

    assert_eq!(hash1, hash2);
}

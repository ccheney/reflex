use blake3::Hasher;

#[inline]
pub fn hash_prompt(prompt: &str) -> [u8; 32] {
    *blake3::hash(prompt.as_bytes()).as_bytes()
}

/// Computes a 64-bit hash of the input data using BLAKE3, truncated from 256 bits.
///
/// # Truncation Rationale
///
/// This function takes the first 8 bytes (64 bits) of a BLAKE3 hash. This truncation
/// is acceptable for the following use cases:
///
/// - **Cache keys**: Fast lookups in hash maps and tiered caches
/// - **Identifiers**: Tenant IDs, context hashes, and content fingerprints
/// - **Deduplication**: Detecting likely-duplicate entries before expensive operations
///
/// # Collision Probability
///
/// With 64 bits of entropy, the birthday paradox gives us the following collision probabilities:
///
/// | Number of Items | Collision Probability |
/// |-----------------|----------------------|
/// | 1 million       | ~0.00003% (negligible) |
/// | 10 million      | ~0.003% (very low) |
/// | 100 million     | ~0.3% (low) |
/// | 1 billion       | ~3% (noticeable) |
/// | ~4.3 billion    | ~50% (birthday bound) |
///
/// For practical cache sizes (millions of entries), the collision probability is negligible.
/// The formula is approximately: `P(collision) ≈ n² / (2 × 2^64)` for `n` items.
///
/// # Collision Tolerance
///
/// The higher-level logic (tiered cache, content-addressed storage) is designed to tolerate
/// rare collisions gracefully:
///
/// - **Cache lookups**: A collision results in a cache miss, not data corruption. The full
///   content is verified downstream, so a false positive simply triggers a cache refresh.
/// - **No security dependency**: This hash is not used for cryptographic verification or
///   authentication—only for fast indexing and probabilistic deduplication.
///
/// # When to Use Full 256-bit Hashes
///
/// If stricter uniqueness guarantees are ever required (e.g., content-addressed storage
/// where collisions would cause data loss), use [`hash_prompt`] or [`hash_cache_content`]
/// which return the full 32-byte BLAKE3 output. The full hash provides ~128 bits of
/// collision resistance, making collisions computationally infeasible.
#[inline]
pub fn hash_to_u64(data: &[u8]) -> u64 {
    let hash = blake3::hash(data);
    let bytes: [u8; 8] = hash.as_bytes()[0..8]
        .try_into()
        .expect("BLAKE3 always produces at least 8 bytes");
    u64::from_le_bytes(bytes)
}

#[inline]
pub fn hash_context(role: &str, plan: &str) -> u64 {
    let mut hasher = Hasher::new();
    hasher.update(role.as_bytes());
    hasher.update(b"|");
    hasher.update(plan.as_bytes());

    let hash = hasher.finalize();
    let bytes: [u8; 8] = hash.as_bytes()[0..8]
        .try_into()
        .expect("BLAKE3 always produces at least 8 bytes");
    u64::from_le_bytes(bytes)
}

#[inline]
pub fn hash_tenant_id(tenant: &str) -> u64 {
    hash_to_u64(tenant.as_bytes())
}

#[inline]
pub fn hash_cache_content(
    tenant_id: u64,
    context_hash: u64,
    embedding: &[u8],
    payload: &[u8],
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(&tenant_id.to_le_bytes());
    hasher.update(&context_hash.to_le_bytes());
    hasher.update(embedding);
    hasher.update(payload);
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_hash_prompt_determinism() {
        let prompt = "What is the capital of France?";

        let hash1 = hash_prompt(prompt);
        let hash2 = hash_prompt(prompt);
        let hash3 = hash_prompt(prompt);

        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
    }

    #[test]
    fn test_hash_prompt_uniqueness() {
        let prompts = [
            "What is the capital of France?",
            "What is the capital of Germany?",
            "what is the capital of france?",
            "What is the capital of France? ",
        ];

        let hashes: Vec<_> = prompts.iter().map(|p| hash_prompt(p)).collect();
        let unique_hashes: HashSet<_> = hashes.iter().collect();

        assert_eq!(unique_hashes.len(), prompts.len());
    }

    #[test]
    fn test_hash_prompt_output_size() {
        let hash = hash_prompt("test");
        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_hash_prompt_empty_string() {
        let hash = hash_prompt("");
        assert!(!hash.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_hash_prompt_unicode() {
        let prompt = "Quelle est la capitale de la France? ";
        let hash = hash_prompt(prompt);
        assert_eq!(hash.len(), 32);

        let hash2 = hash_prompt("What is the capital of France?");
        assert_ne!(hash, hash2);
    }

    #[test]
    fn test_hash_to_u64_determinism() {
        let data = b"test-tenant-id-12345";

        let hash1 = hash_to_u64(data);
        let hash2 = hash_to_u64(data);
        let hash3 = hash_to_u64(data);

        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
    }

    #[test]
    fn test_hash_to_u64_uniqueness() {
        let inputs = [
            b"tenant-001".as_slice(),
            b"tenant-002".as_slice(),
            b"TENANT-001".as_slice(),
            b"tenant-001 ".as_slice(),
        ];

        let hashes: Vec<_> = inputs.iter().map(|i| hash_to_u64(i)).collect();
        let unique_hashes: HashSet<_> = hashes.iter().collect();

        assert_eq!(unique_hashes.len(), inputs.len());
    }

    #[test]
    fn test_hash_to_u64_empty_input() {
        let hash = hash_to_u64(b"");
        let hash2 = hash_to_u64(b"");
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_hash_context_determinism() {
        let hash1 = hash_context("admin", "enterprise");
        let hash2 = hash_context("admin", "enterprise");
        let hash3 = hash_context("admin", "enterprise");

        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
    }

    #[test]
    fn test_hash_context_role_sensitivity() {
        let admin_hash = hash_context("admin", "basic");
        let user_hash = hash_context("user", "basic");
        let guest_hash = hash_context("guest", "basic");

        assert_ne!(admin_hash, user_hash);
        assert_ne!(user_hash, guest_hash);
        assert_ne!(admin_hash, guest_hash);
    }

    #[test]
    fn test_hash_context_plan_sensitivity() {
        let free_hash = hash_context("user", "free");
        let basic_hash = hash_context("user", "basic");
        let premium_hash = hash_context("user", "premium");

        assert_ne!(free_hash, basic_hash);
        assert_ne!(basic_hash, premium_hash);
    }

    #[test]
    fn test_hash_context_separator_prevents_ambiguity() {
        let hash1 = hash_context("ab", "cd");
        let hash2 = hash_context("abc", "d");
        let hash3 = hash_context("a", "bcd");

        assert_ne!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_ne!(hash2, hash3);
    }

    #[test]
    fn test_hash_tenant_id_consistency() {
        let tenant = "acme-corp-production";

        let id1 = hash_tenant_id(tenant);
        let id2 = hash_tenant_id(tenant);

        assert_eq!(id1, id2);
    }

    #[test]
    fn test_hash_tenant_id_equals_hash_to_u64() {
        let tenant = "test-tenant";
        assert_eq!(hash_tenant_id(tenant), hash_to_u64(tenant.as_bytes()));
    }

    #[test]
    fn test_hash_cache_content_determinism() {
        let hash1 = hash_cache_content(100, 200, &[1, 2, 3], &[4, 5, 6]);
        let hash2 = hash_cache_content(100, 200, &[1, 2, 3], &[4, 5, 6]);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_cache_content_sensitivity() {
        let base = hash_cache_content(100, 200, &[1, 2, 3], &[4, 5, 6]);

        let changed_tenant = hash_cache_content(101, 200, &[1, 2, 3], &[4, 5, 6]);
        assert_ne!(base, changed_tenant);

        let changed_context = hash_cache_content(100, 201, &[1, 2, 3], &[4, 5, 6]);
        assert_ne!(base, changed_context);

        let changed_embedding = hash_cache_content(100, 200, &[1, 2, 4], &[4, 5, 6]);
        assert_ne!(base, changed_embedding);

        let changed_payload = hash_cache_content(100, 200, &[1, 2, 3], &[4, 5, 7]);
        assert_ne!(base, changed_payload);
    }

    #[test]
    fn test_hash_cache_content_output_size() {
        let hash = hash_cache_content(0, 0, &[], &[]);
        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_hash_performance_sanity() {
        let prompt = "A moderately long prompt that represents typical user input for testing.";

        let prompt = std::hint::black_box(prompt);
        for _ in 0..10_000 {
            let _ = std::hint::black_box(hash_prompt(std::hint::black_box(prompt)));
        }
    }
}

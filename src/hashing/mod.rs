//! Hash utilities for cache keys and identifiers.
//!
//! Uses BLAKE3. Prefer 32-byte hashes for exact keys and 64-bit hashes for compact ids.

use blake3::Hasher;

/// Hashes a prompt to a 32-byte BLAKE3 digest.
#[inline]
pub fn hash_prompt(prompt: &str) -> [u8; 32] {
    *blake3::hash(prompt.as_bytes()).as_bytes()
}

/// Computes a 64-bit hash of the input data using BLAKE3, truncated from 256 bits.
///
/// Use this for compact ids (tenant/context). Not suitable for security.
#[inline]
pub fn hash_to_u64(data: &[u8]) -> u64 {
    let hash = blake3::hash(data);
    let bytes: [u8; 8] = hash.as_bytes()[0..8]
        .try_into()
        .expect("BLAKE3 always produces at least 8 bytes");
    u64::from_le_bytes(bytes)
}

/// Hashes a `(role, plan)` pair into a compact 64-bit context hash.
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

/// Hashes a tenant identifier string to a compact 64-bit id.
#[inline]
pub fn hash_tenant_id(tenant: &str) -> u64 {
    hash_to_u64(tenant.as_bytes())
}

/// Hashes cache content to a 32-byte digest (tenant + context + embedding + payload).
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

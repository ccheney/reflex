//! Test fixtures for integration tests.

use reflex::storage::CacheEntry;

pub const DEFAULT_TENANT_ID: u64 = 1000;

pub const DEFAULT_CONTEXT_HASH: u64 = 2000;

pub const FIXED_TIMESTAMP: i64 = 1702512000;

pub const EMBEDDING_SIZE_BYTES: usize = reflex::constants::EMBEDDING_F16_BYTES;

#[derive(Default)]
pub struct CacheEntryBuilder {
    tenant_id: Option<u64>,
    context_hash: Option<u64>,
    timestamp: Option<i64>,
    embedding: Option<Vec<u8>>,
    payload_blob: Option<Vec<u8>>,
}

impl CacheEntryBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tenant_id(mut self, id: u64) -> Self {
        self.tenant_id = Some(id);
        self
    }

    pub fn context_hash(mut self, hash: u64) -> Self {
        self.context_hash = Some(hash);
        self
    }

    pub fn timestamp(mut self, ts: i64) -> Self {
        self.timestamp = Some(ts);
        self
    }

    pub fn embedding(mut self, emb: Vec<u8>) -> Self {
        self.embedding = Some(emb);
        self
    }

    pub fn payload_blob(mut self, payload: Vec<u8>) -> Self {
        self.payload_blob = Some(payload);
        self
    }

    pub fn with_realistic_embedding(mut self) -> Self {
        self.embedding = Some(generate_deterministic_embedding(0));
        self
    }

    pub fn with_seeded_embedding(mut self, seed: u64) -> Self {
        self.embedding = Some(generate_deterministic_embedding(seed));
        self
    }

    pub fn with_sample_payload(mut self) -> Self {
        self.payload_blob = Some(create_sample_tauq_payload());
        self
    }

    pub fn with_json_payload(mut self, json: &str) -> Self {
        self.payload_blob = Some(json.as_bytes().to_vec());
        self
    }

    pub fn build(self) -> CacheEntry {
        CacheEntry {
            tenant_id: self.tenant_id.unwrap_or(DEFAULT_TENANT_ID),
            context_hash: self.context_hash.unwrap_or(DEFAULT_CONTEXT_HASH),
            timestamp: self.timestamp.unwrap_or(FIXED_TIMESTAMP),
            embedding: self.embedding.unwrap_or_default(),
            payload_blob: self.payload_blob.unwrap_or_default(),
        }
    }
}

pub fn generate_deterministic_embedding(seed: u64) -> Vec<u8> {
    (0..EMBEDDING_SIZE_BYTES)
        .map(|i| {
            let mixed = (seed.wrapping_mul(31).wrapping_add(i as u64)) % 256;
            mixed as u8
        })
        .collect()
}

pub fn create_sample_tauq_payload() -> Vec<u8> {
    let payload = r#"{
        "semantic_request": "sample request",
        "response": {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a cached response."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20
            }
        }
    }"#;
    payload.as_bytes().to_vec()
}

pub fn create_large_payload(size: usize) -> Vec<u8> {
    let base = r#"{"type":"large_payload","data":""#;
    let suffix = r#""}"#;
    let data_size = size.saturating_sub(base.len() + suffix.len());

    let mut payload = String::with_capacity(size);
    payload.push_str(base);
    payload.extend(std::iter::repeat_n('x', data_size));
    payload.push_str(suffix);

    payload.into_bytes()
}

pub fn create_batch_entries(count: usize) -> Vec<CacheEntry> {
    (0..count)
        .map(|i| {
            CacheEntryBuilder::new()
                .tenant_id(DEFAULT_TENANT_ID + i as u64)
                .context_hash(DEFAULT_CONTEXT_HASH + i as u64)
                .timestamp(FIXED_TIMESTAMP + i as i64)
                .with_seeded_embedding(i as u64)
                .with_sample_payload()
                .build()
        })
        .collect()
}

pub fn create_tenant_entries(tenant_id: u64, count: usize) -> Vec<CacheEntry> {
    (0..count)
        .map(|i| {
            CacheEntryBuilder::new()
                .tenant_id(tenant_id)
                .context_hash(DEFAULT_CONTEXT_HASH + i as u64)
                .timestamp(FIXED_TIMESTAMP + i as i64)
                .with_seeded_embedding(tenant_id.wrapping_mul(1000) + i as u64)
                .with_sample_payload()
                .build()
        })
        .collect()
}

pub fn create_time_series_entries(
    start_timestamp: i64,
    interval_seconds: i64,
    count: usize,
) -> Vec<CacheEntry> {
    (0..count)
        .map(|i| {
            CacheEntryBuilder::new()
                .tenant_id(DEFAULT_TENANT_ID)
                .context_hash(DEFAULT_CONTEXT_HASH + i as u64)
                .timestamp(start_timestamp + (i as i64 * interval_seconds))
                .with_seeded_embedding(i as u64)
                .build()
        })
        .collect()
}

pub fn assert_entries_equal(left: &CacheEntry, right: &CacheEntry) {
    assert_eq!(left.tenant_id, right.tenant_id, "tenant_id mismatch");
    assert_eq!(
        left.context_hash, right.context_hash,
        "context_hash mismatch"
    );
    assert_eq!(left.timestamp, right.timestamp, "timestamp mismatch");
    assert_eq!(left.embedding, right.embedding, "embedding mismatch");
    assert_eq!(
        left.payload_blob, right.payload_blob,
        "payload_blob mismatch"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let entry = CacheEntryBuilder::new().build();

        assert_eq!(entry.tenant_id, DEFAULT_TENANT_ID);
        assert_eq!(entry.context_hash, DEFAULT_CONTEXT_HASH);
        assert_eq!(entry.timestamp, FIXED_TIMESTAMP);
        assert!(entry.embedding.is_empty());
        assert!(entry.payload_blob.is_empty());
    }

    #[test]
    fn test_builder_custom_values() {
        let entry = CacheEntryBuilder::new()
            .tenant_id(42)
            .context_hash(84)
            .timestamp(1000)
            .build();

        assert_eq!(entry.tenant_id, 42);
        assert_eq!(entry.context_hash, 84);
        assert_eq!(entry.timestamp, 1000);
    }

    #[test]
    fn test_realistic_embedding_size() {
        let entry = CacheEntryBuilder::new().with_realistic_embedding().build();

        assert_eq!(entry.embedding.len(), EMBEDDING_SIZE_BYTES);
    }

    #[test]
    fn test_seeded_embeddings_are_deterministic() {
        let emb1 = generate_deterministic_embedding(42);
        let emb2 = generate_deterministic_embedding(42);
        let emb3 = generate_deterministic_embedding(43);

        assert_eq!(emb1, emb2, "Same seed should produce same embedding");
        assert_ne!(
            emb1, emb3,
            "Different seeds should produce different embeddings"
        );
    }

    #[test]
    fn test_batch_entries_creation() {
        let entries = create_batch_entries(5);

        assert_eq!(entries.len(), 5);

        for (i, entry) in entries.iter().enumerate() {
            assert_eq!(entry.tenant_id, DEFAULT_TENANT_ID + i as u64);
            assert_eq!(entry.context_hash, DEFAULT_CONTEXT_HASH + i as u64);
        }
    }

    #[test]
    fn test_tenant_entries_have_same_tenant() {
        let tenant_id = 9999;
        let entries = create_tenant_entries(tenant_id, 3);

        for entry in &entries {
            assert_eq!(entry.tenant_id, tenant_id);
        }

        let hashes: Vec<_> = entries.iter().map(|e| e.context_hash).collect();
        assert_ne!(hashes[0], hashes[1]);
        assert_ne!(hashes[1], hashes[2]);
    }

    #[test]
    fn test_time_series_entries_have_correct_intervals() {
        let entries = create_time_series_entries(1000, 60, 3);

        assert_eq!(entries[0].timestamp, 1000);
        assert_eq!(entries[1].timestamp, 1060);
        assert_eq!(entries[2].timestamp, 1120);
    }

    #[test]
    fn test_large_payload_creation() {
        let payload = create_large_payload(10000);
        assert_eq!(payload.len(), 10000);

        let s = String::from_utf8(payload).expect("Should be valid UTF-8");
        assert!(s.starts_with(r#"{"type":"large_payload","#));
    }

    #[test]
    fn test_builder_with_json_payload() {
        let json = r#"{"custom":"data"}"#;
        let entry = CacheEntryBuilder::new().with_json_payload(json).build();

        assert_eq!(entry.payload_blob, json.as_bytes());
    }

    #[test]
    fn test_builder_explicit_embedding_and_payload_blob() {
        let embedding = generate_deterministic_embedding(42);
        let payload = b"raw payload".to_vec();

        let entry = CacheEntryBuilder::new()
            .embedding(embedding.clone())
            .payload_blob(payload.clone())
            .build();

        assert_eq!(entry.embedding, embedding);
        assert_eq!(entry.payload_blob, payload);
    }

    #[test]
    fn test_assert_entries_equal_helper() {
        let entry = CacheEntryBuilder::new()
            .with_seeded_embedding(1)
            .with_sample_payload()
            .build();

        assert_entries_equal(&entry, &entry);
    }
}

//! Storage model types.

use rkyv::{Archive, Deserialize, Serialize};

/// Cached entry persisted to disk.
///
/// Stored as `rkyv` bytes (often memory-mapped).
///
/// # Example
/// ```rust
/// use reflex::CacheEntry;
///
/// let entry = CacheEntry {
///     tenant_id: 1,
///     context_hash: 42,
///     timestamp: 0,
///     embedding: vec![],
///     payload_blob: vec![],
/// };
/// assert_eq!(entry.tenant_id, 1);
/// ```
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
pub struct CacheEntry {
    /// Tenant identifier (used for isolation).
    pub tenant_id: u64,
    /// Hash of the conversation context.
    pub context_hash: u64,
    /// Unix timestamp when cached.
    pub timestamp: i64,
    /// Embedding vector bytes (little-endian f16).
    pub embedding: Vec<u8>,
    /// Encoded response payload bytes.
    pub payload_blob: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rkyv::rancor::Error;
    use rkyv::{access, from_bytes, to_bytes};

    const TEST_PAYLOAD_BYTES: usize = 1024;
    const SMALL_ENTRY_MAX_SERIALIZED_BYTES: usize = 1024;

    fn create_test_entry() -> CacheEntry {
        CacheEntry {
            tenant_id: 12345678901234567890_u64,
            context_hash: 9876543210987654321_u64,
            timestamp: 1702500000_i64,
            embedding: vec![0x01, 0x02, 0x03, 0x04],
            payload_blob: vec![0xDE, 0xAD, 0xBE, 0xEF],
        }
    }

    fn create_full_embedding_entry() -> CacheEntry {
        CacheEntry {
            tenant_id: 1,
            context_hash: 2,
            timestamp: 1702500000,
            embedding: (0..crate::constants::EMBEDDING_F16_BYTES)
                .map(|i| (i % 256) as u8)
                .collect(),
            payload_blob: vec![0x00; TEST_PAYLOAD_BYTES],
        }
    }

    #[test]
    fn test_cache_entry_field_initialization() {
        let entry = create_test_entry();

        assert_eq!(entry.tenant_id, 12345678901234567890_u64);
        assert_eq!(entry.context_hash, 9876543210987654321_u64);
        assert_eq!(entry.timestamp, 1702500000_i64);
        assert_eq!(entry.embedding, vec![0x01, 0x02, 0x03, 0x04]);
        assert_eq!(entry.payload_blob, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_cache_entry_debug_trait() {
        let entry = create_test_entry();
        let debug_str = format!("{:?}", entry);

        assert!(debug_str.contains("CacheEntry"));
        assert!(debug_str.contains("tenant_id"));
        assert!(debug_str.contains("12345678901234567890"));
    }

    #[test]
    fn test_cache_entry_clone() {
        let entry = create_test_entry();
        let cloned = entry.clone();

        assert_eq!(entry, cloned);
    }

    #[test]
    fn test_cache_entry_equality() {
        let entry1 = create_test_entry();
        let entry2 = create_test_entry();
        let entry3 = CacheEntry {
            tenant_id: 999,
            ..create_test_entry()
        };

        assert_eq!(entry1, entry2);
        assert_ne!(entry1, entry3);
    }

    #[test]
    fn test_serialization_roundtrip_basic() {
        let original = create_test_entry();

        let bytes = to_bytes::<Error>(&original).expect("serialization should succeed");

        let deserialized: CacheEntry =
            from_bytes::<CacheEntry, Error>(&bytes).expect("deserialization should succeed");

        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_serialization_roundtrip_full_embedding() {
        let original = create_full_embedding_entry();

        let bytes = to_bytes::<Error>(&original).expect("serialization should succeed");
        let deserialized: CacheEntry =
            from_bytes::<CacheEntry, Error>(&bytes).expect("deserialization should succeed");

        assert_eq!(original, deserialized);
        assert_eq!(
            deserialized.embedding.len(),
            crate::constants::EMBEDDING_F16_BYTES
        );
    }

    #[test]
    fn test_archived_zero_copy_access() {
        let original = create_test_entry();

        let bytes = to_bytes::<Error>(&original).expect("serialization should succeed");

        let archived =
            access::<ArchivedCacheEntry, Error>(&bytes).expect("archive access should succeed");

        assert_eq!(archived.tenant_id, original.tenant_id);
        assert_eq!(archived.context_hash, original.context_hash);
        assert_eq!(archived.timestamp, original.timestamp);
        assert_eq!(archived.embedding.as_slice(), original.embedding.as_slice());
        assert_eq!(
            archived.payload_blob.as_slice(),
            original.payload_blob.as_slice()
        );
    }

    #[test]
    fn test_archived_embedding_slice_access() {
        let original = create_full_embedding_entry();

        let bytes = to_bytes::<Error>(&original).expect("serialization should succeed");
        let archived =
            access::<ArchivedCacheEntry, Error>(&bytes).expect("archive access should succeed");

        assert_eq!(
            archived.embedding.len(),
            crate::constants::EMBEDDING_F16_BYTES
        );
        assert_eq!(archived.embedding[0], 0);
        assert_eq!(archived.embedding[255], 255);
        assert_eq!(archived.embedding[256], 0);
    }

    #[test]
    fn test_empty_vectors() {
        let entry = CacheEntry {
            tenant_id: 1,
            context_hash: 2,
            timestamp: 3,
            embedding: vec![],
            payload_blob: vec![],
        };

        let bytes = to_bytes::<Error>(&entry).expect("serialization should succeed");
        let deserialized: CacheEntry =
            from_bytes::<CacheEntry, Error>(&bytes).expect("deserialization should succeed");

        assert!(deserialized.embedding.is_empty());
        assert!(deserialized.payload_blob.is_empty());
    }

    #[test]
    fn test_boundary_values_max() {
        let entry = CacheEntry {
            tenant_id: u64::MAX,
            context_hash: u64::MAX,
            timestamp: i64::MAX,
            embedding: vec![0xFF],
            payload_blob: vec![0xFF],
        };

        let bytes = to_bytes::<Error>(&entry).expect("serialization should succeed");
        let deserialized: CacheEntry =
            from_bytes::<CacheEntry, Error>(&bytes).expect("deserialization should succeed");

        assert_eq!(deserialized.tenant_id, u64::MAX);
        assert_eq!(deserialized.context_hash, u64::MAX);
        assert_eq!(deserialized.timestamp, i64::MAX);
    }

    #[test]
    fn test_boundary_values_min() {
        let entry = CacheEntry {
            tenant_id: u64::MIN,
            context_hash: u64::MIN,
            timestamp: i64::MIN,
            embedding: vec![0x00],
            payload_blob: vec![0x00],
        };

        let bytes = to_bytes::<Error>(&entry).expect("serialization should succeed");
        let deserialized: CacheEntry =
            from_bytes::<CacheEntry, Error>(&bytes).expect("deserialization should succeed");

        assert_eq!(deserialized.tenant_id, u64::MIN);
        assert_eq!(deserialized.context_hash, u64::MIN);
        assert_eq!(deserialized.timestamp, i64::MIN);
    }

    #[test]
    fn test_negative_timestamp() {
        let entry = CacheEntry {
            tenant_id: 1,
            context_hash: 2,
            timestamp: -1000000000_i64,
            embedding: vec![],
            payload_blob: vec![],
        };

        let bytes = to_bytes::<Error>(&entry).expect("serialization should succeed");
        let deserialized: CacheEntry =
            from_bytes::<CacheEntry, Error>(&bytes).expect("deserialization should succeed");

        assert_eq!(deserialized.timestamp, -1000000000_i64);
    }

    #[test]
    fn test_large_payload_blob() {
        let large_payload: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();

        let entry = CacheEntry {
            tenant_id: 1,
            context_hash: 2,
            timestamp: 3,
            embedding: vec![],
            payload_blob: large_payload.clone(),
        };

        let bytes = to_bytes::<Error>(&entry).expect("serialization should succeed");
        let deserialized: CacheEntry =
            from_bytes::<CacheEntry, Error>(&bytes).expect("deserialization should succeed");

        assert_eq!(deserialized.payload_blob.len(), 1_000_000);
        assert_eq!(deserialized.payload_blob, large_payload);
    }

    #[test]
    fn test_serialized_size_is_reasonable() {
        let entry = create_test_entry();

        let bytes = to_bytes::<Error>(&entry).expect("serialization should succeed");

        assert!(bytes.len() >= 32);
        assert!(bytes.len() < SMALL_ENTRY_MAX_SERIALIZED_BYTES);
    }

    #[test]
    fn test_full_embedding_serialized_size() {
        let entry = create_full_embedding_entry();

        let bytes = to_bytes::<Error>(&entry).expect("serialization should succeed");

        assert!(bytes.len() >= crate::constants::EMBEDDING_F16_BYTES + TEST_PAYLOAD_BYTES);
    }
}

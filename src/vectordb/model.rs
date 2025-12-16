use half::f16;
use qdrant_client::qdrant::ScoredPoint;
use qdrant_client::qdrant::point_id::PointIdOptions;

use super::VectorDbError;

#[derive(Debug, Clone)]
/// Point payload used for upserts.
pub struct VectorPoint {
    /// Point id.
    pub id: u64,
    /// Vector values.
    pub vector: Vec<f32>,
    /// Tenant identifier.
    pub tenant_id: u64,
    /// Context hash.
    pub context_hash: u64,
    /// Unix timestamp.
    pub timestamp: i64,
    /// Optional storage key for loading the full entry.
    pub storage_key: Option<String>,
}

impl VectorPoint {
    /// Creates a point with `timestamp=0` and no storage key.
    pub fn new(id: u64, vector: Vec<f32>, tenant_id: u64, context_hash: u64) -> Self {
        Self {
            id,
            vector,
            tenant_id,
            context_hash,
            timestamp: 0,
            storage_key: None,
        }
    }

    /// Builds a point from f16 embedding bytes.
    pub fn from_embedding_bytes(
        id: u64,
        embedding_bytes: &[u8],
        tenant_id: u64,
        context_hash: u64,
    ) -> Result<Self, VectorDbError> {
        let vector = embedding_bytes_to_f32(embedding_bytes)?;
        Ok(Self::new(id, vector, tenant_id, context_hash))
    }

    /// Sets the timestamp.
    pub fn with_timestamp(mut self, timestamp: i64) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Sets the storage key.
    pub fn with_storage_key(mut self, key: String) -> Self {
        self.storage_key = Some(key);
        self
    }
}

#[derive(Debug, Clone)]
/// Result returned by a vector search.
pub struct SearchResult {
    /// Point id.
    pub id: u64,
    /// Similarity score (higher is better).
    pub score: f32,
    /// Tenant identifier.
    pub tenant_id: u64,
    /// Context hash.
    pub context_hash: u64,
    /// Unix timestamp.
    pub timestamp: i64,
    /// Optional storage key for loading the full entry.
    pub storage_key: Option<String>,
}

impl SearchResult {
    /// Converts a Qdrant `ScoredPoint` into a typed result (returns `None` if unsupported id).
    pub fn from_scored_point(point: ScoredPoint) -> Option<Self> {
        let id = match point.id.and_then(|pid| pid.point_id_options) {
            Some(PointIdOptions::Num(n)) => n,
            _ => return None,
        };

        let payload = point.payload;

        let tenant_id = payload
            .get("tenant_id")
            .and_then(|v| v.as_integer())
            .map(|i| i as u64)
            .unwrap_or(0);

        let context_hash = payload
            .get("context_hash")
            .and_then(|v| v.as_integer())
            .map(|i| i as u64)
            .unwrap_or(0);

        let timestamp = payload
            .get("timestamp")
            .and_then(|v| v.as_integer())
            .unwrap_or(0);

        let storage_key = payload
            .get("storage_key")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Some(SearchResult {
            id,
            score: point.score,
            tenant_id,
            context_hash,
            timestamp,
            storage_key,
        })
    }
}

/// Convert little-endian f16 bytes to f32 values.
pub fn embedding_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, VectorDbError> {
    if bytes.len() != crate::constants::EMBEDDING_F16_BYTES {
        return Err(VectorDbError::InvalidEmbeddingBytesLength {
            expected: crate::constants::EMBEDDING_F16_BYTES,
            actual: bytes.len(),
        });
    }

    if !bytes.len().is_multiple_of(2) {
        return Err(VectorDbError::InvalidEmbeddingBytesLength {
            expected: crate::constants::EMBEDDING_F16_BYTES,
            actual: bytes.len(),
        });
    }

    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f16::from_bits(bits).to_f32()
        })
        .collect())
}

/// Convert f32 values to little-endian f16 bytes.
pub fn f32_to_embedding_bytes(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect()
}

/// Generate a point id from tenant and context.
pub fn generate_point_id(tenant_id: u64, context_hash: u64) -> u64 {
    tenant_id
        .wrapping_mul(0x517cc1b727220a95)
        .wrapping_add(context_hash)
}

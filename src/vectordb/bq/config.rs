/// Default collection name for the binary-quantized index.
pub const BQ_COLLECTION_NAME: &str = "reflex_cache_bq";

/// Default vector size for BQ collections.
pub const BQ_VECTOR_SIZE: u64 = crate::constants::DEFAULT_VECTOR_SIZE_U64;

/// Bytes per vector in BQ representation (1 bit per dimension).
pub const BQ_BYTES_PER_VECTOR: usize = crate::constants::EMBEDDING_BQ_BYTES;

/// Bytes per vector in f32 representation.
pub const ORIGINAL_BYTES_PER_VECTOR: usize = crate::constants::EMBEDDING_F32_BYTES;

/// Approximate compression ratio (f32 bytes / BQ bytes).
pub const BQ_COMPRESSION_RATIO: usize = ORIGINAL_BYTES_PER_VECTOR / BQ_BYTES_PER_VECTOR;

/// Default number of candidates considered for rescoring.
pub const DEFAULT_RESCORE_CANDIDATES: u64 = 50;

#[derive(Debug, Clone)]
/// Configuration for [`BqClient`](super::client::BqClient).
pub struct BqConfig {
    /// Keep quantized vectors always in RAM.
    pub always_ram: bool,

    /// Enable quantization-time rescoring in Qdrant.
    pub rescore: bool,

    /// Oversampling limit used when rescoring is enabled.
    pub rescore_limit: u64,

    /// Store payload on disk.
    pub on_disk_payload: bool,
}

impl Default for BqConfig {
    fn default() -> Self {
        Self {
            always_ram: true,
            rescore: true,
            rescore_limit: 50,
            on_disk_payload: true,
        }
    }
}

impl BqConfig {
    /// Creates a default config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets `always_ram`.
    pub fn always_ram(mut self, value: bool) -> Self {
        self.always_ram = value;
        self
    }

    /// Sets `rescore`.
    pub fn rescore(mut self, value: bool) -> Self {
        self.rescore = value;
        self
    }

    /// Sets `rescore_limit`.
    pub fn rescore_limit(mut self, value: u64) -> Self {
        self.rescore_limit = value;
        self
    }

    /// Sets `on_disk_payload`.
    pub fn on_disk_payload(mut self, value: bool) -> Self {
        self.on_disk_payload = value;
        self
    }

    /// Validates that rescoring settings make sense for an `expected_limit`.
    pub fn validate_for_limit(&self, expected_limit: u64) -> Result<(), String> {
        if !self.rescore {
            return Ok(());
        }

        if expected_limit == 0 {
            return Err("expected_limit must be > 0".to_string());
        }

        let oversampling_ratio = self.rescore_limit as f64 / expected_limit as f64;
        if oversampling_ratio < 1.0 {
            return Err(format!(
                "rescore_limit ({}) should be >= expected_limit ({}) for effective oversampling; \
                 current ratio is {:.2}",
                self.rescore_limit, expected_limit, oversampling_ratio
            ));
        }

        Ok(())
    }

    /// Estimates RAM bytes used for `num_vectors` quantized vectors.
    pub fn estimate_ram_bytes(&self, num_vectors: u64) -> u64 {
        num_vectors * BQ_BYTES_PER_VECTOR as u64
    }

    /// Estimates byte savings vs f32 vectors for `num_vectors`.
    pub fn estimate_savings_bytes(&self, num_vectors: u64) -> u64 {
        let original = num_vectors * ORIGINAL_BYTES_PER_VECTOR as u64;
        let compressed = self.estimate_ram_bytes(num_vectors);
        original - compressed
    }
}

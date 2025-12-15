pub const BQ_COLLECTION_NAME: &str = "reflex_cache_bq";

pub const BQ_VECTOR_SIZE: u64 = crate::constants::DEFAULT_VECTOR_SIZE_U64;

pub const BQ_BYTES_PER_VECTOR: usize = crate::constants::EMBEDDING_BQ_BYTES;

pub const ORIGINAL_BYTES_PER_VECTOR: usize = crate::constants::EMBEDDING_F32_BYTES;

pub const BQ_COMPRESSION_RATIO: usize = ORIGINAL_BYTES_PER_VECTOR / BQ_BYTES_PER_VECTOR;

pub const DEFAULT_RESCORE_CANDIDATES: u64 = 50;

#[derive(Debug, Clone)]
pub struct BqConfig {
    pub always_ram: bool,

    pub rescore: bool,

    pub rescore_limit: u64,

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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn always_ram(mut self, value: bool) -> Self {
        self.always_ram = value;
        self
    }

    pub fn rescore(mut self, value: bool) -> Self {
        self.rescore = value;
        self
    }

    pub fn rescore_limit(mut self, value: u64) -> Self {
        self.rescore_limit = value;
        self
    }

    pub fn on_disk_payload(mut self, value: bool) -> Self {
        self.on_disk_payload = value;
        self
    }

    /// Validates that the config is internally consistent.
    ///
    /// When rescoring is enabled, this checks that `rescore_limit` is reasonable
    /// for the expected query limits. The oversampling ratio (`rescore_limit / limit`)
    /// should be >= 1.0 for rescoring to be effective.
    ///
    /// # Arguments
    /// * `expected_limit` - The typical query limit that will be used
    ///
    /// # Returns
    /// * `Ok(())` if the config is valid
    /// * `Err(String)` with a description of the issue if invalid
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

    pub fn estimate_ram_bytes(&self, num_vectors: u64) -> u64 {
        num_vectors * BQ_BYTES_PER_VECTOR as u64
    }

    pub fn estimate_savings_bytes(&self, num_vectors: u64) -> u64 {
        let original = num_vectors * ORIGINAL_BYTES_PER_VECTOR as u64;
        let compressed = self.estimate_ram_bytes(num_vectors);
        original - compressed
    }
}

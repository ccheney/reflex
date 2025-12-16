use crate::embedding::sinter::SinterConfig;
use crate::vectordb::bq::{BQ_COLLECTION_NAME, BQ_VECTOR_SIZE, BqConfig};

use super::error::{L2CacheError, L2CacheResult};

/// Default number of candidates fetched from the binary-quantized index.
pub const DEFAULT_TOP_K_BQ: u64 = 50;
/// Default number of candidates kept after full-precision rescoring.
pub const DEFAULT_TOP_K_FINAL: usize = crate::vectordb::rescoring::DEFAULT_TOP_K;
/// Default collection name used for L2 (binary-quantized collection).
pub const L2_COLLECTION_NAME: &str = BQ_COLLECTION_NAME;
/// Default vector size used for L2 (binary-quantized vector size).
pub const L2_VECTOR_SIZE: u64 = BQ_VECTOR_SIZE;

#[derive(Debug, Clone)]
/// Configuration for the L2 semantic cache.
pub struct L2Config {
    /// Number of BQ candidates to fetch before rescoring.
    pub top_k_bq: u64,
    /// Number of candidates to keep after rescoring.
    pub top_k_final: usize,
    /// Vector collection name.
    pub collection_name: String,
    /// Vector size configured in the backend collection.
    pub vector_size: u64,
    /// If true, validate vector dimensions at runtime.
    pub validate_dimensions: bool,
    /// Embedding model configuration.
    pub embedder_config: SinterConfig,
    /// Binary-quantization backend configuration.
    pub bq_config: BqConfig,
}

impl Default for L2Config {
    fn default() -> Self {
        Self {
            top_k_bq: DEFAULT_TOP_K_BQ,
            top_k_final: DEFAULT_TOP_K_FINAL,
            collection_name: L2_COLLECTION_NAME.to_string(),
            vector_size: L2_VECTOR_SIZE,
            validate_dimensions: true,
            embedder_config: SinterConfig::default(),
            bq_config: BqConfig::default(),
        }
    }
}

impl L2Config {
    /// Creates a config overriding the `top_k_*` settings.
    pub fn with_top_k(top_k_bq: u64, top_k_final: usize) -> Self {
        Self {
            top_k_bq,
            top_k_final,
            ..Default::default()
        }
    }

    /// Sets the collection name.
    pub fn collection_name(mut self, name: &str) -> Self {
        self.collection_name = name.to_string();
        self
    }

    /// Sets the embedder config.
    pub fn embedder_config(mut self, config: SinterConfig) -> Self {
        self.embedder_config = config;
        self
    }

    /// Sets the binary-quantization config.
    pub fn bq_config(mut self, config: BqConfig) -> Self {
        self.bq_config = config;
        self
    }

    /// Sets the backend vector size.
    pub fn vector_size(mut self, size: u64) -> Self {
        self.vector_size = size;
        self
    }

    /// Validates basic invariants for the L2 cache.
    pub fn validate(&self) -> L2CacheResult<()> {
        if self.top_k_bq == 0 {
            return Err(L2CacheError::ConfigError {
                reason: "top_k_bq must be > 0".to_string(),
            });
        }
        if self.top_k_final == 0 {
            return Err(L2CacheError::ConfigError {
                reason: "top_k_final must be > 0".to_string(),
            });
        }
        if self.top_k_final as u64 > self.top_k_bq {
            return Err(L2CacheError::ConfigError {
                reason: format!(
                    "top_k_final ({}) cannot be greater than top_k_bq ({})",
                    self.top_k_final, self.top_k_bq
                ),
            });
        }
        Ok(())
    }
}

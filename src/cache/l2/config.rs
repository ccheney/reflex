use crate::embedding::sinter::SinterConfig;
use crate::vectordb::bq::{BQ_COLLECTION_NAME, BQ_VECTOR_SIZE, BqConfig};

use super::error::{L2CacheError, L2CacheResult};

pub const DEFAULT_TOP_K_BQ: u64 = 50;
pub const DEFAULT_TOP_K_FINAL: usize = crate::vectordb::rescoring::DEFAULT_TOP_K;
pub const L2_COLLECTION_NAME: &str = BQ_COLLECTION_NAME;
pub const L2_VECTOR_SIZE: u64 = BQ_VECTOR_SIZE;

#[derive(Debug, Clone)]
pub struct L2Config {
    pub top_k_bq: u64,
    pub top_k_final: usize,
    pub collection_name: String,
    pub vector_size: u64,
    pub validate_dimensions: bool,
    pub embedder_config: SinterConfig,
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
    pub fn with_top_k(top_k_bq: u64, top_k_final: usize) -> Self {
        Self {
            top_k_bq,
            top_k_final,
            ..Default::default()
        }
    }

    pub fn collection_name(mut self, name: &str) -> Self {
        self.collection_name = name.to_string();
        self
    }

    pub fn embedder_config(mut self, config: SinterConfig) -> Self {
        self.embedder_config = config;
        self
    }

    pub fn bq_config(mut self, config: BqConfig) -> Self {
        self.bq_config = config;
        self
    }

    pub fn vector_size(mut self, size: u64) -> Self {
        self.vector_size = size;
        self
    }

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

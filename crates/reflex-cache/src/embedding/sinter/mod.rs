//! Sinter embedder (GGUF + tokenizer).
//!
//! Use [`SinterConfig::stub`] for tests/examples without model files.

/// Sinter configuration.
pub mod config;
pub(crate) mod model;

#[cfg(test)]
mod tests;

pub use config::{SINTER_EMBEDDING_DIM, SINTER_MAX_SEQ_LEN, SinterConfig};

use std::sync::Arc;

use candle_core::{Device, IndexOp, Tensor};
use half::f16;
use parking_lot::Mutex;
use tracing::{debug, info, warn};

use crate::embedding::device::select_device;
use crate::embedding::error::EmbeddingError;
use crate::embedding::utils::load_tokenizer;

use model::Qwen2ForEmbedding;

enum EmbedderBackend {
    Model {
        model: Arc<Mutex<Qwen2ForEmbedding>>,
        tokenizer: Arc<tokenizers::Tokenizer>,
        device: Device,
    },
    Stub {
        device: Device,
    },
}

/// Embedding generator for semantic search (supports stub mode).
pub struct SinterEmbedder {
    backend: EmbedderBackend,
    config: SinterConfig,
}

impl std::fmt::Debug for SinterEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SinterEmbedder")
            .field(
                "backend",
                &match &self.backend {
                    EmbedderBackend::Model { device, .. } => format!("Model({:?})", device),
                    EmbedderBackend::Stub { device } => format!("Stub({:?})", device),
                },
            )
            .field("embedding_dim", &self.config.embedding_dim)
            .field("max_seq_len", &self.config.max_seq_len)
            .finish()
    }
}

impl SinterEmbedder {
    /// Loads the embedder from a config (stub mode is supported).
    pub fn load(config: SinterConfig) -> Result<Self, EmbeddingError> {
        config.validate()?;

        let device = select_device()?;
        debug!(?device, "Selected compute device for Sinter");

        if config.testing_stub {
            warn!("Sinter running in STUB mode (testing only)");
            return Ok(Self {
                backend: EmbedderBackend::Stub { device },
                config,
            });
        }

        if !config.model_available() || !config.tokenizer_available() {
            return Err(EmbeddingError::ModelNotFound {
                path: config.model_path.clone(),
            });
        }

        let (model, tokenizer) = Self::load_model(&config, &device)?;

        info!(
            model_path = %config.model_path.display(),
            embedding_dim = config.embedding_dim,
            max_seq_len = config.max_seq_len,
            hidden_size = model.config().hidden_size,
            num_layers = model.config().num_layers,
            "Sinter model loaded successfully (full transformer)"
        );

        Ok(Self {
            backend: EmbedderBackend::Model {
                model: Arc::new(Mutex::new(model)),
                tokenizer: Arc::new(tokenizer),
                device,
            },
            config,
        })
    }

    fn load_model(
        config: &SinterConfig,
        device: &Device,
    ) -> Result<(Qwen2ForEmbedding, tokenizers::Tokenizer), EmbeddingError> {
        let tokenizer = load_tokenizer(&config.tokenizer_path).map_err(|e| {
            EmbeddingError::TokenizationFailed {
                reason: format!("Failed to load tokenizer: {}", e),
            }
        })?;

        let mut model_file = std::fs::File::open(&config.model_path)?;
        let model_content = candle_core::quantized::gguf_file::Content::read(&mut model_file)
            .map_err(|e| EmbeddingError::ModelLoadFailed {
                reason: format!("Failed to read GGUF content: {}", e),
            })?;

        let model = Qwen2ForEmbedding::from_gguf(
            model_content,
            &mut model_file,
            device,
            config.max_seq_len,
        )
        .map_err(|e| EmbeddingError::ModelLoadFailed {
            reason: format!("Failed to load Qwen2 model: {}", e),
        })?;

        // Validate embedding dimension
        if config.embedding_dim > model.config().hidden_size {
            return Err(EmbeddingError::InvalidConfig {
                reason: format!(
                    "embedding_dim ({}) exceeds model hidden_size ({})",
                    config.embedding_dim,
                    model.config().hidden_size
                ),
            });
        }

        info!(
            hidden_size = model.config().hidden_size,
            num_layers = model.config().num_layers,
            "Qwen2 transformer loaded"
        );

        Ok((model, tokenizer))
    }

    /// Generates an embedding for a single string.
    pub fn embed(&self, text: &str) -> Result<Vec<f16>, EmbeddingError> {
        match &self.backend {
            EmbedderBackend::Model {
                model,
                tokenizer,
                device,
            } => self.embed_with_model(text, model, tokenizer, device),
            EmbedderBackend::Stub { .. } => self.embed_stub(text),
        }
    }

    /// Generates embeddings for a batch of strings.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f16>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        match &self.backend {
            EmbedderBackend::Model {
                model,
                tokenizer,
                device,
            } => self.embed_batch_with_model(texts, model, tokenizer, device),
            EmbedderBackend::Stub { .. } => {
                texts.iter().map(|text| self.embed_stub(text)).collect()
            }
        }
    }

    fn embed_with_model(
        &self,
        text: &str,
        model: &Arc<Mutex<Qwen2ForEmbedding>>,
        tokenizer: &tokenizers::Tokenizer,
        device: &Device,
    ) -> Result<Vec<f16>, EmbeddingError> {
        let encoding =
            tokenizer
                .encode(text, true)
                .map_err(|e| EmbeddingError::TokenizationFailed {
                    reason: e.to_string(),
                })?;

        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        if tokens.is_empty() {
            return Ok(vec![f16::from_f32(0.0); self.config.embedding_dim]);
        }

        if tokens.len() > self.config.max_seq_len {
            tokens.truncate(self.config.max_seq_len);
        }

        debug!(
            text_len = text.len(),
            token_count = tokens.len(),
            "Generating embedding (transformer forward pass)"
        );

        // Create input tensor: [1, seq_len]
        let input_ids = Tensor::new(&tokens[..], device)
            .map_err(|e| EmbeddingError::InferenceFailed {
                reason: format!("Failed to create input tensor: {}", e),
            })?
            .unsqueeze(0)
            .map_err(|e| EmbeddingError::InferenceFailed {
                reason: format!("Failed to unsqueeze input: {}", e),
            })?;

        // Run full transformer forward pass
        let hidden_states =
            model
                .lock()
                .forward(&input_ids)
                .map_err(|e| EmbeddingError::InferenceFailed {
                    reason: format!("Transformer forward pass failed: {}", e),
                })?;

        // Last-token pooling: extract the final token's hidden state
        // hidden_states shape: [1, seq_len, hidden_size]
        let last_idx = tokens.len() - 1;
        let embedding = hidden_states
            .i((0, last_idx, ..self.config.embedding_dim))
            .map_err(|e| EmbeddingError::InferenceFailed {
                reason: format!("Failed to extract last token embedding: {}", e),
            })?
            .to_vec1::<f32>()
            .map_err(|e| EmbeddingError::InferenceFailed {
                reason: format!("Failed to convert embedding to vec: {}", e),
            })?;

        Ok(self.normalize_and_convert_f16(embedding))
    }

    fn embed_batch_with_model(
        &self,
        texts: &[&str],
        model: &Arc<Mutex<Qwen2ForEmbedding>>,
        tokenizer: &tokenizers::Tokenizer,
        device: &Device,
    ) -> Result<Vec<Vec<f16>>, EmbeddingError> {
        // Process sequentially for now (proper batching would need padding)
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed_with_model(text, model, tokenizer, device)?);
        }
        Ok(results)
    }

    fn embed_stub(&self, text: &str) -> Result<Vec<f16>, EmbeddingError> {
        use std::hash::{DefaultHasher, Hash, Hasher};

        debug!(text_len = text.len(), "Generating stub embedding");

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();

        let mut embedding = Vec::with_capacity(self.config.embedding_dim);
        let mut state = seed;

        for _ in 0..self.config.embedding_dim {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let value = ((state >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0;
            embedding.push(value);
        }

        let result = self.normalize_and_convert_f16(embedding);

        Ok(result)
    }

    fn normalize_and_convert_f16(&self, mut embedding: Vec<f32>) -> Vec<f16> {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        embedding.into_iter().map(f16::from_f32).collect()
    }

    /// Returns the configured output embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Returns `true` if running in stub mode.
    pub fn is_stub(&self) -> bool {
        matches!(self.backend, EmbedderBackend::Stub { .. })
    }

    /// Returns `true` if a model is loaded.
    pub fn has_model(&self) -> bool {
        matches!(self.backend, EmbedderBackend::Model { .. })
    }

    /// Returns the embedder configuration.
    pub fn config(&self) -> &SinterConfig {
        &self.config
    }
}

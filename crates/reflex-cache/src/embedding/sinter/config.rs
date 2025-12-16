use std::path::PathBuf;

use crate::embedding::error::EmbeddingError;

/// Default Sinter embedding dimension.
pub const SINTER_EMBEDDING_DIM: usize = crate::constants::DEFAULT_EMBEDDING_DIM;

/// Default Sinter max sequence length.
pub const SINTER_MAX_SEQ_LEN: usize = crate::constants::DEFAULT_MAX_SEQ_LEN;

#[derive(Debug, Clone)]
/// Configuration for [`SinterEmbedder`](super::SinterEmbedder).
pub struct SinterConfig {
    /// Path to the GGUF model file.
    pub model_path: PathBuf,
    /// Path to `tokenizer.json`.
    pub tokenizer_path: PathBuf,
    /// Max tokens to consider.
    pub max_seq_len: usize,
    /// Output embedding dimension.
    pub embedding_dim: usize,
    /// If true, run in deterministic stub mode (no model files required).
    pub testing_stub: bool,
}

impl Default for SinterConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            tokenizer_path: PathBuf::new(),
            max_seq_len: SINTER_MAX_SEQ_LEN,
            embedding_dim: SINTER_EMBEDDING_DIM,
            testing_stub: false,
        }
    }
}

impl SinterConfig {
    /// Env var used to locate the model file.
    pub const ENV_MODEL_PATH: &'static str = "REFLEX_MODEL_PATH";
    /// Env var used to locate the tokenizer file.
    pub const ENV_TOKENIZER_PATH: &'static str = "REFLEX_TOKENIZER_PATH";

    /// Loads config from environment variables (missing values become empty paths).
    pub fn from_env() -> Result<Self, EmbeddingError> {
        let model_path = std::env::var(Self::ENV_MODEL_PATH)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .unwrap_or_default();

        let tokenizer_path = std::env::var(Self::ENV_TOKENIZER_PATH)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                if !model_path.as_os_str().is_empty() {
                    let parent = model_path.parent().unwrap_or(model_path.as_path());
                    parent.join("tokenizer.json")
                } else {
                    PathBuf::new()
                }
            });

        Ok(Self {
            model_path,
            tokenizer_path,
            ..Default::default()
        })
    }

    /// Creates a config for a model file, inferring `tokenizer.json` from its directory.
    pub fn new<P: Into<PathBuf>>(model_path: P) -> Self {
        let model_path = model_path.into();
        let tokenizer_path = model_path
            .parent()
            .map(|p| p.join("tokenizer.json"))
            .unwrap_or_default();

        Self {
            model_path,
            tokenizer_path,
            ..Default::default()
        }
    }

    /// Creates a stub config (no model files; produces deterministic embeddings).
    pub fn stub() -> Self {
        Self {
            testing_stub: true,
            ..Default::default()
        }
    }

    /// Validates required fields for non-stub mode.
    pub fn validate(&self) -> Result<(), EmbeddingError> {
        if self.testing_stub {
            return Ok(());
        }

        if self.model_path.as_os_str().is_empty() {
            return Err(EmbeddingError::InvalidConfig {
                reason: "model_path is required (stubbing is disabled)".to_string(),
            });
        }

        if !self.model_path.exists() {
            return Err(EmbeddingError::ModelNotFound {
                path: self.model_path.clone(),
            });
        }

        Ok(())
    }

    /// Returns `true` if the model file path exists.
    pub fn model_available(&self) -> bool {
        !self.model_path.as_os_str().is_empty() && self.model_path.exists()
    }

    /// Returns `true` if the tokenizer path exists.
    pub fn tokenizer_available(&self) -> bool {
        !self.tokenizer_path.as_os_str().is_empty() && self.tokenizer_path.exists()
    }
}

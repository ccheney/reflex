use std::path::PathBuf;

/// Default verification threshold (cross-encoder score).
pub const DEFAULT_THRESHOLD: f32 = crate::constants::DEFAULT_VERIFICATION_THRESHOLD;

/// Maximum sequence length used for reranker tokenization.
pub const MAX_SEQ_LEN: usize = 512;

#[derive(Debug, Clone)]
/// Configuration for [`Reranker`](super::Reranker).
pub struct RerankerConfig {
    /// Directory containing `config.json`, `model.safetensors`, and tokenizer files.
    pub model_path: Option<PathBuf>,

    /// Minimum score to consider a candidate verified.
    pub threshold: f32,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            threshold: DEFAULT_THRESHOLD,
        }
    }
}

impl RerankerConfig {
    /// Creates a config for a model directory.
    pub fn new<P: Into<PathBuf>>(model_path: P) -> Self {
        Self {
            model_path: Some(model_path.into()),
            threshold: DEFAULT_THRESHOLD,
        }
    }

    /// Creates a config that runs without a model (stub scoring).
    pub fn stub() -> Self {
        Self {
            model_path: None,
            threshold: DEFAULT_THRESHOLD,
        }
    }

    /// Sets the threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&threshold),
            "threshold must be between 0.0 and 1.0"
        );
        self.threshold = threshold;
        self
    }

    /// Validates basic invariants.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.threshold) {
            return Err(format!(
                "threshold must be between 0.0 and 1.0, got {}",
                self.threshold
            ));
        }

        if let Some(ref path) = self.model_path
            && path.as_os_str().is_empty()
        {
            return Err("model_path cannot be empty when provided".to_string());
        }

        Ok(())
    }

    /// Loads config from `REFLEX_RERANKER_PATH` and `REFLEX_RERANKER_THRESHOLD`.
    pub fn from_env() -> Self {
        let model_path = std::env::var("REFLEX_RERANKER_PATH")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .map(PathBuf::from);

        let threshold = std::env::var("REFLEX_RERANKER_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_THRESHOLD);

        Self {
            model_path,
            threshold,
        }
    }
}

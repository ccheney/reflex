use std::path::PathBuf;

pub const DEFAULT_THRESHOLD: f32 = crate::constants::DEFAULT_VERIFICATION_THRESHOLD;

pub const MAX_SEQ_LEN: usize = 512;

#[derive(Debug, Clone)]
pub struct RerankerConfig {
    pub model_path: Option<PathBuf>,

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
    pub fn new<P: Into<PathBuf>>(model_path: P) -> Self {
        Self {
            model_path: Some(model_path.into()),
            threshold: DEFAULT_THRESHOLD,
        }
    }

    pub fn stub() -> Self {
        Self {
            model_path: None,
            threshold: DEFAULT_THRESHOLD,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&threshold),
            "threshold must be between 0.0 and 1.0"
        );
        self.threshold = threshold;
        self
    }

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

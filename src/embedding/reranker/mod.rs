pub mod config;
pub mod error;

#[cfg(test)]
mod tests;

pub use config::{DEFAULT_THRESHOLD, MAX_SEQ_LEN, RerankerConfig};
pub use error::RerankerError;

use crate::embedding::bert::BertClassifier;
use candle_core::Tensor;
use tokenizers::Tokenizer;
use tracing::{debug, info};

use crate::embedding::device::select_device;
use crate::embedding::utils::load_tokenizer_with_truncation;

pub struct Reranker {
    device: candle_core::Device,
    config: RerankerConfig,
    model_loaded: bool,
    model: Option<BertClassifier>,
    tokenizer: Option<Tokenizer>,
}

impl std::fmt::Debug for Reranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reranker")
            .field("device", &format!("{:?}", self.device))
            .field("config", &self.config)
            .field("model_loaded", &self.model_loaded)
            .finish()
    }
}

impl Reranker {
    pub fn load(config: RerankerConfig) -> Result<Self, RerankerError> {
        if let Err(msg) = config.validate() {
            return Err(RerankerError::InvalidConfig { reason: msg });
        }

        let device = select_device()?;
        debug!(?device, "Selected compute device for reranker");

        if let Some(ref model_path) = config.model_path {
            if !model_path.exists() {
                return Err(RerankerError::ModelLoadFailed {
                    reason: format!("Reranker model path not found: {}", model_path.display()),
                });
            }

            let config_path = model_path.join("config.json");
            if !config_path.exists() {
                return Err(RerankerError::ModelLoadFailed {
                    reason: format!("Missing config.json in {}", model_path.display()),
                });
            }

            let weights_path = model_path.join("model.safetensors");
            if !weights_path.exists() {
                return Err(RerankerError::ModelLoadFailed {
                    reason: format!("Missing model.safetensors in {}", model_path.display()),
                });
            }

            info!(
                model_path = %model_path.display(),
                threshold = config.threshold,
                "Loading reranker model"
            );

            let model = BertClassifier::load(model_path, &device).map_err(|e| {
                RerankerError::ModelLoadFailed {
                    reason: format!("Failed to load BERT model: {}", e),
                }
            })?;

            let tokenizer =
                load_tokenizer_with_truncation(model_path, MAX_SEQ_LEN).map_err(|e| {
                    RerankerError::ModelLoadFailed {
                        reason: format!("Failed to load tokenizer: {}", e),
                    }
                })?;

            info!(
                threshold = config.threshold,
                "Reranker model loaded successfully"
            );

            Ok(Self {
                device,
                config,
                model_loaded: true,
                model: Some(model),
                tokenizer: Some(tokenizer),
            })
        } else {
            info!("No reranker model path configured, operating in stub mode");
            Ok(Self::create_stub(device, config))
        }
    }

    pub fn stub() -> Result<Self, RerankerError> {
        Self::load(RerankerConfig::stub())
    }

    fn create_stub(device: candle_core::Device, config: RerankerConfig) -> Self {
        Self {
            device,
            config,
            model_loaded: false,
            model: None,
            tokenizer: None,
        }
    }

    pub fn score(&self, query: &str, candidate: &str) -> Result<f32, RerankerError> {
        debug!(
            query_len = query.len(),
            candidate_len = candidate.len(),
            model_loaded = self.model_loaded,
            "Scoring query-candidate pair"
        );

        if let (Some(model), Some(tokenizer)) = (&self.model, &self.tokenizer) {
            let tokens = tokenizer.encode((query, candidate), true).map_err(|e| {
                RerankerError::TokenizationFailed {
                    reason: e.to_string(),
                }
            })?;

            let token_ids = tokens.get_ids();
            let token_ids = Tensor::new(token_ids, &self.device)
                .map_err(RerankerError::from)?
                .unsqueeze(0)
                .map_err(RerankerError::from)?;

            let type_ids = tokens.get_type_ids();
            let type_ids = Tensor::new(type_ids, &self.device)
                .map_err(RerankerError::from)?
                .unsqueeze(0)
                .map_err(RerankerError::from)?;

            // Use the tokenizer's attention mask to properly handle padding tokens.
            // Previously used ones_like() which is incorrect when padding is present.
            let attention_mask_data = tokens.get_attention_mask();
            let attention_mask = Tensor::new(attention_mask_data, &self.device)
                .map_err(RerankerError::from)?
                .unsqueeze(0)
                .map_err(RerankerError::from)?;

            let logits = model
                .forward(&token_ids, &type_ids, Some(&attention_mask))
                .map_err(|e| RerankerError::InferenceFailed {
                    reason: e.to_string(),
                })?;

            let score = logits
                .flatten_all()
                .map_err(RerankerError::from)?
                .to_vec1::<f32>()
                .map_err(RerankerError::from)?[0];
            return Ok(score);
        }

        let score = self.compute_placeholder_score(query, candidate);

        debug!(score = score, "Computed score (stub)");

        Ok(score)
    }

    pub fn rerank(
        &self,
        query: &str,
        candidates: &[&str],
    ) -> Result<Vec<(usize, f32)>, RerankerError> {
        debug!(
            query_len = query.len(),
            num_candidates = candidates.len(),
            "Reranking candidates"
        );

        let mut scored: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(idx, candidate)| {
                let score = self.score(query, candidate)?;
                Ok((idx, score))
            })
            .collect::<Result<Vec<_>, RerankerError>>()?;

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        debug!(
            top_score = scored.first().map(|(_, s)| *s),
            "Reranking complete"
        );

        Ok(scored)
    }

    pub fn rerank_with_threshold(
        &self,
        query: &str,
        candidates: &[&str],
    ) -> Result<Vec<(usize, f32)>, RerankerError> {
        let ranked = self.rerank(query, candidates)?;
        let threshold = self.config.threshold;

        let filtered: Vec<_> = ranked
            .into_iter()
            .filter(|(_, score)| *score > threshold)
            .collect();

        debug!(
            threshold = threshold,
            hits = filtered.len(),
            total = candidates.len(),
            "Filtered by threshold"
        );

        Ok(filtered)
    }

    pub fn is_model_loaded(&self) -> bool {
        self.model_loaded
    }

    pub fn threshold(&self) -> f32 {
        self.config.threshold
    }

    pub fn config(&self) -> &RerankerConfig {
        &self.config
    }

    pub fn device(&self) -> &candle_core::Device {
        &self.device
    }

    pub fn is_hit(&self, score: f32) -> bool {
        score > self.config.threshold
    }

    fn compute_placeholder_score(&self, query: &str, candidate: &str) -> f32 {
        use std::collections::HashSet;

        let stop_words: HashSet<&str> = [
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
            "shall", "can", "need", "dare", "ought", "used", "to", "of", "in", "for", "on", "with",
            "at", "by", "from", "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "again", "further", "then", "once", "here", "there",
            "when", "where", "why", "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "and", "but", "if", "or", "because", "until", "while", "what", "which", "who", "whom",
            "this", "that", "these", "those", "am", "it", "its",
        ]
        .into_iter()
        .collect();

        let query_lower = query.to_lowercase();
        let query_words: HashSet<&str> = query_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty() && !stop_words.contains(w))
            .collect();

        let candidate_lower = candidate.to_lowercase();
        let candidate_words: HashSet<&str> = candidate_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty() && !stop_words.contains(w))
            .collect();

        if query_words.is_empty() {
            let len_ratio = (query.len().min(candidate.len()) as f32)
                / (query.len().max(candidate.len()).max(1) as f32);
            return len_ratio * 0.3;
        }

        let matches = query_words.intersection(&candidate_words).count();
        let recall = matches as f32 / query_words.len() as f32;

        let union = query_words.union(&candidate_words).count();
        let jaccard = if union > 0 {
            matches as f32 / union as f32
        } else {
            0.0
        };

        let base_score = 0.6 * recall + 0.4 * jaccard;

        let normalized = 1.0 / (1.0 + (-8.0 * (base_score - 0.5)).exp());

        normalized.clamp(0.0, 1.0)
    }
}

//! Full-precision rescoring of candidates.

use half::f16;
use std::cmp::Ordering;
use thiserror::Error;
use tracing::warn;

use crate::storage::CacheEntry;

/// Default number of rescored candidates returned.
pub const DEFAULT_TOP_K: usize = 5;

/// Default embedding dimension used for validation.
pub const DEFAULT_EMBEDDING_DIM: usize = crate::constants::DEFAULT_EMBEDDING_DIM;

/// Expected embedding byte length (f16).
pub const EMBEDDING_BYTES: usize = crate::constants::EMBEDDING_F16_BYTES;

#[derive(Debug, Error)]
/// Errors returned by rescoring.
pub enum RescoringError {
    /// Query dimension mismatch.
    #[error("invalid query dimension: expected {expected}, got {actual}")]
    InvalidQueryDimension {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },

    /// Candidate embedding byte length mismatch.
    #[error("invalid embedding size for candidate {id}: expected {expected} bytes, got {actual}")]
    InvalidEmbeddingSize {
        /// Candidate id.
        id: u64,
        /// Expected bytes.
        expected: usize,
        /// Actual bytes.
        actual: usize,
    },

    #[error("no candidates provided for rescoring")]
    /// No candidates were provided.
    NoCandidates,
}

/// Convenience result type for rescoring.
pub type RescoringResult<T> = Result<T, RescoringError>;

#[derive(Debug, Clone)]
/// Candidate with optional BQ score, ready for full-precision rescoring.
pub struct CandidateEntry {
    /// Point id.
    pub id: u64,
    /// Full cache entry.
    pub entry: CacheEntry,
    /// Optional BQ score from the first-stage search.
    pub bq_score: Option<f32>,
}

impl CandidateEntry {
    /// Creates a candidate entry without a BQ score.
    pub fn new(id: u64, entry: CacheEntry) -> Self {
        Self {
            id,
            entry,
            bq_score: None,
        }
    }

    /// Creates a candidate entry with a BQ score.
    pub fn with_bq_score(id: u64, entry: CacheEntry, bq_score: f32) -> Self {
        Self {
            id,
            entry,
            bq_score: Some(bq_score),
        }
    }

    /// Views the candidate embedding as an f16 slice (if the bytes are valid).
    pub fn embedding_as_f16(&self) -> Option<&[f16]> {
        bytes_to_f16_slice(&self.entry.embedding)
    }
}

#[derive(Debug, Clone)]
/// Rescored candidate (full-precision cosine similarity).
pub struct ScoredCandidate {
    /// Point id.
    pub id: u64,
    /// Full cache entry.
    pub entry: CacheEntry,
    /// Full-precision score.
    pub score: f32,
    /// Optional BQ score for debugging/analysis.
    pub bq_score: Option<f32>,
}

impl ScoredCandidate {
    /// Returns `score - bq_score` if a BQ score is present.
    pub fn score_delta(&self) -> Option<f32> {
        self.bq_score.map(|bq| self.score - bq)
    }
}

#[derive(Debug, Clone)]
/// Configuration for [`VectorRescorer`].
pub struct RescorerConfig {
    /// Number of candidates returned after rescoring.
    pub top_k: usize,
    /// If true, validate dimensions before scoring.
    pub validate_dimensions: bool,
}

impl Default for RescorerConfig {
    fn default() -> Self {
        Self {
            top_k: DEFAULT_TOP_K,
            validate_dimensions: true,
        }
    }
}

impl RescorerConfig {
    /// Creates a config overriding `top_k`.
    pub fn with_top_k(top_k: usize) -> Self {
        Self {
            top_k,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone)]
/// Full-precision rescoring of candidates (cosine similarity).
pub struct VectorRescorer {
    config: RescorerConfig,
}

impl VectorRescorer {
    /// Creates a rescorer with default config.
    pub fn new() -> Self {
        Self {
            config: RescorerConfig::default(),
        }
    }

    /// Creates a rescorer overriding `top_k`.
    pub fn with_top_k(top_k: usize) -> Self {
        Self {
            config: RescorerConfig::with_top_k(top_k),
        }
    }

    /// Creates a rescorer with an explicit config.
    pub fn with_config(config: RescorerConfig) -> Self {
        Self { config }
    }

    /// Returns the active config.
    pub fn config(&self) -> &RescorerConfig {
        &self.config
    }

    /// Rescores candidates and returns the top-k results.
    pub fn rescore(
        &self,
        query: &[f16],
        candidates: Vec<CandidateEntry>,
    ) -> RescoringResult<Vec<ScoredCandidate>> {
        if self.config.validate_dimensions && query.len() != DEFAULT_EMBEDDING_DIM {
            return Err(RescoringError::InvalidQueryDimension {
                expected: DEFAULT_EMBEDDING_DIM,
                actual: query.len(),
            });
        }

        if candidates.is_empty() {
            return Err(RescoringError::NoCandidates);
        }

        let mut scored: Vec<ScoredCandidate> = candidates
            .into_iter()
            .filter_map(|candidate| {
                let embedding = match candidate.embedding_as_f16() {
                    Some(emb) => emb,
                    None => {
                        warn!(
                            candidate_id = candidate.id,
                            "Dropping candidate: failed to parse embedding as F16"
                        );
                        return None;
                    }
                };

                if self.config.validate_dimensions && embedding.len() != DEFAULT_EMBEDDING_DIM {
                    warn!(
                        candidate_id = candidate.id,
                        expected_dim = DEFAULT_EMBEDDING_DIM,
                        actual_dim = embedding.len(),
                        "Dropping candidate: embedding dimension mismatch"
                    );
                    return None;
                }

                let score = cosine_similarity_f16(query, embedding);

                Some(ScoredCandidate {
                    id: candidate.id,
                    entry: candidate.entry,
                    score,
                    bq_score: candidate.bq_score,
                })
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        scored.truncate(self.config.top_k);

        Ok(scored)
    }

    /// Like [`Self::rescore`], but takes query bytes (little-endian f16).
    pub fn rescore_from_bytes(
        &self,
        query_bytes: &[u8],
        candidates: Vec<CandidateEntry>,
    ) -> RescoringResult<Vec<ScoredCandidate>> {
        let query =
            bytes_to_f16_slice(query_bytes).ok_or(RescoringError::InvalidQueryDimension {
                expected: EMBEDDING_BYTES,
                actual: query_bytes.len(),
            })?;

        self.rescore(query, candidates)
    }
}

impl Default for VectorRescorer {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
/// Computes cosine similarity between two f16 vectors.
pub fn cosine_similarity_f16(a: &[f16], b: &[f16]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let (dot, norm_a_sq, norm_b_sq) =
        a.iter()
            .zip(b.iter())
            .fold((0.0f32, 0.0f32, 0.0f32), |(dot, na, nb), (av, bv)| {
                let av = av.to_f32();
                let bv = bv.to_f32();
                (dot + av * bv, na + av * av, nb + bv * bv)
            });

    let norm_a = norm_a_sq.sqrt();
    let norm_b = norm_b_sq.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[inline]
/// Computes cosine similarity between an f16 query and an f32 candidate.
pub fn cosine_similarity_f16_f32(a: &[f16], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot_product = 0.0f32;
    let mut norm_a_sq = 0.0f32;
    let mut norm_b_sq = 0.0f32;

    for (av_f16, &bv) in a.iter().zip(b.iter()) {
        let av = av_f16.to_f32();
        dot_product += av * bv;
        norm_a_sq += av * av;
        norm_b_sq += bv * bv;
    }

    let norm_a = norm_a_sq.sqrt();
    let norm_b = norm_b_sq.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

/// Reinterprets little-endian f16 bytes as an `&[f16]` (no copy).
#[inline]
pub fn bytes_to_f16_slice(bytes: &[u8]) -> Option<&[f16]> {
    bytemuck::try_cast_slice(bytes).ok()
}

#[inline]
/// Views an `&[f16]` as bytes (no copy).
pub fn f16_slice_to_bytes(values: &[f16]) -> &[u8] {
    bytemuck::cast_slice(values)
}

/// Converts `&[f32]` to `Vec<f16>`.
pub fn f32_to_f16_vec(values: &[f32]) -> Vec<f16> {
    values.iter().map(|&v| f16::from_f32(v)).collect()
}

/// Converts `&[f16]` to `Vec<f32>`.
pub fn f16_to_f32_vec(values: &[f16]) -> Vec<f32> {
    values.iter().map(|v| v.to_f32()).collect()
}

#[cfg(test)]
mod tests;

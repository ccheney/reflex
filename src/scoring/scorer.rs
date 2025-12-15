use std::cmp::Ordering;
use tracing::{debug, info};

use crate::embedding::{Reranker, RerankerConfig};
use crate::storage::CacheEntry;

use super::error::ScoringError;
use super::types::{VerificationResult, VerifiedCandidate};

pub struct CrossEncoderScorer {
    reranker: Reranker,
}

impl std::fmt::Debug for CrossEncoderScorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossEncoderScorer")
            .field("reranker", &self.reranker)
            .finish()
    }
}

impl CrossEncoderScorer {
    pub fn new(config: RerankerConfig) -> Result<Self, ScoringError> {
        let reranker = Reranker::load(config)?;
        Ok(Self { reranker })
    }

    pub fn stub() -> Result<Self, ScoringError> {
        Ok(Self {
            reranker: Reranker::stub()?,
        })
    }

    pub fn is_model_loaded(&self) -> bool {
        self.reranker.is_model_loaded()
    }

    pub fn threshold(&self) -> f32 {
        self.reranker.threshold()
    }

    pub fn reranker(&self) -> &Reranker {
        &self.reranker
    }

    pub fn score(&self, query: &str, candidate_text: &str) -> Result<f32, ScoringError> {
        Ok(self.reranker.score(query, candidate_text)?)
    }

    pub fn verify_candidates(
        &self,
        query: &str,
        candidates: Vec<(CacheEntry, f32)>,
    ) -> Result<(Option<CacheEntry>, VerificationResult), ScoringError> {
        if candidates.is_empty() {
            debug!("No candidates provided for verification");
            return Ok((None, VerificationResult::NoCandidates));
        }

        debug!(
            query_len = query.len(),
            num_candidates = candidates.len(),
            "Starting L3 verification"
        );

        let mut verified_candidates = self.score_candidates(query, candidates)?;

        verified_candidates.sort_by(|a, b| {
            b.cross_encoder_score
                .partial_cmp(&a.cross_encoder_score)
                .unwrap_or(Ordering::Equal)
        });

        // SAFETY: candidates is non-empty (checked above), and score_candidates
        // maps 1:1, so verified_candidates is guaranteed non-empty
        let top = &verified_candidates[0];

        debug!(
            top_score = top.cross_encoder_score,
            original_score = top.original_score,
            threshold = self.threshold(),
            "Top candidate after reranking"
        );

        let score = top.cross_encoder_score;

        if score > self.threshold() {
            let entry = top.entry.clone();

            info!(
                score = score,
                threshold = self.threshold(),
                "L3 verification passed - cache hit"
            );

            Ok((Some(entry), VerificationResult::Verified { score }))
        } else {
            debug!(
                score = score,
                threshold = self.threshold(),
                "Top candidate below threshold - cache miss"
            );

            Ok((None, VerificationResult::Rejected { top_score: score }))
        }
    }

    pub fn score_candidates(
        &self,
        query: &str,
        candidates: Vec<(CacheEntry, f32)>,
    ) -> Result<Vec<VerifiedCandidate>, ScoringError> {
        candidates
            .into_iter()
            .map(|(entry, original_score)| {
                let candidate_text = String::from_utf8_lossy(&entry.payload_blob);
                let cross_encoder_score = self.reranker.score(query, &candidate_text)?;

                Ok(VerifiedCandidate::new(
                    entry,
                    cross_encoder_score,
                    original_score,
                ))
            })
            .collect()
    }

    pub fn verify_candidates_with_details(
        &self,
        query: &str,
        candidates: Vec<(CacheEntry, f32)>,
    ) -> Result<(Vec<VerifiedCandidate>, VerificationResult), ScoringError> {
        if candidates.is_empty() {
            return Ok((vec![], VerificationResult::NoCandidates));
        }

        let mut scored = self.score_candidates(query, candidates)?;

        scored.sort_by(|a, b| {
            b.cross_encoder_score
                .partial_cmp(&a.cross_encoder_score)
                .unwrap_or(Ordering::Equal)
        });

        // SAFETY: candidates is non-empty (checked above), and score_candidates
        // maps 1:1, so scored is guaranteed non-empty
        let score = scored[0].cross_encoder_score;
        let result = if score > self.threshold() {
            VerificationResult::Verified { score }
        } else {
            VerificationResult::Rejected { top_score: score }
        };

        Ok((scored, result))
    }

    pub fn rerank_top_n(
        &self,
        query: &str,
        candidates: Vec<(CacheEntry, f32)>,
        top_n: usize,
    ) -> Result<Vec<VerifiedCandidate>, ScoringError> {
        let mut scored = self.score_candidates(query, candidates)?;

        scored.sort_by(|a, b| {
            b.cross_encoder_score
                .partial_cmp(&a.cross_encoder_score)
                .unwrap_or(Ordering::Equal)
        });

        scored.truncate(top_n);
        Ok(scored)
    }
}

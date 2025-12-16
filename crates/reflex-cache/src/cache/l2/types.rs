use crate::vectordb::rescoring::ScoredCandidate;
use half::f16;

#[derive(Debug, Clone)]
/// Result of an L2 semantic lookup.
pub struct L2LookupResult {
    pub(crate) query_embedding: Vec<f16>,
    pub(crate) candidates: Vec<ScoredCandidate>,
    pub(crate) tenant_id: u64,
    pub(crate) bq_candidates_count: usize,
}

impl L2LookupResult {
    /// Creates a new lookup result.
    pub fn new(
        query_embedding: Vec<f16>,
        candidates: Vec<ScoredCandidate>,
        tenant_id: u64,
        bq_candidates_count: usize,
    ) -> Self {
        Self {
            query_embedding,
            candidates,
            tenant_id,
            bq_candidates_count,
        }
    }

    /// Returns the query embedding (f16).
    pub fn query_embedding(&self) -> &[f16] {
        &self.query_embedding
    }

    /// Returns scored candidates (sorted best-first).
    pub fn candidates(&self) -> &[ScoredCandidate] {
        &self.candidates
    }

    /// Consumes the result and returns the candidate vector.
    pub fn into_candidates(self) -> Vec<ScoredCandidate> {
        self.candidates
    }

    /// Returns the tenant id used for the search.
    pub fn tenant_id(&self) -> u64 {
        self.tenant_id
    }

    /// Returns the number of candidates returned by the BQ stage.
    pub fn bq_candidates_count(&self) -> usize {
        self.bq_candidates_count
    }

    /// Returns `true` if any candidates are present.
    pub fn has_candidates(&self) -> bool {
        !self.candidates.is_empty()
    }

    /// Returns the best candidate (if any).
    pub fn best_candidate(&self) -> Option<&ScoredCandidate> {
        self.candidates.first()
    }
}

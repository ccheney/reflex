use crate::vectordb::rescoring::ScoredCandidate;
use half::f16;

#[derive(Debug, Clone)]
pub struct L2LookupResult {
    pub(crate) query_embedding: Vec<f16>,
    pub(crate) candidates: Vec<ScoredCandidate>,
    pub(crate) tenant_id: u64,
    pub(crate) bq_candidates_count: usize,
}

impl L2LookupResult {
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

    pub fn query_embedding(&self) -> &[f16] {
        &self.query_embedding
    }

    pub fn candidates(&self) -> &[ScoredCandidate] {
        &self.candidates
    }

    pub fn into_candidates(self) -> Vec<ScoredCandidate> {
        self.candidates
    }

    pub fn tenant_id(&self) -> u64 {
        self.tenant_id
    }

    pub fn bq_candidates_count(&self) -> usize {
        self.bq_candidates_count
    }

    pub fn has_candidates(&self) -> bool {
        !self.candidates.is_empty()
    }

    pub fn best_candidate(&self) -> Option<&ScoredCandidate> {
        self.candidates.first()
    }
}

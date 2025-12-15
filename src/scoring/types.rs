use crate::cache::ReflexStatus;
use crate::storage::CacheEntry;

#[derive(Debug, Clone, PartialEq)]
pub enum VerificationResult {
    Verified { score: f32 },
    Rejected { top_score: f32 },
    NoCandidates,
}

impl VerificationResult {
    pub fn to_cache_status(&self) -> ReflexStatus {
        match self {
            VerificationResult::Verified { .. } => ReflexStatus::HitL3Verified,
            VerificationResult::Rejected { .. } | VerificationResult::NoCandidates => {
                ReflexStatus::Miss
            }
        }
    }

    pub fn is_verified(&self) -> bool {
        matches!(self, VerificationResult::Verified { .. })
    }

    pub fn score(&self) -> Option<f32> {
        match self {
            VerificationResult::Verified { score }
            | VerificationResult::Rejected { top_score: score } => Some(*score),
            VerificationResult::NoCandidates => None,
        }
    }

    pub fn debug_status(&self) -> &'static str {
        match self {
            VerificationResult::Verified { .. } => "VERIFIED",
            VerificationResult::Rejected { .. } => "REJECTED",
            VerificationResult::NoCandidates => "NO_CANDIDATES",
        }
    }
}

impl std::fmt::Display for VerificationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerificationResult::Verified { score } => {
                write!(f, "VERIFIED (score: {:.4})", score)
            }
            VerificationResult::Rejected { top_score } => {
                write!(f, "REJECTED (top_score: {:.4})", top_score)
            }
            VerificationResult::NoCandidates => write!(f, "NO_CANDIDATES"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VerifiedCandidate {
    pub entry: CacheEntry,
    pub cross_encoder_score: f32,
    pub original_score: f32,
}

impl VerifiedCandidate {
    pub fn new(entry: CacheEntry, cross_encoder_score: f32, original_score: f32) -> Self {
        Self {
            entry,
            cross_encoder_score,
            original_score,
        }
    }

    pub fn exceeds_threshold(&self, threshold: f32) -> bool {
        self.cross_encoder_score > threshold
    }
}

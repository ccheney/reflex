use crate::cache::ReflexStatus;
use crate::storage::CacheEntry;

#[derive(Debug, Clone, PartialEq)]
/// Outcome of L3 verification.
pub enum VerificationResult {
    /// Verified above threshold.
    Verified {
        /// Cross-encoder score.
        score: f32,
    },
    /// Top candidate did not exceed threshold.
    Rejected {
        /// Best cross-encoder score observed.
        top_score: f32,
    },
    /// No candidates were provided.
    NoCandidates,
}

impl VerificationResult {
    /// Maps the result to a [`ReflexStatus`].
    pub fn to_cache_status(&self) -> ReflexStatus {
        match self {
            VerificationResult::Verified { .. } => ReflexStatus::HitL3Verified,
            VerificationResult::Rejected { .. } | VerificationResult::NoCandidates => {
                ReflexStatus::Miss
            }
        }
    }

    /// Returns `true` if verified.
    pub fn is_verified(&self) -> bool {
        matches!(self, VerificationResult::Verified { .. })
    }

    /// Returns the score (if available).
    pub fn score(&self) -> Option<f32> {
        match self {
            VerificationResult::Verified { score }
            | VerificationResult::Rejected { top_score: score } => Some(*score),
            VerificationResult::NoCandidates => None,
        }
    }

    /// Returns a short debug string.
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
/// Candidate annotated with cross-encoder score and original score.
pub struct VerifiedCandidate {
    /// The candidate entry.
    pub entry: CacheEntry,
    /// Cross-encoder score.
    pub cross_encoder_score: f32,
    /// Original (pre-L3) score, if any.
    pub original_score: f32,
}

impl VerifiedCandidate {
    /// Creates a new verified-candidate record.
    pub fn new(entry: CacheEntry, cross_encoder_score: f32, original_score: f32) -> Self {
        Self {
            entry,
            cross_encoder_score,
            original_score,
        }
    }

    /// Returns `true` if `cross_encoder_score` exceeds `threshold`.
    pub fn exceeds_threshold(&self, threshold: f32) -> bool {
        self.cross_encoder_score > threshold
    }
}

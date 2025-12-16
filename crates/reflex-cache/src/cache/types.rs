/// Response header used to report cache status.
pub const REFLEX_STATUS_HEADER: &str = "X-Reflex-Status";
/// Health value for status endpoints.
pub const REFLEX_STATUS_HEALTHY: &str = "healthy";
/// Ready value for status endpoints.
pub const REFLEX_STATUS_READY: &str = "ready";
/// Not-ready value for status endpoints.
pub const REFLEX_STATUS_NOT_READY: &str = "not_ready";
/// Stored value for status endpoints.
pub const REFLEX_STATUS_STORED: &str = "stored";
/// Error value for status endpoints.
pub const REFLEX_STATUS_ERROR: &str = "error";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// High-level cache status used for metrics and response headers.
pub enum ReflexStatus {
    /// L1 exact-match hit.
    HitL1Exact,
    /// L2 semantic hit.
    HitL2Semantic,
    /// L3 verified hit.
    HitL3Verified,
    /// Cache miss.
    Miss,
}

impl ReflexStatus {
    #[inline]
    /// Returns a stable string suitable for the `X-Reflex-Status` header.
    pub fn as_header_value(&self) -> &'static str {
        match self {
            ReflexStatus::HitL1Exact => "HIT_L1_EXACT",
            ReflexStatus::HitL2Semantic => "HIT_L2_SEMANTIC",
            ReflexStatus::HitL3Verified => "HIT_L3_VERIFIED",
            ReflexStatus::Miss => "MISS",
        }
    }

    #[inline]
    /// Returns `true` if this is not a miss.
    pub fn is_hit(&self) -> bool {
        !matches!(self, ReflexStatus::Miss)
    }
}

impl std::fmt::Display for ReflexStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_header_value())
    }
}

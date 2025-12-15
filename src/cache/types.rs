pub const REFLEX_STATUS_HEADER: &str = "X-Reflex-Status";
pub const REFLEX_STATUS_HEALTHY: &str = "healthy";
pub const REFLEX_STATUS_READY: &str = "ready";
pub const REFLEX_STATUS_NOT_READY: &str = "not_ready";
pub const REFLEX_STATUS_STORED: &str = "stored";
pub const REFLEX_STATUS_ERROR: &str = "error";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReflexStatus {
    HitL1Exact,
    HitL2Semantic,
    HitL3Verified,
    Miss,
}

impl ReflexStatus {
    #[inline]
    pub fn as_header_value(&self) -> &'static str {
        match self {
            ReflexStatus::HitL1Exact => "HIT_L1_EXACT",
            ReflexStatus::HitL2Semantic => "HIT_L2_SEMANTIC",
            ReflexStatus::HitL3Verified => "HIT_L3_VERIFIED",
            ReflexStatus::Miss => "MISS",
        }
    }

    #[inline]
    pub fn is_hit(&self) -> bool {
        !matches!(self, ReflexStatus::Miss)
    }
}

impl std::fmt::Display for ReflexStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_header_value())
    }
}

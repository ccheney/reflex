#[derive(Debug, Clone)]
pub enum HydrationResult {
    Success { bytes: u64 },
    NotFound,
    Skipped { reason: String },
}

#[derive(Debug, Clone)]
pub enum DehydrationResult {
    Success { bytes: u64 },
    NoSnapshot,
    Skipped { reason: String },
}

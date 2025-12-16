#[derive(Debug, Clone)]
/// Result of a hydration attempt.
pub enum HydrationResult {
    /// Download succeeded.
    Success {
        /// Downloaded size in bytes.
        bytes: u64,
    },
    /// Snapshot was not found.
    NotFound,
    /// Skipped (disabled or not configured).
    Skipped {
        /// Reason for skipping.
        reason: String,
    },
}

#[derive(Debug, Clone)]
/// Result of a dehydration attempt.
pub enum DehydrationResult {
    /// Upload succeeded.
    Success {
        /// Uploaded size in bytes.
        bytes: u64,
    },
    /// No local snapshot to upload.
    NoSnapshot,
    /// Skipped (disabled or not configured).
    Skipped {
        /// Reason for skipping.
        reason: String,
    },
}

use thiserror::Error;

#[derive(Debug, Error)]
/// Errors returned by vector database operations.
pub enum VectorDbError {
    /// Could not connect to the Qdrant endpoint.
    #[error("failed to connect to Qdrant at '{url}': {message}")]
    ConnectionFailed {
        /// Endpoint URL.
        url: String,
        /// Error message.
        message: String,
    },

    /// Collection creation failed.
    #[error("failed to create collection '{collection}': {message}")]
    CreateCollectionFailed {
        /// Collection name.
        collection: String,
        /// Error message.
        message: String,
    },

    /// Collection does not exist.
    #[error("collection not found: {collection}")]
    CollectionNotFound {
        /// Collection name.
        collection: String,
    },

    /// Upsert failed.
    #[error("failed to upsert points to '{collection}': {message}")]
    UpsertFailed {
        /// Collection name.
        collection: String,
        /// Error message.
        message: String,
    },

    /// Search failed.
    #[error("failed to search in '{collection}': {message}")]
    SearchFailed {
        /// Collection name.
        collection: String,
        /// Error message.
        message: String,
    },

    /// Vector dimension mismatch.
    #[error("invalid vector dimension: expected {expected}, got {actual}")]
    InvalidDimension {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },

    /// Embedding bytes had the wrong length.
    #[error("invalid embedding byte length: expected {expected} bytes, got {actual}")]
    InvalidEmbeddingBytesLength {
        /// Expected byte length.
        expected: usize,
        /// Actual byte length.
        actual: usize,
    },

    /// Delete failed.
    #[error("failed to delete points from '{collection}': {message}")]
    DeleteFailed {
        /// Collection name.
        collection: String,
        /// Error message.
        message: String,
    },
}

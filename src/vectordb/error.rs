use thiserror::Error;

#[derive(Debug, Error)]
pub enum VectorDbError {
    #[error("failed to connect to Qdrant at '{url}': {message}")]
    ConnectionFailed { url: String, message: String },

    #[error("failed to create collection '{collection}': {message}")]
    CreateCollectionFailed { collection: String, message: String },

    #[error("collection not found: {collection}")]
    CollectionNotFound { collection: String },

    #[error("failed to upsert points to '{collection}': {message}")]
    UpsertFailed { collection: String, message: String },

    #[error("failed to search in '{collection}': {message}")]
    SearchFailed { collection: String, message: String },

    #[error("invalid vector dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },

    #[error("invalid embedding byte length: expected {expected} bytes, got {actual}")]
    InvalidEmbeddingBytesLength { expected: usize, actual: usize },

    #[error("failed to delete points from '{collection}': {message}")]
    DeleteFailed { collection: String, message: String },
}

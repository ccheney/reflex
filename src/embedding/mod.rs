//! Embedding generation (Candle).

pub mod bert;
pub mod device;
mod error;
pub mod reranker;
pub mod sinter;
pub mod utils;

pub use error::EmbeddingError;
pub use reranker::{DEFAULT_THRESHOLD, Reranker, RerankerConfig, RerankerError};

pub use sinter::{SINTER_EMBEDDING_DIM, SINTER_MAX_SEQ_LEN, SinterConfig, SinterEmbedder};

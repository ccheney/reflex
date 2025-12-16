//! Embedding + model utilities.
//!
//! - [`sinter`] provides embedding generation.
//! - [`reranker`] provides cross-encoder scoring used by [`crate::scoring`].

/// BERT classifier wrapper used by the reranker.
pub mod bert;
/// Device selection (CPU / Metal / CUDA).
pub mod device;
mod error;
/// Cross-encoder reranker (L3).
pub mod reranker;
/// Sinter embedder (L2 embeddings).
pub mod sinter;
/// Tokenizer/model loading helpers.
pub mod utils;

pub use error::EmbeddingError;
pub use reranker::{DEFAULT_THRESHOLD, Reranker, RerankerConfig, RerankerError};

pub use sinter::{SINTER_EMBEDDING_DIM, SINTER_MAX_SEQ_LEN, SinterConfig, SinterEmbedder};

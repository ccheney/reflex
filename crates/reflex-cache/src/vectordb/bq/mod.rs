//! Binary quantization helpers and Qdrant BQ client.

/// Backend wrapper (real or mock).
pub mod backend;
/// BQ-optimized Qdrant client.
pub mod client;
/// BQ configuration and constants.
pub mod config;
#[cfg(any(test, feature = "mock"))]
/// In-memory mock BQ client (enabled with `mock` feature).
pub mod mock;
/// Quantization utilities.
pub mod utils;

#[cfg(test)]
mod tests;

pub use backend::BqBackend;
pub use client::BqClient;
pub use config::{
    BQ_BYTES_PER_VECTOR, BQ_COLLECTION_NAME, BQ_COMPRESSION_RATIO, BQ_VECTOR_SIZE, BqConfig,
    DEFAULT_RESCORE_CANDIDATES, ORIGINAL_BYTES_PER_VECTOR,
};
#[cfg(any(test, feature = "mock"))]
pub use mock::MockBqClient;
pub use utils::{hamming_distance, quantize_to_binary};

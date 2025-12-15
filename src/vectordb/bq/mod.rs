pub mod backend;
pub mod client;
pub mod config;
#[cfg(any(test, feature = "mock"))]
pub mod mock;
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

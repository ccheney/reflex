//! # Reflex Cache
//!
//! Episodic memory + semantic cache for LLM responses.
//!
//! Reflex sits between a client (agent/server) and an LLM provider.
//!
//! ```text
//! Request → L1 (exact) → L2 (semantic) → L3 (rerank) → Provider
//! ```
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use reflex::Config;
//!
//! # async fn run() -> anyhow::Result<()> {
//! let config = Config::from_env()?;
//! println!("Listening on {}", config.socket_addr());
//! # Ok(())
//! # }
//! ```
//!
//! ## Feature flags
//!
//! | Feature | Purpose |
//! |---------|---------|
//! | `cpu` | CPU-only inference (docs.rs default) |
//! | `metal` | Apple Silicon GPU acceleration |
//! | `cuda` | NVIDIA GPU acceleration |
//! | `mock` | Mock backends for tests/examples |
//!
//! ## Modules
//!
//! - [`cache`] - Tiered cache (L1 exact + L2 semantic)
//! - [`config`] - Environment-backed configuration
//! - [`embedding`] - Embedding + reranker models
//! - [`scoring`] - L3 verification (cross-encoder)
//! - [`storage`] - Persistent cache entry storage
//! - [`vectordb`] - Qdrant + binary quantization utilities
//!
//! Links: repo/issues at the crate `repository` URL.

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod cache;
pub mod config;
pub mod constants;
pub mod embedding;
pub mod gateway;
pub mod hashing;
pub mod lifecycle;
pub mod payload;
pub mod scoring;
pub mod storage;
pub mod vectordb;

pub use cache::{
    BqSearchBackend, DEFAULT_TOP_K_BQ, DEFAULT_TOP_K_FINAL, L2_COLLECTION_NAME, L2_VECTOR_SIZE,
    L2CacheError, L2CacheResult, L2Config, L2LookupResult, L2SemanticCache, L2SemanticCacheHandle,
    NvmeStorageLoader, StorageLoader,
};
#[cfg(any(test, feature = "mock"))]
pub use cache::{MockL2SemanticCache, MockStorageLoader};

#[cfg(any(test, feature = "mock"))]
pub use cache::MockTieredCache;
pub use cache::{
    L1Cache, L1CacheHandle, L1LookupResult, REFLEX_STATUS_ERROR, REFLEX_STATUS_HEADER,
    REFLEX_STATUS_HEALTHY, REFLEX_STATUS_NOT_READY, REFLEX_STATUS_READY, REFLEX_STATUS_STORED,
    ReflexStatus,
};
pub use cache::{TieredCache, TieredCacheHandle, TieredLookupResult};

pub use config::{Config, ConfigError};
pub use constants::{DimConfig, DimValidationError, validate_embedding_dim};
pub use embedding::{
    DEFAULT_THRESHOLD, EmbeddingError, Reranker, RerankerConfig, RerankerError,
    SINTER_EMBEDDING_DIM, SINTER_MAX_SEQ_LEN, SinterConfig, SinterEmbedder,
};
pub use hashing::{hash_context, hash_prompt, hash_tenant_id, hash_to_u64};
pub use lifecycle::{
    ActivityRecorder, DEFAULT_IDLE_TIMEOUT_SECS, DEFAULT_SNAPSHOT_FILENAME, DehydrationResult,
    HydrationResult, LifecycleConfig, LifecycleError, LifecycleManager, LifecycleResult,
    REAPER_CHECK_INTERVAL_SECS,
};
pub use payload::{TauqBatchEncoder, TauqDecoder, TauqEncoder, TauqError};
pub use scoring::{CrossEncoderScorer, ScoringError, VerificationResult, VerifiedCandidate};
pub use storage::CacheEntry;
#[cfg(any(test, feature = "mock"))]
pub use vectordb::bq::MockBqClient;
pub use vectordb::bq::{
    BQ_BYTES_PER_VECTOR, BQ_COLLECTION_NAME, BQ_COMPRESSION_RATIO, BQ_VECTOR_SIZE, BqClient,
    BqConfig, DEFAULT_RESCORE_CANDIDATES, ORIGINAL_BYTES_PER_VECTOR, hamming_distance,
    quantize_to_binary,
};

#[cfg(any(test, feature = "mock"))]
pub use vectordb::MockVectorDbClient;
pub use vectordb::rescoring::{
    CandidateEntry, DEFAULT_EMBEDDING_DIM, DEFAULT_TOP_K, EMBEDDING_BYTES, RescorerConfig,
    RescoringError, RescoringResult, ScoredCandidate, VectorRescorer, bytes_to_f16_slice,
    cosine_similarity_f16, cosine_similarity_f16_f32, f16_slice_to_bytes, f16_to_f32_vec,
    f32_to_f16_vec,
};
pub use vectordb::{
    DEFAULT_COLLECTION_NAME, DEFAULT_VECTOR_SIZE, QdrantClient, SearchResult, VectorDbClient,
    VectorDbError, VectorPoint, embedding_bytes_to_f32, f32_to_embedding_bytes, generate_point_id,
};

//! Reflex library crate (used by the server and integration tests).
//!
//! # Public API Surface
//!
//! This crate exposes a large public API to support both the server binary and
//! integration tests. The exports are organized by module:
//!
//! ## Core Types (Stable)
//! - [`Config`], [`ConfigError`] - Server configuration
//! - [`CacheEntry`] - Storage format for cached responses
//! - [`TieredCache`], [`L1CacheHandle`], [`L2SemanticCache`] - Cache infrastructure
//! - [`CrossEncoderScorer`], [`VerificationResult`] - L3 verification
//!
//! ## Embedding & Scoring
//! - [`SinterEmbedder`], [`SinterConfig`] - Embedding generation
//! - [`Reranker`], [`RerankerConfig`] - Cross-encoder reranking
//! - [`VectorRescorer`] - Full-precision rescoring
//!
//! ## Vector Database
//! - [`BqClient`], [`BqConfig`] - Binary quantized Qdrant client
//! - [`QdrantClient`] - Direct Qdrant access
//!
//! ## Utilities
//! - [`TauqEncoder`], [`TauqDecoder`] - Response encoding
//! - [`DimConfig`], [`validate_embedding_dim`] - Dimension validation
//! - Hashing functions for cache keys and tenant IDs
//!
//! ## Constants
//! Many dimension and configuration constants are exported for consistency
//! across modules. Prefer using [`DimConfig`] for runtime configuration.
//!
//! ## Test/Mock Support
//! Mock implementations are available behind `#[cfg(any(test, feature = "mock"))]`.

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

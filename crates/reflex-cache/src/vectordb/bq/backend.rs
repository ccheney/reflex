use crate::cache::BqSearchBackend;
use crate::vectordb::{SearchResult, VectorDbError, VectorPoint, WriteConsistency};

use super::client::BqClient;
use super::config::BqConfig;

#[cfg(any(test, feature = "mock"))]
use super::mock::MockBqClient;

#[derive(Clone)]
/// Binary-quantized backend wrapper (real or mock).
pub enum BqBackend {
    /// Real Qdrant-backed client.
    Real(BqClient),
    #[cfg(any(test, feature = "mock"))]
    /// In-memory mock backend.
    Mock(MockBqClient),
}

impl BqBackend {
    /// Builds a backend from a URL and config (`mock:` URLs require `mock` feature).
    pub async fn from_config(url: &str, config: BqConfig) -> Result<Self, VectorDbError> {
        if url.starts_with("mock:") {
            #[cfg(any(test, feature = "mock"))]
            {
                Ok(Self::Mock(MockBqClient::with_config(config)))
            }
            #[cfg(not(any(test, feature = "mock")))]
            {
                Err(VectorDbError::ConnectionFailed {
                    url: url.to_string(),
                    message: "Mock backend not enabled. Compile with --features mock".to_string(),
                })
            }
        } else {
            let client = BqClient::with_config(url, config).await?;
            Ok(Self::Real(client))
        }
    }
}

impl BqSearchBackend for BqBackend {
    async fn is_ready(&self) -> bool {
        match self {
            BqBackend::Real(c) => c.health_check().await.is_ok(),
            #[cfg(any(test, feature = "mock"))]
            BqBackend::Mock(_) => true,
        }
    }

    async fn ensure_collection(&self, name: &str, vector_size: u64) -> Result<(), VectorDbError> {
        match self {
            BqBackend::Real(c) => c.ensure_bq_collection(name, vector_size).await,
            #[cfg(any(test, feature = "mock"))]
            BqBackend::Mock(c) => c.ensure_bq_collection(name, vector_size).await,
        }
    }

    async fn search_bq(
        &self,
        collection: &str,
        query: Vec<f32>,
        limit: u64,
        tenant_filter: Option<u64>,
    ) -> Result<Vec<SearchResult>, VectorDbError> {
        match self {
            BqBackend::Real(c) => c.search_bq(collection, query, limit, tenant_filter).await,
            #[cfg(any(test, feature = "mock"))]
            BqBackend::Mock(c) => c.search_bq(collection, query, limit, tenant_filter).await,
        }
    }

    async fn upsert_points(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
        consistency: WriteConsistency,
    ) -> Result<(), VectorDbError> {
        match self {
            BqBackend::Real(c) => c.upsert_points(collection, points, consistency).await,
            #[cfg(any(test, feature = "mock"))]
            BqBackend::Mock(c) => c.upsert_points(collection, points, consistency).await,
        }
    }
}

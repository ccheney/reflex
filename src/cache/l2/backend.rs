use crate::vectordb::bq::BqClient;
#[cfg(any(test, feature = "mock"))]
use crate::vectordb::bq::MockBqClient;
use crate::vectordb::{SearchResult, VectorDbError, VectorPoint, WriteConsistency};

/// Backend required by the L2 cache for vector search + upsert.
pub trait BqSearchBackend: Send + Sync {
    /// Returns `true` if the backend is ready for requests.
    fn is_ready(&self) -> impl std::future::Future<Output = bool> + Send;

    /// Creates the collection if it doesn't exist.
    fn ensure_collection(
        &self,
        name: &str,
        vector_size: u64,
    ) -> impl std::future::Future<Output = Result<(), VectorDbError>> + Send;

    /// Performs a search against the binary-quantized index.
    fn search_bq(
        &self,
        collection: &str,
        query: Vec<f32>,
        limit: u64,
        tenant_filter: Option<u64>,
    ) -> impl std::future::Future<Output = Result<Vec<SearchResult>, VectorDbError>> + Send;

    /// Upserts points into the collection.
    fn upsert_points(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
        consistency: WriteConsistency,
    ) -> impl std::future::Future<Output = Result<(), VectorDbError>> + Send;
}

impl BqSearchBackend for BqClient {
    async fn is_ready(&self) -> bool {
        self.health_check().await.is_ok()
    }

    async fn ensure_collection(&self, name: &str, vector_size: u64) -> Result<(), VectorDbError> {
        self.ensure_bq_collection(name, vector_size).await
    }

    async fn search_bq(
        &self,
        collection: &str,
        query: Vec<f32>,
        limit: u64,
        tenant_filter: Option<u64>,
    ) -> Result<Vec<SearchResult>, VectorDbError> {
        self.search_bq(collection, query, limit, tenant_filter)
            .await
    }

    async fn upsert_points(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
        consistency: WriteConsistency,
    ) -> Result<(), VectorDbError> {
        self.upsert_points(collection, points, consistency).await
    }
}

#[cfg(any(test, feature = "mock"))]
impl BqSearchBackend for MockBqClient {
    async fn is_ready(&self) -> bool {
        true
    }

    async fn ensure_collection(&self, name: &str, vector_size: u64) -> Result<(), VectorDbError> {
        self.ensure_bq_collection(name, vector_size).await
    }

    async fn search_bq(
        &self,
        collection: &str,
        query: Vec<f32>,
        limit: u64,
        tenant_filter: Option<u64>,
    ) -> Result<Vec<SearchResult>, VectorDbError> {
        self.search_bq(collection, query, limit, tenant_filter)
            .await
    }

    async fn upsert_points(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
        consistency: WriteConsistency,
    ) -> Result<(), VectorDbError> {
        self.upsert_points(collection, points, consistency).await
    }
}

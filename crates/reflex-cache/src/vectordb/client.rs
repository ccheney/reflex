use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, Distance, Filter, PointStruct, SearchPointsBuilder,
    UpsertPointsBuilder, VectorParamsBuilder,
};
use std::collections::HashMap;

use super::error::VectorDbError;
use super::model::{SearchResult, VectorPoint};
use crate::vectordb::WriteConsistency;

#[derive(Clone)]
/// Direct Qdrant client wrapper.
pub struct QdrantClient {
    client: Qdrant,
    url: String,
}

impl QdrantClient {
    /// Creates a client for `url`.
    pub async fn new(url: &str) -> Result<Self, VectorDbError> {
        let client =
            Qdrant::from_url(url)
                .build()
                .map_err(|e| VectorDbError::ConnectionFailed {
                    url: url.to_string(),
                    message: e.to_string(),
                })?;

        Ok(Self {
            client,
            url: url.to_string(),
        })
    }

    /// Returns the underlying Qdrant client.
    pub fn client(&self) -> &Qdrant {
        &self.client
    }

    /// Returns the configured URL.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Performs a basic health check request.
    pub async fn health_check(&self) -> Result<(), VectorDbError> {
        self.client
            .health_check()
            .await
            .map_err(|e| VectorDbError::ConnectionFailed {
                url: self.url.clone(),
                message: e.to_string(),
            })?;
        Ok(())
    }

    /// Creates a collection with cosine distance.
    pub async fn create_collection(
        &self,
        name: &str,
        vector_size: u64,
    ) -> Result<(), VectorDbError> {
        let vectors_config = VectorParamsBuilder::new(vector_size, Distance::Cosine);

        self.client
            .create_collection(
                CreateCollectionBuilder::new(name)
                    .vectors_config(vectors_config)
                    .on_disk_payload(true),
            )
            .await
            .map_err(|e| VectorDbError::CreateCollectionFailed {
                collection: name.to_string(),
                message: e.to_string(),
            })?;

        Ok(())
    }

    /// Ensures a collection exists (creates it if missing).
    pub async fn ensure_collection(
        &self,
        name: &str,
        vector_size: u64,
    ) -> Result<(), VectorDbError> {
        let exists = self.client.collection_exists(name).await.map_err(|e| {
            VectorDbError::CreateCollectionFailed {
                collection: name.to_string(),
                message: e.to_string(),
            }
        })?;

        if !exists {
            self.create_collection(name, vector_size).await?;
        }

        Ok(())
    }

    /// Returns `true` if the collection exists.
    pub async fn collection_exists(&self, name: &str) -> Result<bool, VectorDbError> {
        self.client.collection_exists(name).await.map_err(|e| {
            VectorDbError::CreateCollectionFailed {
                collection: name.to_string(),
                message: e.to_string(),
            }
        })
    }

    /// Upserts points into a collection.
    pub async fn upsert_points(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
        consistency: WriteConsistency,
    ) -> Result<(), VectorDbError> {
        if points.is_empty() {
            return Ok(());
        }

        let qdrant_points: Vec<PointStruct> = points
            .into_iter()
            .map(|p| {
                let mut payload: HashMap<String, qdrant_client::qdrant::Value> = HashMap::new();
                payload.insert("tenant_id".to_string(), (p.tenant_id as i64).into());
                payload.insert("context_hash".to_string(), (p.context_hash as i64).into());
                payload.insert("timestamp".to_string(), p.timestamp.into());
                if let Some(key) = p.storage_key {
                    payload.insert("storage_key".to_string(), key.into());
                }

                PointStruct::new(p.id, p.vector, payload)
            })
            .collect();

        self.client
            .upsert_points(
                UpsertPointsBuilder::new(collection, qdrant_points).wait(consistency.into()),
            )
            .await
            .map_err(|e| VectorDbError::UpsertFailed {
                collection: collection.to_string(),
                message: e.to_string(),
            })?;

        Ok(())
    }

    /// Searches a collection by vector similarity.
    pub async fn search(
        &self,
        collection: &str,
        query: Vec<f32>,
        limit: u64,
        tenant_filter: Option<u64>,
    ) -> Result<Vec<SearchResult>, VectorDbError> {
        let mut search_builder =
            SearchPointsBuilder::new(collection, query, limit).with_payload(true);

        if let Some(tenant_id) = tenant_filter {
            let filter = Filter::must([Condition::matches("tenant_id", tenant_id as i64)]);
            search_builder = search_builder.filter(filter);
        }

        let search_result = self
            .client
            .search_points(search_builder)
            .await
            .map_err(|e| VectorDbError::SearchFailed {
                collection: collection.to_string(),
                message: e.to_string(),
            })?;

        let results = search_result
            .result
            .into_iter()
            .filter_map(SearchResult::from_scored_point)
            .collect();

        Ok(results)
    }

    /// Deletes points by id.
    pub async fn delete_points(
        &self,
        collection: &str,
        ids: Vec<u64>,
    ) -> Result<(), VectorDbError> {
        if ids.is_empty() {
            return Ok(());
        }

        use qdrant_client::qdrant::{DeletePointsBuilder, PointsIdsList};

        let points_selector = PointsIdsList {
            ids: ids.into_iter().map(|id| id.into()).collect(),
        };

        self.client
            .delete_points(
                DeletePointsBuilder::new(collection)
                    .points(points_selector)
                    .wait(true),
            )
            .await
            .map_err(|e| VectorDbError::DeleteFailed {
                collection: collection.to_string(),
                message: e.to_string(),
            })?;

        Ok(())
    }
}

/// Minimal async interface used by higher-level code.
pub trait VectorDbClient: Send + Sync {
    /// Ensures a collection exists.
    fn ensure_collection(
        &self,
        name: &str,
        vector_size: u64,
    ) -> impl std::future::Future<Output = Result<(), VectorDbError>> + Send;

    /// Upserts points.
    fn upsert_points(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
        consistency: WriteConsistency,
    ) -> impl std::future::Future<Output = Result<(), VectorDbError>> + Send;

    /// Searches for similar points.
    fn search(
        &self,
        collection: &str,
        query: Vec<f32>,
        limit: u64,
        tenant_filter: Option<u64>,
    ) -> impl std::future::Future<Output = Result<Vec<SearchResult>, VectorDbError>> + Send;

    /// Deletes points.
    fn delete_points(
        &self,
        collection: &str,
        ids: Vec<u64>,
    ) -> impl std::future::Future<Output = Result<(), VectorDbError>> + Send;
}

impl VectorDbClient for QdrantClient {
    async fn ensure_collection(&self, name: &str, vector_size: u64) -> Result<(), VectorDbError> {
        self.ensure_collection(name, vector_size).await
    }

    async fn upsert_points(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
        consistency: WriteConsistency,
    ) -> Result<(), VectorDbError> {
        self.upsert_points(collection, points, consistency).await
    }

    async fn search(
        &self,
        collection: &str,
        query: Vec<f32>,
        limit: u64,
        tenant_filter: Option<u64>,
    ) -> Result<Vec<SearchResult>, VectorDbError> {
        self.search(collection, query, limit, tenant_filter).await
    }

    async fn delete_points(&self, collection: &str, ids: Vec<u64>) -> Result<(), VectorDbError> {
        self.delete_points(collection, ids).await
    }
}

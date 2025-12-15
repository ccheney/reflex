use qdrant_client::qdrant::{
    BinaryQuantizationBuilder, Condition, CreateCollectionBuilder, Distance, Filter,
    QuantizationSearchParamsBuilder, SearchParamsBuilder, SearchPointsBuilder, VectorParamsBuilder,
};

use super::config::BqConfig;
use crate::vectordb::{QdrantClient, SearchResult, VectorDbError, VectorPoint, WriteConsistency};

#[derive(Clone)]
pub struct BqClient {
    inner: QdrantClient,
    config: BqConfig,
}

impl BqClient {
    pub async fn new(url: &str) -> Result<Self, VectorDbError> {
        Self::with_config(url, BqConfig::default()).await
    }

    pub async fn with_config(url: &str, config: BqConfig) -> Result<Self, VectorDbError> {
        let inner = QdrantClient::new(url).await?;
        Ok(Self { inner, config })
    }

    pub fn url(&self) -> &str {
        self.inner.url()
    }

    pub fn config(&self) -> &BqConfig {
        &self.config
    }

    pub async fn create_bq_collection(
        &self,
        name: &str,
        vector_size: u64,
    ) -> Result<(), VectorDbError> {
        let vectors_config = VectorParamsBuilder::new(vector_size, Distance::Cosine);
        let quantization_config = BinaryQuantizationBuilder::new(self.config.always_ram);

        self.inner
            .client()
            .create_collection(
                CreateCollectionBuilder::new(name)
                    .vectors_config(vectors_config)
                    .quantization_config(quantization_config)
                    .on_disk_payload(self.config.on_disk_payload),
            )
            .await
            .map_err(|e| VectorDbError::CreateCollectionFailed {
                collection: name.to_string(),
                message: e.to_string(),
            })?;

        Ok(())
    }

    pub async fn ensure_bq_collection(
        &self,
        name: &str,
        vector_size: u64,
    ) -> Result<(), VectorDbError> {
        let exists = self.inner.collection_exists(name).await?;

        if !exists {
            self.create_bq_collection(name, vector_size).await?;
        }

        Ok(())
    }

    pub async fn collection_exists(&self, name: &str) -> Result<bool, VectorDbError> {
        self.inner.collection_exists(name).await
    }

    pub async fn health_check(&self) -> Result<(), VectorDbError> {
        self.inner.health_check().await
    }

    pub async fn search_bq(
        &self,
        collection: &str,
        query: Vec<f32>,
        limit: u64,
        tenant_filter: Option<u64>,
    ) -> Result<Vec<SearchResult>, VectorDbError> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let quantization_params = QuantizationSearchParamsBuilder::default()
            .rescore(self.config.rescore)
            .oversampling(self.config.rescore_limit as f64 / limit as f64);

        let search_params = SearchParamsBuilder::default().quantization(quantization_params);

        let mut search_builder = SearchPointsBuilder::new(collection, query, limit)
            .with_payload(true)
            .params(search_params);

        if let Some(tenant_id) = tenant_filter {
            let filter = Filter::must([Condition::matches("tenant_id", tenant_id as i64)]);
            search_builder = search_builder.filter(filter);
        }

        let search_result = self
            .inner
            .client()
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

    pub async fn upsert_points(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
        consistency: WriteConsistency,
    ) -> Result<(), VectorDbError> {
        self.inner
            .upsert_points(collection, points, consistency)
            .await
    }

    pub async fn delete_points(
        &self,
        collection: &str,
        ids: Vec<u64>,
    ) -> Result<(), VectorDbError> {
        self.inner.delete_points(collection, ids).await
    }
}

use genai::Client;
use std::path::PathBuf;
use std::sync::Arc;

use reflex::cache::{BqSearchBackend, StorageLoader, TieredCache};
use reflex::scoring::CrossEncoderScorer;

#[derive(Clone)]
pub struct HandlerState<
    B: BqSearchBackend + Clone + Send + Sync + 'static,
    S: StorageLoader + Clone + Send + Sync + 'static,
> {
    pub tiered_cache: Arc<TieredCache<B, S>>,

    pub scorer: Arc<CrossEncoderScorer>,

    pub storage_path: PathBuf,

    pub bq_client: B,

    pub collection_name: String,

    pub genai_client: Client,

    pub mock_provider: bool,
}

impl<B, S> HandlerState<B, S>
where
    B: BqSearchBackend + Clone + Send + Sync + 'static,
    S: StorageLoader + Clone + Send + Sync + 'static,
{
    pub fn new(
        tiered_cache: Arc<TieredCache<B, S>>,
        scorer: Arc<CrossEncoderScorer>,
        storage_path: PathBuf,
        bq_client: B,
        collection_name: String,
    ) -> Self {
        let mock_provider = std::env::var_os("REFLEX_MOCK_PROVIDER").is_some_and(|v| !v.is_empty());
        Self {
            tiered_cache,
            scorer,
            storage_path,
            bq_client,
            collection_name,
            genai_client: Client::default(),
            mock_provider,
        }
    }

    pub fn new_with_mock_provider(
        tiered_cache: Arc<TieredCache<B, S>>,
        scorer: Arc<CrossEncoderScorer>,
        storage_path: PathBuf,
        bq_client: B,
        collection_name: String,
        mock_provider: bool,
    ) -> Self {
        Self {
            tiered_cache,
            scorer,
            storage_path,
            bq_client,
            collection_name,
            genai_client: Client::default(),
            mock_provider,
        }
    }
}

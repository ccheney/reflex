//! Test server harness.

use reflex::cache::TieredCache;
use reflex::cache::{BqSearchBackend, L1CacheHandle, L2Config, L2SemanticCache, NvmeStorageLoader};
use reflex::embedding::RerankerConfig;
use reflex::embedding::sinter::{SinterConfig, SinterEmbedder};
use reflex::scoring::CrossEncoderScorer;
use reflex::vectordb::bq::{BqClient, MockBqClient};
use reflex_server::gateway::{HandlerState, create_router_with_state};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

const STARTUP_WAIT_TIMEOUT_SECS: u64 = 5;
const STARTUP_POLL_INTERVAL_MS: u64 = 50;
const TEST_COLLECTION_NAME: &str = "reflex_test_bq";

#[derive(Debug, Clone)]
pub struct TestServerConfig {
    pub port: u16,
    pub collection_name: Option<String>,
    pub storage_path: Option<std::path::PathBuf>,
    pub reranker_threshold: f32,
}

impl Default for TestServerConfig {
    fn default() -> Self {
        Self {
            port: 0,
            collection_name: None,
            storage_path: None,
            reranker_threshold: 0.70,
        }
    }
}

impl TestServerConfig {}

pub struct TestServer {
    pub addr: SocketAddr,
    _server_handle: JoinHandle<()>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    _temp_dir: Option<TempDir>,
}

impl TestServer {
    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

pub async fn find_available_port() -> std::io::Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    Ok(addr.port())
}

pub async fn wait_for_server_ready(
    addr: SocketAddr,
    timeout: Duration,
    interval: Duration,
) -> Result<(), ServerStartupError> {
    let start = std::time::Instant::now();

    loop {
        if start.elapsed() > timeout {
            return Err(ServerStartupError::Timeout);
        }

        match tokio::net::TcpStream::connect(addr).await {
            Ok(_) => return Ok(()),
            Err(_) => {
                tokio::time::sleep(interval).await;
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ServerStartupError {
    #[error("Server failed to start within timeout")]
    Timeout,
    #[error("Failed to bind to address: {0}")]
    BindError(#[from] std::io::Error),
    #[error("Server startup failed: {0}")]
    StartupFailed(String),
}

/// Spawns a fully-mocked test server for unit and integration tests.
///
/// # What "test" means
///
/// This function creates a server with **all external dependencies mocked**:
/// - **Vector database**: Uses `MockBqClient` (in-memory, no real Qdrant required)
/// - **LLM provider**: Uses mock provider (`mock_provider=true`)
/// - **Embedder**: Uses stub embedder (deterministic, fast)
/// - **Reranker**: Uses stub reranker (deterministic scores)
///
/// # Use cases
///
/// - Unit tests that need a running server but not real infrastructure
/// - Fast CI tests without external service dependencies
/// - Testing HTTP routing, request/response handling, and business logic
///
/// # Comparison with `spawn_real_server`
///
/// | Component       | `spawn_test_server` | `spawn_real_server`          |
/// |-----------------|---------------------|------------------------------|
/// | Qdrant          | Mock (in-memory)    | Real (requires running instance) |
/// | LLM Provider    | Mock                | Mock                         |
/// | Embedder        | Stub                | Real (if `REFLEX_MODEL_PATH` set) |
/// | Reranker        | Stub                | Real (if `REFLEX_RERANKER_PATH` set) |
///
/// # Example
///
/// ```ignore
/// let server = spawn_test_server(TestServerConfig::default()).await?;
/// let client = reqwest::Client::new();
/// let resp = client.get(format!("{}/health", server.url())).send().await?;
/// assert!(resp.status().is_success());
/// ```
pub async fn spawn_test_server(config: TestServerConfig) -> Result<TestServer, ServerStartupError> {
    let port = if config.port == 0 {
        find_available_port().await?
    } else {
        config.port
    };

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = TcpListener::bind(addr).await?;
    let local_addr = listener.local_addr()?;

    let (storage_path, _temp_dir) = if let Some(path) = config.storage_path {
        (path, None)
    } else {
        let temp_dir =
            TempDir::new().map_err(|e| ServerStartupError::StartupFailed(e.to_string()))?;
        (temp_dir.path().to_path_buf(), Some(temp_dir))
    };

    let collection_name = config
        .collection_name
        .unwrap_or_else(|| TEST_COLLECTION_NAME.to_string());

    let bq_client = MockBqClient::new();
    bq_client
        .ensure_bq_collection(&collection_name, reflex::constants::DEFAULT_VECTOR_SIZE_U64)
        .await
        .map_err(|e| ServerStartupError::StartupFailed(e.to_string()))?;

    let loader = NvmeStorageLoader::new(storage_path.clone());

    let embedder = SinterEmbedder::load(SinterConfig::stub())
        .map_err(|e| ServerStartupError::StartupFailed(e.to_string()))?;

    let l2_config = L2Config::default().collection_name(&collection_name);
    let l2_cache = L2SemanticCache::new(embedder, bq_client.clone(), loader, l2_config)
        .map_err(|e| ServerStartupError::StartupFailed(e.to_string()))?;

    let l1_cache = L1CacheHandle::new();
    let tiered_cache = TieredCache::new(l1_cache, l2_cache);
    let tiered_cache = Arc::new(tiered_cache);

    let scorer =
        CrossEncoderScorer::new(RerankerConfig::stub().with_threshold(config.reranker_threshold))
            .map_err(|e| ServerStartupError::StartupFailed(e.to_string()))?;
    let scorer = Arc::new(scorer);

    let state = HandlerState::new_with_mock_provider(
        tiered_cache,
        scorer,
        storage_path,
        bq_client,
        collection_name,
        true,
    );

    let app = create_router_with_state(state);

    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server_handle = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await
            .unwrap();
    });

    wait_for_server_ready(
        local_addr,
        Duration::from_secs(STARTUP_WAIT_TIMEOUT_SECS),
        Duration::from_millis(STARTUP_POLL_INTERVAL_MS),
    )
    .await?;

    Ok(TestServer {
        addr: local_addr,
        _server_handle: server_handle,
        shutdown_tx: Some(shutdown_tx),
        _temp_dir,
    })
}

/// Spawns a test server with a **real Qdrant** vector database connection.
///
/// # What "real" means (and what it does NOT mean)
///
/// The "real" in this function name refers specifically to the **vector database**:
/// - **Vector database**: Uses real `BqClient` connected to an actual Qdrant instance
/// - **LLM provider**: Still uses mock provider (`mock_provider=true`) -- NOT a real LLM
///
/// This naming convention exists because Qdrant integration is the primary differentiator
/// for end-to-end vector search testing, while LLM calls are typically mocked to avoid
/// costs and rate limits during testing.
///
/// # Component configuration
///
/// | Component       | Default                          | With env var                    |
/// |-----------------|----------------------------------|----------------------------------|
/// | Qdrant          | Real (`localhost:6334`)          | `REFLEX_QDRANT_URL`             |
/// | LLM Provider    | Mock (always)                    | N/A                             |
/// | Embedder        | Stub                             | Real if `REFLEX_MODEL_PATH`     |
/// | Reranker        | Stub                             | Real if `REFLEX_RERANKER_PATH`  |
///
/// # Prerequisites
///
/// Requires a running Qdrant instance. Start one with:
/// ```bash
/// docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
/// ```
///
/// # Use cases
///
/// - Integration tests validating vector storage and retrieval
/// - Testing semantic search with real embeddings
/// - Verifying Qdrant collection management
///
/// # Comparison with `spawn_test_server`
///
/// Use `spawn_test_server` when you do NOT need real Qdrant (faster, no dependencies).
/// Use `spawn_real_server` when you need to validate actual vector database operations.
///
/// # Example
///
/// ```ignore
/// // Ensure Qdrant is running at localhost:6334 (or set REFLEX_QDRANT_URL)
/// let server = spawn_real_server(TestServerConfig::default()).await?;
/// // The server now uses real Qdrant for vector operations
/// ```
pub async fn spawn_real_server(config: TestServerConfig) -> Result<TestServer, ServerStartupError> {
    let port = if config.port == 0 {
        find_available_port().await?
    } else {
        config.port
    };

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = TcpListener::bind(addr).await?;
    let local_addr = listener.local_addr()?;

    let (storage_path, _temp_dir) = if let Some(path) = config.storage_path {
        (path, None)
    } else {
        let temp_dir =
            TempDir::new().map_err(|e| ServerStartupError::StartupFailed(e.to_string()))?;
        (temp_dir.path().to_path_buf(), Some(temp_dir))
    };

    let collection_name = config
        .collection_name
        .unwrap_or_else(|| format!("{}_{}", TEST_COLLECTION_NAME, uuid::Uuid::new_v4().simple()));

    // Connect to real Qdrant instance (default: localhost:6334)
    let qdrant_url =
        std::env::var("REFLEX_QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_string());
    let bq_client = BqClient::new(&qdrant_url).await.map_err(|e| {
        ServerStartupError::StartupFailed(format!("Failed to connect to Qdrant: {}", e))
    })?;

    bq_client
        .ensure_collection(&collection_name, reflex::constants::DEFAULT_VECTOR_SIZE_U64)
        .await
        .map_err(|e| {
            ServerStartupError::StartupFailed(format!("Failed to ensure collection: {}", e))
        })?;

    let loader = NvmeStorageLoader::new(storage_path.clone());

    let embedder = if let Ok(path) = std::env::var("REFLEX_MODEL_PATH") {
        println!("Using Real Embedder: {}", path);
        SinterEmbedder::load(SinterConfig::new(path))
            .map_err(|e| ServerStartupError::StartupFailed(e.to_string()))?
    } else {
        println!("Using Stub Embedder");
        SinterEmbedder::load(SinterConfig::stub())
            .map_err(|e| ServerStartupError::StartupFailed(e.to_string()))?
    };

    let l2_config = L2Config::default().collection_name(&collection_name);
    let l2_cache = L2SemanticCache::new(embedder, bq_client.clone(), loader, l2_config)
        .map_err(|e| ServerStartupError::StartupFailed(e.to_string()))?;

    let l1_cache = L1CacheHandle::new();
    let tiered_cache = TieredCache::new(l1_cache, l2_cache);
    let tiered_cache = Arc::new(tiered_cache);

    let scorer = if let Ok(path) = std::env::var("REFLEX_RERANKER_PATH") {
        println!("Using Real Reranker: {}", path);
        let config = RerankerConfig::new(path).with_threshold(config.reranker_threshold);
        CrossEncoderScorer::new(config)
            .map_err(|e| ServerStartupError::StartupFailed(e.to_string()))?
    } else {
        println!("Using Stub Reranker");
        CrossEncoderScorer::new(RerankerConfig::stub().with_threshold(config.reranker_threshold))
            .map_err(|e| ServerStartupError::StartupFailed(e.to_string()))?
    };
    let scorer = Arc::new(scorer);

    // NOTE: Even though this is spawn_real_server, we still use mock_provider=true
    // for the LLM. "Real" refers to real Qdrant, not real LLM provider.
    // This avoids API costs and rate limits during integration tests.
    let state = HandlerState::new_with_mock_provider(
        tiered_cache,
        scorer,
        storage_path,
        bq_client,
        collection_name,
        true, // mock_provider: true = mock LLM, false = real LLM
    );

    let app = create_router_with_state(state);

    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server_handle = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await
            .unwrap();
    });

    wait_for_server_ready(
        local_addr,
        Duration::from_secs(STARTUP_WAIT_TIMEOUT_SECS),
        Duration::from_millis(STARTUP_POLL_INTERVAL_MS),
    )
    .await?;

    Ok(TestServer {
        addr: local_addr,
        _server_handle: server_handle,
        shutdown_tx: Some(shutdown_tx),
        _temp_dir,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_find_available_port() {
        let port = find_available_port()
            .await
            .expect("Should find available port");
        assert!(port > 0);
    }

    #[tokio::test]
    async fn test_server_config_defaults() {
        let config = TestServerConfig::default();
        assert_eq!(config.port, 0);
    }

    #[tokio::test]
    async fn test_server_helpers_are_callable() {
        let (shutdown_tx, _shutdown_rx) = oneshot::channel();
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();

        let server = TestServer {
            addr,
            _server_handle: tokio::spawn(async {}),
            shutdown_tx: Some(shutdown_tx),
            _temp_dir: None,
        };

        let _ = server.url();
        server.shutdown().await;
    }

    #[test]
    fn test_spawners_are_referenced() {
        std::mem::drop(spawn_test_server(TestServerConfig::default()));
        std::mem::drop(spawn_real_server(TestServerConfig::default()));
    }

    #[test]
    fn test_server_url_formatting() {
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let url = format!("http://{}", addr);
        assert_eq!(url, "http://127.0.0.1:8080");
    }
}

//! Reflex HTTP server entrypoint.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use mimalloc::MiMalloc;
use tokio::net::TcpListener;
use tokio::signal;

use reflex::cache::{L2Config, L2SemanticCache, NvmeStorageLoader, TieredCache};
use reflex::config::Config;
use reflex::embedding::{RerankerConfig, SinterConfig, SinterEmbedder};
use reflex::lifecycle::{LifecycleConfig, LifecycleManager, build_cloud_ops};
use reflex::scoring::CrossEncoderScorer;
use reflex::vectordb::bq::{BQ_COLLECTION_NAME, BqBackend, BqConfig};
use reflex_server::gateway::{HandlerState, create_router_with_state};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!(
        r#"
██████╗ ███████╗███████╗██╗     ███████╗██╗  ██╗
██╔══██╗██╔════╝██╔════╝██║     ██╔════╝╚██╗██╔╝
██████╔╝█████╗  █████╗  ██║     █████╗   ╚███╔╝ 
██╔══██╗██╔══╝  ██╔══╝  ██║     ██╔══╝   ██╔██╗ 
██║  ██║███████╗██║     ███████╗███████╗██╔╝ ██╗
╚═╝  ╚═╝╚══════╝╚═╝     ╚══════╝╚══════╝╚═╝  ╚═╝

        STOP. ROUTE. DODGE.
                                        AGPL-3.0
"#
    );

    if std::env::args().any(|arg| arg == "--health-check") {
        std::process::exit(run_health_check());
    }

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let config = Config::from_env()?;
    config.validate()?;
    let addr: SocketAddr = config.socket_addr().parse()?;

    tracing::info!(
        bind_addr = %config.bind_addr,
        port = config.port,
        "Reflex starting (Ingress Architecture)"
    );

    let lifecycle_config = LifecycleConfig::from_env()?;
    let cloud_ops = build_cloud_ops(&lifecycle_config).await;
    let lifecycle = Arc::new(LifecycleManager::new_with_ops(lifecycle_config, cloud_ops));

    tracing::info!("Hydrating state from cloud storage...");
    if let Err(e) = lifecycle.hydrate().await {
        tracing::warn!("Failed to hydrate state: {}. Starting empty.", e);
    } else {
        tracing::info!("Hydration complete.");
    }

    lifecycle.start_reaper_thread();

    let storage_loader = NvmeStorageLoader::new(config.storage_path.clone());

    let bq_config = BqConfig::default();
    let bq_client = BqBackend::from_config(&config.qdrant_url, bq_config.clone()).await?;

    let sinter_config = if let Some(path) = &config.model_path {
        SinterConfig::new(path.clone())
    } else {
        tracing::warn!("No REFLEX_MODEL_PATH configured, running embedder in stub mode");
        SinterConfig::stub()
    };
    let embedder = SinterEmbedder::load(sinter_config)?;

    let l2_config = L2Config::default()
        .collection_name(BQ_COLLECTION_NAME)
        .bq_config(bq_config)
        .vector_size(embedder.embedding_dim() as u64);

    let l2_cache = L2SemanticCache::new(
        embedder,
        bq_client.clone(),
        storage_loader.clone(),
        l2_config,
    )?;

    let l1_handle = reflex::cache::L1CacheHandle::with_capacity(config.l1_capacity as usize);
    let tiered_cache = Arc::new(TieredCache::new(l1_handle, l2_cache));

    tiered_cache.l2().ensure_collection().await?;

    let reranker_config = RerankerConfig::from_env();
    let scorer = Arc::new(CrossEncoderScorer::new(reranker_config)?);

    let state = HandlerState::new(
        tiered_cache,
        scorer,
        config.storage_path.clone(),
        bq_client,
        BQ_COLLECTION_NAME.to_string(),
    );

    let app = create_router_with_state(state);

    let listener = TcpListener::bind(addr).await?;
    tracing::info!(addr = %addr, "Server listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal_with_lifecycle(lifecycle))
        .await?;

    tracing::info!("Reflex shutdown complete");
    Ok(())
}

fn run_health_check() -> i32 {
    let port = std::env::var("REFLEX_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(8080);

    let url = format!("http://127.0.0.1:{}/healthz", port);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("failed to build runtime");

    rt.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(1))
            .build()
            .expect("failed to build client");

        match client.get(&url).send().await {
            Ok(res) if res.status().is_success() => 0,
            _ => 1,
        }
    })
}

async fn shutdown_signal_with_lifecycle(lifecycle: Arc<LifecycleManager>) {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Received Ctrl+C, initiating graceful shutdown");
        }
        _ = terminate => {
            tracing::info!("Received SIGTERM, initiating graceful shutdown");
        }
    }

    tracing::info!("Dehydrating state to cloud storage...");
    if let Err(e) = lifecycle.shutdown().await {
        tracing::error!("Failed to dehydrate state: {}", e);
    } else {
        tracing::info!("Dehydration complete.");
    }
}

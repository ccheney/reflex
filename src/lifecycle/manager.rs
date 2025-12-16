use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use tokio::time;

use super::cloud::{CloudOps, GcpCloudOps, LocalCloudOps};
use super::config::{
    CloudProviderType, DEFAULT_SNAPSHOT_FILENAME, LifecycleConfig, REAPER_CHECK_INTERVAL_SECS,
};
use super::error::{LifecycleError, LifecycleResult};
use super::types::{DehydrationResult, HydrationResult};

/// Manages hydrate/dehydrate and an idle "reaper" that can stop the instance.
pub struct LifecycleManager {
    config: LifecycleConfig,
    last_activity: Arc<RwLock<Instant>>,
    shutdown_initiated: Arc<AtomicBool>,
    reaper_running: Arc<AtomicBool>,
    ops: Arc<dyn CloudOps>,
}

impl LifecycleManager {
    /// Creates a manager using the configured cloud provider implementation.
    pub fn new(config: LifecycleConfig) -> Self {
        let ops: Arc<dyn CloudOps> = match config.cloud_provider {
            CloudProviderType::Gcp => Arc::new(GcpCloudOps::new()),
            CloudProviderType::Local => Arc::new(LocalCloudOps::new()),
        };
        Self {
            config,
            last_activity: Arc::new(RwLock::new(Instant::now())),
            shutdown_initiated: Arc::new(AtomicBool::new(false)),
            reaper_running: Arc::new(AtomicBool::new(false)),
            ops,
        }
    }

    /// Creates a manager with an explicit [`CloudOps`] implementation.
    pub fn new_with_ops(config: LifecycleConfig, ops: Arc<dyn CloudOps>) -> Self {
        Self {
            config,
            last_activity: Arc::new(RwLock::new(Instant::now())),
            shutdown_initiated: Arc::new(AtomicBool::new(false)),
            reaper_running: Arc::new(AtomicBool::new(false)),
            ops,
        }
    }

    /// Creates a manager from environment configuration.
    pub fn from_env() -> LifecycleResult<Self> {
        Ok(Self::new(LifecycleConfig::from_env()?))
    }

    /// Returns the active config.
    pub fn config(&self) -> &LifecycleConfig {
        &self.config
    }

    /// Records activity (resets the idle timer).
    pub async fn record_activity(&self) {
        *self.last_activity.write().await = Instant::now();
    }

    /// Returns the time since last activity.
    pub async fn idle_duration(&self) -> Duration {
        self.last_activity.read().await.elapsed()
    }

    /// Returns `true` if `idle_timeout` is exceeded.
    pub async fn is_idle_timeout_exceeded(&self) -> bool {
        self.idle_duration().await >= self.config.idle_timeout
    }

    /// Returns `true` if shutdown has been initiated.
    pub fn is_shutdown_initiated(&self) -> bool {
        // Acquire: ensures we see all writes that happened before the Release store
        // that set this flag to true (e.g., the dehydration in shutdown())
        self.shutdown_initiated.load(Ordering::Acquire)
    }

    /// Downloads a snapshot from cloud storage (if configured).
    pub async fn hydrate(&self) -> LifecycleResult<HydrationResult> {
        if !self.config.has_gcs_bucket() {
            return Ok(HydrationResult::Skipped {
                reason: "GCS bucket not configured".to_string(),
            });
        }

        let bucket = &self.config.gcs_bucket;
        let object = DEFAULT_SNAPSHOT_FILENAME;
        let local_path = &self.config.local_snapshot_path;

        if let Some(parent) = local_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        match self.ops.download_file(bucket, object, local_path).await {
            Ok(_) => {
                let meta = tokio::fs::metadata(local_path).await?;
                Ok(HydrationResult::Success { bytes: meta.len() })
            }
            Err(LifecycleError::CloudError(msg))
                if msg.contains("No URLs matched") || msg.contains("not found") =>
            {
                Ok(HydrationResult::NotFound)
            }
            Err(e) => Err(e),
        }
    }

    /// Uploads the local snapshot to cloud storage (if configured and present).
    pub async fn dehydrate(&self) -> LifecycleResult<DehydrationResult> {
        if !self.config.has_gcs_bucket() {
            return Ok(DehydrationResult::Skipped {
                reason: "GCS bucket not configured".to_string(),
            });
        }

        let bucket = &self.config.gcs_bucket;
        let object = DEFAULT_SNAPSHOT_FILENAME;
        let local_path = &self.config.local_snapshot_path;

        if !local_path.exists() {
            return Ok(DehydrationResult::NoSnapshot);
        }

        self.ops.upload_file(bucket, object, local_path).await?;
        let bytes = tokio::fs::metadata(local_path).await?.len();
        Ok(DehydrationResult::Success { bytes })
    }

    /// Starts the idle reaper background task (no-op if already running).
    pub fn start_reaper_thread(&self) -> tokio::task::JoinHandle<()> {
        // AcqRel: swap needs both load and store semantics to ensure only one
        // reaper thread starts. Acquire sees prior stores, Release publishes our store.
        if self.reaper_running.swap(true, Ordering::AcqRel) {
            return tokio::spawn(async {});
        }

        let config = self.config.clone();
        let last_activity = Arc::clone(&self.last_activity);
        let shutdown_initiated = Arc::clone(&self.shutdown_initiated);
        let reaper_running = Arc::clone(&self.reaper_running);
        let ops = Arc::clone(&self.ops);

        tokio::spawn(async move {
            let mut interval = time::interval(Duration::from_secs(REAPER_CHECK_INTERVAL_SECS));
            loop {
                interval.tick().await;
                // Acquire: synchronizes with the Release store from shutdown() to ensure
                // we see any side effects that occurred before shutdown was initiated
                if shutdown_initiated.load(Ordering::Acquire) {
                    break;
                }

                let idle = last_activity.read().await.elapsed();
                if idle >= config.idle_timeout {
                    // Release: ensures the dehydration and stop operations that follow
                    // are visible to any thread that later loads this flag with Acquire
                    shutdown_initiated.store(true, Ordering::Release);

                    if config.has_gcs_bucket() {
                        let bucket = &config.gcs_bucket;
                        let object = DEFAULT_SNAPSHOT_FILENAME;
                        let path = &config.local_snapshot_path;
                        if path.exists() {
                            let _ = ops.upload_file(bucket, object, path).await;
                        }
                    }

                    if config.enable_instance_stop {
                        let _ = ops.stop_self().await;
                    }
                    break;
                }
            }
            // Release: ensures all reaper work is visible before marking it as not running,
            // so a subsequent start_reaper_thread() with Acquire sees the completed state
            reaper_running.store(false, Ordering::Release);
        })
    }

    /// Initiates shutdown (idempotent) and runs dehydration once.
    pub async fn shutdown(&self) -> LifecycleResult<()> {
        // AcqRel: swap needs both semantics - Acquire to see if already shut down,
        // Release to publish shutdown state so other threads (reaper, callers) see it
        if self.shutdown_initiated.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        self.dehydrate().await.map(|_| ())
    }
}

#[derive(Clone)]
/// Convenience wrapper to record activity without exposing the full manager.
pub struct ActivityRecorder {
    manager: Arc<LifecycleManager>,
}

impl ActivityRecorder {
    /// Creates a new recorder for a manager.
    pub fn new(manager: Arc<LifecycleManager>) -> Self {
        Self { manager }
    }

    /// Records activity.
    pub async fn record(&self) {
        self.manager.record_activity().await;
    }
}

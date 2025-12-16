use std::env;
use std::path::PathBuf;
use std::time::Duration;

use super::error::LifecycleResult;

/// Default idle timeout before triggering dehydration/stop.
pub const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 15 * 60;
/// Default snapshot filename (object name).
pub const DEFAULT_SNAPSHOT_FILENAME: &str = "snapshot.rkyv";
/// Default reaper poll interval.
pub const REAPER_CHECK_INTERVAL_SECS: u64 = 30;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
/// Cloud provider selection for lifecycle operations.
pub enum CloudProviderType {
    #[default]
    /// Google Cloud Platform.
    Gcp,
    /// Local filesystem mock.
    Local,
}

impl std::str::FromStr for CloudProviderType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "gcp" | "google" => Ok(Self::Gcp),
            "local" => Ok(Self::Local),
            _ => Err(format!("Unknown cloud provider: {}", s)),
        }
    }
}

#[derive(Debug, Clone)]
/// Lifecycle configuration for hydration/dehydration and idle reaping.
pub struct LifecycleConfig {
    /// GCS bucket name (empty disables cloud ops).
    pub gcs_bucket: String,
    /// Local path used for the snapshot file.
    pub local_snapshot_path: PathBuf,
    /// Idle timeout before the reaper triggers dehydration/stop.
    pub idle_timeout: Duration,
    /// If true, attempt to stop the instance after dehydration.
    pub enable_instance_stop: bool,
    /// Cloud provider selection.
    pub cloud_provider: CloudProviderType,
    /// Optional cloud region hint.
    pub cloud_region: Option<String>,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            gcs_bucket: String::new(),
            local_snapshot_path: PathBuf::from("/mnt/nvme/reflex_data")
                .join(DEFAULT_SNAPSHOT_FILENAME),
            idle_timeout: Duration::from_secs(DEFAULT_IDLE_TIMEOUT_SECS),
            enable_instance_stop: true,
            cloud_provider: CloudProviderType::default(),
            cloud_region: None,
        }
    }
}

impl LifecycleConfig {
    const ENV_GCS_BUCKET: &'static str = "REFLEX_GCS_BUCKET";
    const ENV_LOCAL_SNAPSHOT_PATH: &'static str = "REFLEX_SNAPSHOT_PATH";
    const ENV_IDLE_TIMEOUT_SECS: &'static str = "REFLEX_IDLE_TIMEOUT_SECS";
    const ENV_ENABLE_INSTANCE_STOP: &'static str = "REFLEX_ENABLE_INSTANCE_STOP";
    const ENV_CLOUD_PROVIDER: &'static str = "REFLEX_CLOUD_PROVIDER";
    const ENV_CLOUD_REGION: &'static str = "REFLEX_CLOUD_REGION";

    /// Loads config from environment variables (with defaults).
    pub fn from_env() -> LifecycleResult<Self> {
        let defaults = Self::default();
        let gcs_bucket = env::var(Self::ENV_GCS_BUCKET).unwrap_or_default();
        let local_snapshot_path = env::var(Self::ENV_LOCAL_SNAPSHOT_PATH)
            .map(PathBuf::from)
            .unwrap_or(defaults.local_snapshot_path);
        let idle_timeout = env::var(Self::ENV_IDLE_TIMEOUT_SECS)
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or(defaults.idle_timeout);
        let enable_instance_stop = env::var(Self::ENV_ENABLE_INSTANCE_STOP)
            .map(|s| s != "false" && s != "0")
            .unwrap_or(defaults.enable_instance_stop);

        let cloud_provider = env::var(Self::ENV_CLOUD_PROVIDER)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(defaults.cloud_provider);

        let cloud_region = env::var(Self::ENV_CLOUD_REGION).ok();

        Ok(Self {
            gcs_bucket,
            local_snapshot_path,
            idle_timeout,
            enable_instance_stop,
            cloud_provider,
            cloud_region,
        })
    }

    /// Returns `true` if a non-empty bucket is configured.
    pub fn has_gcs_bucket(&self) -> bool {
        !self.gcs_bucket.is_empty()
    }

    #[cfg(test)]
    pub fn for_testing(gcs_bucket: &str, local_path: PathBuf) -> Self {
        Self {
            gcs_bucket: gcs_bucket.to_string(),
            local_snapshot_path: local_path,
            idle_timeout: Duration::from_secs(1),
            enable_instance_stop: false,
            cloud_provider: CloudProviderType::Local,
            cloud_region: None,
        }
    }
}

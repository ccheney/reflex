use std::sync::Arc;

use super::cloud::{CloudOps, GcpCloudOps, LocalCloudOps};
use super::config::{CloudProviderType, LifecycleConfig};

/// Builds the appropriate [`CloudOps`] implementation for the config.
pub async fn build_cloud_ops(config: &LifecycleConfig) -> Arc<dyn CloudOps> {
    match config.cloud_provider {
        CloudProviderType::Gcp => Arc::new(GcpCloudOps::new()),
        CloudProviderType::Local => Arc::new(LocalCloudOps::new()),
    }
}

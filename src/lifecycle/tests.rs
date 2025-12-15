use super::cloud::CloudOps;
use super::config::LifecycleConfig;
use super::error::{LifecycleError, LifecycleResult};
use super::manager::LifecycleManager;
use super::types::{DehydrationResult, HydrationResult};

use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::RwLock;

pub struct MockCloudOps {
    pub files: Arc<RwLock<std::collections::HashMap<String, Vec<u8>>>>,
    pub stop_called: Arc<RwLock<bool>>,
}

impl Default for MockCloudOps {
    fn default() -> Self {
        Self::new()
    }
}

impl MockCloudOps {
    pub fn new() -> Self {
        Self {
            files: Arc::new(RwLock::new(std::collections::HashMap::new())),
            stop_called: Arc::new(RwLock::new(false)),
        }
    }
}

#[async_trait]
impl CloudOps for MockCloudOps {
    async fn download_file(
        &self,
        _container: &str,
        object: &str,
        dest: &Path,
    ) -> LifecycleResult<()> {
        let files = self.files.read().await;
        if let Some(data) = files.get(object) {
            tokio::fs::write(dest, data).await?;
            Ok(())
        } else {
            Err(LifecycleError::CloudError("No URLs matched".to_string()))
        }
    }

    async fn upload_file(&self, _container: &str, object: &str, src: &Path) -> LifecycleResult<()> {
        let data = tokio::fs::read(src).await?;
        self.files.write().await.insert(object.to_string(), data);
        Ok(())
    }

    async fn stop_self(&self) -> LifecycleResult<()> {
        *self.stop_called.write().await = true;
        Ok(())
    }
}

#[tokio::test]
async fn test_manager_integration() {
    let temp = TempDir::new().unwrap();
    let local_path = temp.path().join("snap.rkyv");
    let config = LifecycleConfig::for_testing("bucket", local_path.clone());
    let ops = Arc::new(MockCloudOps::new());
    let manager = LifecycleManager::new_with_ops(config, ops.clone());

    let res = manager.dehydrate().await.unwrap();
    assert!(matches!(res, DehydrationResult::NoSnapshot));

    tokio::fs::write(&local_path, b"data").await.unwrap();

    let res = manager.dehydrate().await.unwrap();
    assert!(matches!(res, DehydrationResult::Success { bytes: 4 }));

    tokio::fs::remove_file(&local_path).await.unwrap();
    let res = manager.hydrate().await.unwrap();
    assert!(matches!(res, HydrationResult::Success { bytes: 4 }));
}

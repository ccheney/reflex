//! Cloud storage operations used by lifecycle management.
//!
//! `GcpCloudOps` shells out to `gsutil`/`gcloud`. `LocalCloudOps` is a filesystem mock.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use tokio::process::Command;

use super::error::{LifecycleError, LifecycleResult};

const METADATA_TIMEOUT: Duration = Duration::from_secs(5);
const CMD_TIMEOUT: Duration = Duration::from_secs(120);
const CMD_RETRIES: usize = 3;
const CMD_RETRY_BACKOFF: Duration = Duration::from_millis(750);

#[async_trait]
/// Cloud operations required for hydration/dehydration and self-stop.
pub trait CloudOps: Send + Sync {
    /// Downloads `object` from `container` to `dest`.
    async fn download_file(
        &self,
        container: &str,
        object: &str,
        dest: &Path,
    ) -> LifecycleResult<()>;
    /// Uploads `src` to `container/object`.
    async fn upload_file(&self, container: &str, object: &str, src: &Path) -> LifecycleResult<()>;
    /// Attempts to stop the current instance.
    async fn stop_self(&self) -> LifecycleResult<()>;
}

/// Google Cloud Storage / Compute Engine implementation.
pub struct GcpCloudOps {
    gsutil_path: PathBuf,
    gcloud_path: PathBuf,
    http: HttpClient,
}

impl GcpCloudOps {
    /// Creates a new GCP implementation using `gsutil` and `gcloud` from `PATH`.
    pub fn new() -> Self {
        Self {
            gsutil_path: PathBuf::from("gsutil"),
            gcloud_path: PathBuf::from("gcloud"),
            http: HttpClient::builder()
                .timeout(METADATA_TIMEOUT)
                .build()
                .unwrap_or_else(|_| HttpClient::new()),
        }
    }

    async fn run_command_with_retries(
        &self,
        program: &Path,
        args: Vec<String>,
        label: &str,
    ) -> LifecycleResult<()> {
        let mut attempt = 0usize;
        loop {
            attempt += 1;

            let mut cmd = Command::new(program);
            cmd.args(&args)
                .kill_on_drop(true)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());

            let child = cmd
                .spawn()
                .map_err(|e| LifecycleError::CloudError(format!("Failed to spawn {label}: {e}")))?;

            let output = match tokio::time::timeout(CMD_TIMEOUT, child.wait_with_output()).await {
                Ok(res) => res.map_err(|e| {
                    LifecycleError::CloudError(format!("Failed waiting for {label}: {e}"))
                })?,
                Err(_) => {
                    return Err(LifecycleError::CloudError(format!(
                        "{label} timed out after {:?}",
                        CMD_TIMEOUT
                    )));
                }
            };

            if output.status.success() {
                return Ok(());
            }

            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let err = LifecycleError::CloudError(format!("{label} failed: {stderr}"));

            if attempt >= CMD_RETRIES {
                return Err(err);
            }

            tokio::time::sleep(CMD_RETRY_BACKOFF).await;
        }
    }

    async fn get_instance_metadata(&self, key: &str) -> LifecycleResult<String> {
        let url = format!(
            "http://metadata.google.internal/computeMetadata/v1/instance/{}",
            key
        );
        let resp = self
            .http
            .get(&url)
            .header("Metadata-Flavor", "Google")
            .send()
            .await
            .map_err(|e| LifecycleError::CloudError(format!("Metadata request failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(LifecycleError::CloudError(format!(
                "Metadata error: {}",
                resp.status()
            )));
        }

        let text = resp
            .text()
            .await
            .map_err(|e| LifecycleError::CloudError(format!("Failed to read metadata: {}", e)))?;

        Ok(text.trim().to_string())
    }

    async fn stop_instance(&self, zone: &str, instance: &str) -> LifecycleResult<()> {
        self.run_command_with_retries(
            &self.gcloud_path,
            vec![
                "compute".to_string(),
                "instances".to_string(),
                "stop".to_string(),
                instance.to_string(),
                "--zone".to_string(),
                zone.to_string(),
                "--quiet".to_string(),
            ],
            "gcloud stop",
        )
        .await
    }
}

impl Default for GcpCloudOps {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CloudOps for GcpCloudOps {
    async fn download_file(
        &self,
        container: &str,
        object: &str,
        dest: &Path,
    ) -> LifecycleResult<()> {
        let uri = format!("gs://{}/{}", container, object);
        self.run_command_with_retries(
            &self.gsutil_path,
            vec!["cp".to_string(), uri, dest.to_string_lossy().to_string()],
            "gsutil cp (download)",
        )
        .await
    }

    async fn upload_file(&self, container: &str, object: &str, src: &Path) -> LifecycleResult<()> {
        let uri = format!("gs://{}/{}", container, object);
        self.run_command_with_retries(
            &self.gsutil_path,
            vec!["cp".to_string(), src.to_string_lossy().to_string(), uri],
            "gsutil cp (upload)",
        )
        .await
    }

    async fn stop_self(&self) -> LifecycleResult<()> {
        let zone_full = self.get_instance_metadata("zone").await?;
        let instance = self.get_instance_metadata("name").await?;
        let zone = zone_full.rsplit('/').next().unwrap_or(&zone_full);
        self.stop_instance(zone, &instance).await
    }
}

/// Local filesystem mock implementation of [`CloudOps`].
pub struct LocalCloudOps;

impl LocalCloudOps {
    /// Creates a new local implementation.
    pub fn new() -> Self {
        Self
    }
}

impl Default for LocalCloudOps {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CloudOps for LocalCloudOps {
    async fn download_file(
        &self,
        container: &str,
        object: &str,
        dest: &Path,
    ) -> LifecycleResult<()> {
        let remote_dir = std::env::temp_dir()
            .join("reflex_cloud_mock")
            .join(container);
        let src = remote_dir.join(object);

        if !src.exists() {
            return Err(LifecycleError::CloudError(format!(
                "Local object not found: {:?}",
                src
            )));
        }

        tokio::fs::copy(&src, dest).await?;
        Ok(())
    }

    async fn upload_file(&self, container: &str, object: &str, src: &Path) -> LifecycleResult<()> {
        let remote_dir = std::env::temp_dir()
            .join("reflex_cloud_mock")
            .join(container);
        tokio::fs::create_dir_all(&remote_dir).await?;
        let dest = remote_dir.join(object);

        tokio::fs::copy(src, &dest).await?;
        Ok(())
    }

    async fn stop_self(&self) -> LifecycleResult<()> {
        println!("LocalCloudOps: Stop requested (simulated).");
        Ok(())
    }
}

//! Spot-instance lifecycle helpers for GCE (hydrate/dehydrate + idle reaper).

pub mod cloud;
/// Lifecycle configuration.
pub mod config;
/// Lifecycle error types.
pub mod error;
/// Factory helpers.
pub mod factory;
/// Lifecycle manager and reaper.
pub mod manager;
/// Result types for hydrate/dehydrate.
pub mod types;

#[cfg(test)]
mod tests;

pub use cloud::CloudOps;
pub use config::{
    DEFAULT_IDLE_TIMEOUT_SECS, DEFAULT_SNAPSHOT_FILENAME, LifecycleConfig,
    REAPER_CHECK_INTERVAL_SECS,
};
pub use error::{LifecycleError, LifecycleResult};
pub use factory::build_cloud_ops;
pub use manager::{ActivityRecorder, LifecycleManager};
pub use types::{DehydrationResult, HydrationResult};

//! Spot-instance lifecycle helpers for GCE (hydrate/dehydrate + idle reaper).

pub mod cloud;
pub mod config;
pub mod error;
pub mod factory;
pub mod manager;
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

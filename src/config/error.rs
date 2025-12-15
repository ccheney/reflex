//! Configuration error types.

use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during configuration loading and validation.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// Port value is outside valid range (1-65535).
    #[error("invalid port '{value}': must be between 1 and 65535")]
    InvalidPort { value: String },

    /// Port string could not be parsed as a number.
    #[error("failed to parse port '{value}': {source}")]
    PortParseError {
        value: String,
        #[source]
        source: std::num::ParseIntError,
    },

    /// Bind address string could not be parsed.
    #[error("failed to parse bind address '{value}': {source}")]
    InvalidBindAddr {
        value: String,
        #[source]
        source: std::net::AddrParseError,
    },

    /// A required environment variable was not set.
    ///
    /// # Current Usage
    ///
    /// This variant is currently **not used in production code**. The current configuration
    /// design uses graceful defaults for all optional settings, allowing the server to start
    /// with minimal configuration.
    ///
    /// The variant exists for:
    /// - Future stricter configuration policies where certain env vars could be required
    /// - Explicit error reporting when validation logic is added
    /// - Extension by downstream consumers who may need stricter validation
    ///
    /// If you need to enforce required configuration, use this variant in your validation
    /// logic rather than adding a new error type.
    #[error("missing required environment variable: {name}")]
    MissingEnvVar { name: &'static str },

    /// Specified path does not exist on the filesystem.
    #[error("path does not exist: {path}")]
    PathNotFound { path: PathBuf },

    /// Path exists but is not a file (when a file was expected).
    #[error("path is not a file: {path}")]
    NotAFile { path: PathBuf },

    /// Path exists but is not a directory (when a directory was expected).
    #[error("path is not a directory: {path}")]
    NotADirectory { path: PathBuf },
}

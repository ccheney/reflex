//! Configuration error types.

use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during configuration loading and validation.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// Port value is outside valid range (1-65535).
    #[error("invalid port '{value}': must be between 1 and 65535")]
    InvalidPort {
        /// Original string value.
        value: String,
    },

    /// Port string could not be parsed as a number.
    #[error("failed to parse port '{value}': {source}")]
    PortParseError {
        /// Original string value.
        value: String,
        #[source]
        /// Parse error.
        source: std::num::ParseIntError,
    },

    /// Bind address string could not be parsed.
    #[error("failed to parse bind address '{value}': {source}")]
    InvalidBindAddr {
        /// Original string value.
        value: String,
        #[source]
        /// Parse error.
        source: std::net::AddrParseError,
    },

    /// A required environment variable was not set.
    /// (Reserved for stricter validation policies.)
    #[error("missing required environment variable: {name}")]
    MissingEnvVar {
        /// Environment variable name.
        name: &'static str,
    },

    /// Specified path does not exist on the filesystem.
    #[error("path does not exist: {path}")]
    PathNotFound {
        /// Path that was missing.
        path: PathBuf,
    },

    /// Path exists but is not a file (when a file was expected).
    #[error("path is not a file: {path}")]
    NotAFile {
        /// Path that was not a file.
        path: PathBuf,
    },

    /// Path exists but is not a directory (when a directory was expected).
    #[error("path is not a directory: {path}")]
    NotADirectory {
        /// Path that was not a directory.
        path: PathBuf,
    },
}

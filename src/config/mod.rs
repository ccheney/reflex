//! Environment configuration.

pub mod error;

#[cfg(test)]
mod tests;

pub use error::ConfigError;

use std::env;
use std::net::IpAddr;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,

    pub bind_addr: IpAddr,

    pub storage_path: PathBuf,

    pub model_path: Option<PathBuf>,

    pub reranker_path: Option<PathBuf>,

    pub qdrant_url: String,

    pub l1_capacity: u64,
}

pub const DEFAULT_QDRANT_URL: &str = "http://localhost:6334";

impl Default for Config {
    fn default() -> Self {
        Self {
            port: 8080,
            bind_addr: IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1)),
            storage_path: PathBuf::from("./.data"),
            model_path: None,
            reranker_path: None,
            qdrant_url: DEFAULT_QDRANT_URL.to_string(),
            l1_capacity: 10_000,
        }
    }
}

impl Config {
    const ENV_PORT: &'static str = "REFLEX_PORT";
    const ENV_BIND_ADDR: &'static str = "REFLEX_BIND_ADDR";
    const ENV_STORAGE_PATH: &'static str = "REFLEX_STORAGE_PATH";
    const ENV_MODEL_PATH: &'static str = "REFLEX_MODEL_PATH";
    const ENV_RERANKER_PATH: &'static str = "REFLEX_RERANKER_PATH";
    const ENV_QDRANT_URL: &'static str = "REFLEX_QDRANT_URL";
    const ENV_L1_CAPACITY: &'static str = "REFLEX_L1_CAPACITY";

    pub fn from_env() -> Result<Self, ConfigError> {
        let defaults = Self::default();

        let port = Self::parse_port_from_env(defaults.port)?;
        let bind_addr = Self::parse_bind_addr_from_env(defaults.bind_addr)?;
        let storage_path = Self::parse_path_from_env(Self::ENV_STORAGE_PATH, defaults.storage_path);
        let model_path = Self::parse_optional_path_from_env(Self::ENV_MODEL_PATH);
        let reranker_path = Self::parse_optional_path_from_env(Self::ENV_RERANKER_PATH);
        let qdrant_url = Self::parse_string_from_env(Self::ENV_QDRANT_URL, defaults.qdrant_url);
        let l1_capacity = Self::parse_u64_from_env(Self::ENV_L1_CAPACITY, defaults.l1_capacity);

        Ok(Self {
            port,
            bind_addr,
            storage_path,
            model_path,
            reranker_path,
            qdrant_url,
            l1_capacity,
        })
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.storage_path.exists() && !self.storage_path.is_dir() {
            return Err(ConfigError::NotADirectory {
                path: self.storage_path.clone(),
            });
        }

        if let Some(ref path) = self.model_path {
            if !path.exists() {
                return Err(ConfigError::PathNotFound { path: path.clone() });
            }
            if !path.is_file() {
                return Err(ConfigError::NotAFile { path: path.clone() });
            }
        }

        if let Some(ref path) = self.reranker_path {
            if !path.exists() {
                return Err(ConfigError::PathNotFound { path: path.clone() });
            }
            if !path.is_dir() {
                return Err(ConfigError::NotADirectory { path: path.clone() });
            }
        }

        Ok(())
    }

    pub fn socket_addr(&self) -> String {
        format!("{}:{}", self.bind_addr, self.port)
    }

    fn parse_port_from_env(default: u16) -> Result<u16, ConfigError> {
        match env::var(Self::ENV_PORT) {
            Ok(value) => {
                let port: u16 = value.parse().map_err(|e| ConfigError::PortParseError {
                    value: value.clone(),
                    source: e,
                })?;

                if port == 0 {
                    return Err(ConfigError::InvalidPort { value });
                }

                Ok(port)
            }
            Err(_) => Ok(default),
        }
    }

    fn parse_bind_addr_from_env(default: IpAddr) -> Result<IpAddr, ConfigError> {
        match env::var(Self::ENV_BIND_ADDR) {
            Ok(value) => value
                .parse()
                .map_err(|e| ConfigError::InvalidBindAddr { value, source: e }),
            Err(_) => Ok(default),
        }
    }

    fn parse_path_from_env(var_name: &str, default: PathBuf) -> PathBuf {
        env::var(var_name).map(PathBuf::from).unwrap_or(default)
    }

    fn parse_optional_path_from_env(var_name: &str) -> Option<PathBuf> {
        env::var(var_name)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    }

    fn parse_string_from_env(var_name: &str, default: String) -> String {
        env::var(var_name).unwrap_or(default)
    }

    fn parse_u64_from_env(var_name: &str, default: u64) -> u64 {
        env::var(var_name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }
}

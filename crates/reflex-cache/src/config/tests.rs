use super::*;
use serial_test::serial;
use std::env;
use std::net::IpAddr;
use std::path::PathBuf;

fn with_env_vars<F, R>(vars: &[(&str, &str)], f: F) -> R
where
    F: FnOnce() -> R,
{
    // SAFETY: Test code only, we accept the thread-safety risk in tests.
    for (key, value) in vars {
        unsafe { env::set_var(key, value) };
    }

    let result = f();

    // SAFETY: Test code only, we accept the thread-safety risk in tests.
    for (key, _) in vars {
        unsafe { env::remove_var(key) };
    }

    result
}

fn clear_reflex_env() {
    // SAFETY: Test code only, we accept the thread-safety risk in tests.
    unsafe {
        env::remove_var("REFLEX_PORT");
        env::remove_var("REFLEX_BIND_ADDR");
        env::remove_var("REFLEX_STORAGE_PATH");
        env::remove_var("REFLEX_MODEL_PATH");
        env::remove_var("REFLEX_RERANKER_PATH");
        env::remove_var("REFLEX_QDRANT_URL");
        env::remove_var("REFLEX_L1_CAPACITY");
    }
}

#[test]
fn test_default_config() {
    let config = Config::default();

    assert_eq!(config.port, 8080);
    assert_eq!(
        config.bind_addr,
        IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1))
    );
    assert_eq!(config.storage_path, PathBuf::from("./.data"));
    assert!(config.model_path.is_none());
    assert!(config.reranker_path.is_none());
    assert_eq!(config.qdrant_url, "http://localhost:6334");
}

#[test]
fn test_socket_addr() {
    let config = Config::default();
    assert_eq!(config.socket_addr(), "127.0.0.1:8080");

    let config = Config {
        port: 3000,
        bind_addr: IpAddr::V4(std::net::Ipv4Addr::new(0, 0, 0, 0)),
        ..Default::default()
    };
    assert_eq!(config.socket_addr(), "0.0.0.0:3000");
}

#[test]
#[serial]
fn test_from_env_with_defaults() {
    clear_reflex_env();

    let config = Config::from_env().expect("should parse with defaults");

    assert_eq!(config.port, 8080);
    assert_eq!(
        config.bind_addr,
        IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1))
    );
}

#[test]
#[serial]
fn test_from_env_custom_port() {
    clear_reflex_env();

    with_env_vars(&[("REFLEX_PORT", "3000")], || {
        let config = Config::from_env().expect("should parse");
        assert_eq!(config.port, 3000);
    });
}

#[test]
#[serial]
fn test_from_env_custom_bind_addr() {
    clear_reflex_env();

    with_env_vars(&[("REFLEX_BIND_ADDR", "0.0.0.0")], || {
        let config = Config::from_env().expect("should parse");
        assert_eq!(
            config.bind_addr,
            IpAddr::V4(std::net::Ipv4Addr::new(0, 0, 0, 0))
        );
    });
}

#[test]
#[serial]
fn test_from_env_ipv6_bind_addr() {
    clear_reflex_env();

    with_env_vars(&[("REFLEX_BIND_ADDR", "::1")], || {
        let config = Config::from_env().expect("should parse");
        assert_eq!(
            config.bind_addr,
            IpAddr::V6(std::net::Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1))
        );
    });
}

#[test]
#[serial]
fn test_from_env_custom_paths() {
    clear_reflex_env();

    with_env_vars(
        &[
            ("REFLEX_STORAGE_PATH", "/mnt/nvme/reflex_data"),
            ("REFLEX_MODEL_PATH", "/models/qwen3-8b-q4.gguf"),
            ("REFLEX_RERANKER_PATH", "/models/modernbert-gte"),
        ],
        || {
            let config = Config::from_env().expect("should parse");

            assert_eq!(config.storage_path, PathBuf::from("/mnt/nvme/reflex_data"));
            assert_eq!(
                config.model_path,
                Some(PathBuf::from("/models/qwen3-8b-q4.gguf"))
            );
            assert_eq!(
                config.reranker_path,
                Some(PathBuf::from("/models/modernbert-gte"))
            );
        },
    );
}

#[test]
#[serial]
fn test_invalid_port_zero() {
    clear_reflex_env();

    with_env_vars(&[("REFLEX_PORT", "0")], || {
        let result = Config::from_env();
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, ConfigError::InvalidPort { .. }));
        assert!(err.to_string().contains("invalid port"));
    });
}

#[test]
#[serial]
fn test_invalid_port_not_number() {
    clear_reflex_env();

    with_env_vars(&[("REFLEX_PORT", "not_a_port")], || {
        let result = Config::from_env();
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, ConfigError::PortParseError { .. }));
        assert!(err.to_string().contains("failed to parse port"));
    });
}

#[test]
#[serial]
fn test_invalid_port_too_large() {
    clear_reflex_env();

    with_env_vars(&[("REFLEX_PORT", "99999")], || {
        let result = Config::from_env();
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, ConfigError::PortParseError { .. }));
    });
}

#[test]
#[serial]
fn test_invalid_bind_addr() {
    clear_reflex_env();

    with_env_vars(&[("REFLEX_BIND_ADDR", "not.an.ip.address")], || {
        let result = Config::from_env();
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, ConfigError::InvalidBindAddr { .. }));
        assert!(err.to_string().contains("failed to parse bind address"));
    });
}

#[test]
fn test_validate_nonexistent_model_path() {
    let config = Config {
        model_path: Some(PathBuf::from("/nonexistent/path/to/model.gguf")),
        ..Default::default()
    };

    let result = config.validate();
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(matches!(err, ConfigError::PathNotFound { .. }));
}

#[test]
fn test_validate_nonexistent_reranker_path() {
    let config = Config {
        reranker_path: Some(PathBuf::from("/nonexistent/path/to/reranker")),
        ..Default::default()
    };

    let result = config.validate();
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(matches!(err, ConfigError::PathNotFound { .. }));
}

#[test]
fn test_validate_storage_path_is_file() {
    let config = Config {
        storage_path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"),
        ..Default::default()
    };

    let result = config.validate();
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(matches!(err, ConfigError::NotADirectory { .. }));
}

#[test]
fn test_validate_model_path_is_directory() {
    let config = Config {
        model_path: Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")),
        ..Default::default()
    };

    let result = config.validate();
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(matches!(err, ConfigError::NotAFile { .. }));
}

#[test]
#[serial]
fn test_full_config_parse() {
    clear_reflex_env();

    with_env_vars(
        &[
            ("REFLEX_PORT", "8080"),
            ("REFLEX_BIND_ADDR", "0.0.0.0"),
            ("REFLEX_STORAGE_PATH", "/mnt/nvme/reflex_data"),
            ("REFLEX_MODEL_PATH", "/models/qwen3-8b-q4.gguf"),
            ("REFLEX_RERANKER_PATH", "/models/modernbert-gte"),
            ("REFLEX_QDRANT_URL", "http://qdrant.cluster:6334"),
        ],
        || {
            let config = Config::from_env().expect("should parse full config");

            assert_eq!(config.port, 8080);
            assert_eq!(
                config.bind_addr,
                IpAddr::V4(std::net::Ipv4Addr::new(0, 0, 0, 0))
            );
            assert_eq!(config.storage_path, PathBuf::from("/mnt/nvme/reflex_data"));
            assert_eq!(
                config.model_path,
                Some(PathBuf::from("/models/qwen3-8b-q4.gguf"))
            );
            assert_eq!(
                config.reranker_path,
                Some(PathBuf::from("/models/modernbert-gte"))
            );
            assert_eq!(config.qdrant_url, "http://qdrant.cluster:6334");
            assert_eq!(config.socket_addr(), "0.0.0.0:8080");
        },
    );
}

/// Test that reranker_path pointing to a file (not a directory) returns NotADirectory error.
/// This covers the validation branch at lines 98-99 in mod.rs.
#[test]
fn test_validate_reranker_path_is_file() {
    // Use Cargo.toml as a file that definitely exists
    let config = Config {
        reranker_path: Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")),
        ..Default::default()
    };

    let result = config.validate();
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(matches!(err, ConfigError::NotADirectory { .. }));
}

/// Test that validate() returns Ok(()) when all paths are valid.
/// This covers the successful validation path at line 103 in mod.rs.
#[test]
fn test_validate_success_with_valid_paths() {
    // Use existing directories and files from the project
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let config = Config {
        // storage_path can be non-existent (only checked if it exists AND is not a dir)
        storage_path: manifest_dir.join("src"),
        // model_path: use an actual file that exists
        model_path: Some(manifest_dir.join("Cargo.toml")),
        // reranker_path: use an actual directory that exists
        reranker_path: Some(manifest_dir.join("src")),
        ..Default::default()
    };

    let result = config.validate();
    assert!(result.is_ok(), "validate() should succeed with valid paths");
}

/// Test that validate() returns Ok(()) with default config (no optional paths set).
/// This also covers line 103.
#[test]
fn test_validate_success_with_defaults() {
    let config = Config::default();

    // Default config has no model_path or reranker_path, and storage_path doesn't exist yet
    let result = config.validate();
    assert!(
        result.is_ok(),
        "validate() should succeed with default config"
    );
}

/// Test parsing of REFLEX_L1_CAPACITY environment variable with a valid value.
/// This covers the parse closure branch at line 156 in mod.rs.
#[test]
#[serial]
fn test_from_env_custom_l1_capacity() {
    clear_reflex_env();

    with_env_vars(&[("REFLEX_L1_CAPACITY", "50000")], || {
        let config = Config::from_env().expect("should parse");
        assert_eq!(config.l1_capacity, 50000);
    });
}

/// Test that invalid (non-numeric) L1 capacity falls back to default.
/// This exercises the .ok() branch that converts parse errors to None.
#[test]
#[serial]
fn test_from_env_invalid_l1_capacity_uses_default() {
    clear_reflex_env();

    with_env_vars(&[("REFLEX_L1_CAPACITY", "not_a_number")], || {
        let config = Config::from_env().expect("should parse with fallback");
        assert_eq!(config.l1_capacity, 10_000); // default value
    });
}

#[test]
fn test_error_messages_are_descriptive() {
    let err = ConfigError::InvalidPort {
        value: "0".to_string(),
    };
    assert!(err.to_string().contains("invalid port"));
    assert!(err.to_string().contains("0"));
    assert!(err.to_string().contains("1 and 65535"));

    let err = ConfigError::PathNotFound {
        path: PathBuf::from("/some/path"),
    };
    assert!(err.to_string().contains("/some/path"));

    let err = ConfigError::MissingEnvVar {
        name: "REFLEX_MODEL_PATH",
    };
    assert!(err.to_string().contains("REFLEX_MODEL_PATH"));
}

use super::*;
use std::path::PathBuf;

#[test]
fn test_config_default() {
    let config = RerankerConfig::default();

    assert!(config.model_path.is_none());
    assert_eq!(config.threshold, DEFAULT_THRESHOLD);
}

#[test]
fn test_config_new() {
    let config = RerankerConfig::new("/models/modernbert.gguf");

    assert_eq!(
        config.model_path,
        Some(PathBuf::from("/models/modernbert.gguf"))
    );
    assert_eq!(config.threshold, DEFAULT_THRESHOLD);
}

#[test]
fn test_config_stub() {
    let config = RerankerConfig::stub();

    assert!(config.model_path.is_none());
}

#[test]
fn test_config_with_threshold() {
    let config = RerankerConfig::default().with_threshold(0.95);

    assert_eq!(config.threshold, 0.95);
}

#[test]
#[should_panic(expected = "threshold must be between 0.0 and 1.0")]
fn test_config_invalid_threshold_high() {
    let _ = RerankerConfig::default().with_threshold(1.5);
}

#[test]
#[should_panic(expected = "threshold must be between 0.0 and 1.0")]
fn test_config_invalid_threshold_negative() {
    let _ = RerankerConfig::default().with_threshold(-0.1);
}

#[test]
fn test_config_validate() {
    let valid = RerankerConfig::default();
    assert!(valid.validate().is_ok());

    let invalid = RerankerConfig {
        threshold: 1.5,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());
}

#[test]
fn test_stub_checker_creation() {
    let checker = Reranker::stub().unwrap();

    assert!(!checker.is_model_loaded());
    assert_eq!(checker.threshold(), DEFAULT_THRESHOLD);
}

#[test]
fn test_load_with_missing_model() {
    let config = RerankerConfig::new("/nonexistent/path/model.gguf");
    let result = Reranker::load(config);

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        RerankerError::ModelLoadFailed { .. }
    ));
}

#[test]
fn test_load_stub_config() {
    let config = RerankerConfig::stub();
    let result = Reranker::load(config);

    assert!(result.is_ok());
    let checker = result.unwrap();
    assert!(!checker.is_model_loaded());
}

#[test]
fn test_score_basic() {
    let checker = Reranker::stub().unwrap();

    let score = checker.score("query", "candidate").unwrap();

    assert!(score >= 0.0);
    assert!(score <= 1.0);
}

#[test]
fn test_score_determinism() {
    let checker = Reranker::stub().unwrap();

    let score1 = checker
        .score("What is Rust?", "Rust is a language")
        .unwrap();
    let score2 = checker
        .score("What is Rust?", "Rust is a language")
        .unwrap();

    assert_eq!(score1, score2);
}

#[test]
fn test_score_similar_texts_higher() {
    let checker = Reranker::stub().unwrap();

    let similar_score = checker
        .score("What is Rust?", "Rust is a systems programming language")
        .unwrap();
    let dissimilar_score = checker
        .score("What is Rust?", "Python is great for data science")
        .unwrap();

    assert!(similar_score > dissimilar_score);
}

#[test]
fn test_rerank_ordering() {
    let checker = Reranker::stub().unwrap();

    let candidates = vec![
        "Python is a scripting language",
        "Rust is a systems programming language",
        "JavaScript runs in browsers",
    ];

    let ranked = checker.rerank("What is Rust?", &candidates).unwrap();

    assert_eq!(ranked.len(), 3);
    assert!(ranked[0].1 >= ranked[1].1);
    assert!(ranked[1].1 >= ranked[2].1);
}

#[test]
fn test_rerank_with_threshold() {
    let config = RerankerConfig::stub().with_threshold(0.5);
    let checker = Reranker::load(config).unwrap();

    let candidates = vec!["Rust Rust Rust Rust Rust", "completely different topic"];

    let hits = checker
        .rerank_with_threshold("What is Rust?", &candidates)
        .unwrap();

    assert!(!hits.is_empty() || checker.threshold() > 0.9);
}

#[test]
fn test_rerank_empty_candidates() {
    let checker = Reranker::stub().unwrap();

    let candidates: Vec<&str> = vec![];
    let ranked = checker.rerank("query", &candidates).unwrap();

    assert!(ranked.is_empty());
}

#[test]
fn test_is_hit() {
    let config = RerankerConfig::stub().with_threshold(0.5);
    let checker = Reranker::load(config).unwrap();

    assert!(checker.is_hit(0.6));
    assert!(!checker.is_hit(0.5));
    assert!(!checker.is_hit(0.4));
}

#[test]
fn test_error_invalid_config() {
    let config = RerankerConfig {
        threshold: 2.0,
        ..Default::default()
    };

    let result = Reranker::load(config);
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(matches!(err, RerankerError::InvalidConfig { .. }));
}

#[test]
fn test_error_messages_descriptive() {
    let err = RerankerError::ModelNotFound {
        path: PathBuf::from("/some/path"),
    };
    assert!(err.to_string().contains("/some/path"));

    let err = RerankerError::NotAvailable {
        reason: "test reason".to_string(),
    };
    assert!(err.to_string().contains("test reason"));
}

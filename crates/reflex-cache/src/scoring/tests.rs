use super::error::ScoringError;
use super::scorer::CrossEncoderScorer;
use super::types::{VerificationResult, VerifiedCandidate};
use crate::cache::ReflexStatus;
use crate::embedding::{RerankerConfig, RerankerError};
use crate::storage::CacheEntry;

fn create_test_entry(payload: &str) -> CacheEntry {
    CacheEntry {
        tenant_id: 1,
        context_hash: 2,
        timestamp: 1702500000,
        embedding: vec![],
        payload_blob: payload.as_bytes().to_vec(),
    }
}

fn create_test_entry_with_context(payload: &str, tenant_id: u64, context_hash: u64) -> CacheEntry {
    CacheEntry {
        tenant_id,
        context_hash,
        timestamp: 1702500000,
        embedding: vec![],
        payload_blob: payload.as_bytes().to_vec(),
    }
}

#[test]
fn test_verification_result_to_cache_status() {
    assert_eq!(
        VerificationResult::Verified { score: 0.99 }.to_cache_status(),
        ReflexStatus::HitL3Verified
    );
    assert_eq!(
        VerificationResult::Rejected { top_score: 0.5 }.to_cache_status(),
        ReflexStatus::Miss
    );
    assert_eq!(
        VerificationResult::NoCandidates.to_cache_status(),
        ReflexStatus::Miss
    );
}

#[test]
fn test_verification_result_is_verified() {
    assert!(VerificationResult::Verified { score: 0.99 }.is_verified());
    assert!(!VerificationResult::Rejected { top_score: 0.5 }.is_verified());
    assert!(!VerificationResult::NoCandidates.is_verified());
}

#[test]
fn test_verification_result_score() {
    assert_eq!(
        VerificationResult::Verified { score: 0.99 }.score(),
        Some(0.99)
    );
    assert_eq!(
        VerificationResult::Rejected { top_score: 0.5 }.score(),
        Some(0.5)
    );
    assert_eq!(VerificationResult::NoCandidates.score(), None);
}

#[test]
fn test_verification_result_debug_status() {
    assert_eq!(
        VerificationResult::Verified { score: 0.99 }.debug_status(),
        "VERIFIED"
    );
    assert_eq!(
        VerificationResult::Rejected { top_score: 0.5 }.debug_status(),
        "REJECTED"
    );
    assert_eq!(
        VerificationResult::NoCandidates.debug_status(),
        "NO_CANDIDATES"
    );
}

#[test]
fn test_verification_result_display() {
    assert!(format!("{}", VerificationResult::Verified { score: 0.99 }).contains("0.99"));
    assert!(format!("{}", VerificationResult::Rejected { top_score: 0.5 }).contains("0.5"));
    assert_eq!(
        format!("{}", VerificationResult::NoCandidates),
        "NO_CANDIDATES"
    );
}

#[test]
fn test_verification_result_equality() {
    assert_eq!(
        VerificationResult::NoCandidates,
        VerificationResult::NoCandidates
    );
    assert_eq!(
        VerificationResult::Verified { score: 0.99 },
        VerificationResult::Verified { score: 0.99 }
    );
    assert_ne!(
        VerificationResult::Verified { score: 0.99 },
        VerificationResult::Rejected { top_score: 0.99 }
    );
}

#[test]
fn test_verified_candidate_new() {
    let entry = create_test_entry("test payload");
    let candidate = VerifiedCandidate::new(entry.clone(), 0.95, 0.85);

    assert_eq!(candidate.cross_encoder_score, 0.95);
    assert_eq!(candidate.original_score, 0.85);
    assert_eq!(candidate.entry.payload_blob, entry.payload_blob);
}

#[test]
fn test_verified_candidate_exceeds_threshold() {
    let entry = create_test_entry("test");
    let high_score = VerifiedCandidate::new(entry.clone(), 0.99, 0.85);
    let low_score = VerifiedCandidate::new(entry, 0.50, 0.85);

    assert!(high_score.exceeds_threshold(crate::constants::DEFAULT_VERIFICATION_THRESHOLD));
    assert!(!high_score.exceeds_threshold(0.99));
    assert!(!low_score.exceeds_threshold(crate::constants::DEFAULT_VERIFICATION_THRESHOLD));
}

#[test]
fn test_scorer_stub_creation() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    assert!(!scorer.is_model_loaded());
    assert_eq!(
        scorer.threshold(),
        crate::constants::DEFAULT_VERIFICATION_THRESHOLD
    );
}

#[test]
fn test_scorer_new_with_stub_config() {
    let config = RerankerConfig::stub();
    let scorer = CrossEncoderScorer::new(config).unwrap();

    assert!(!scorer.is_model_loaded());
}

#[test]
fn test_scorer_new_with_custom_threshold() {
    let config = RerankerConfig::stub().with_threshold(0.90);
    let scorer = CrossEncoderScorer::new(config).unwrap();

    assert_eq!(scorer.threshold(), 0.90);
}

#[test]
fn test_score_single_pair() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let score = scorer
        .score("What is Rust?", "Rust is a programming language")
        .unwrap();

    assert!(score >= 0.0);
    assert!(score <= 1.0);
}

#[test]
fn test_score_determinism() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let score1 = scorer.score("query", "candidate").unwrap();
    let score2 = scorer.score("query", "candidate").unwrap();

    assert_eq!(score1, score2);
}

#[test]
fn test_similar_content_scores_higher() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let similar = scorer
        .score("What is Rust?", "Rust is a systems programming language")
        .unwrap();
    let dissimilar = scorer
        .score("What is Rust?", "Python is great for data science")
        .unwrap();

    assert!(similar > dissimilar);
}

#[test]
fn test_verify_empty_candidates() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let (result, verification) = scorer.verify_candidates("query", vec![]).unwrap();

    assert!(result.is_none());
    assert_eq!(verification, VerificationResult::NoCandidates);
}

#[test]
fn test_verify_candidates_returns_best_match() {
    let config = RerankerConfig::stub().with_threshold(0.5);
    let scorer = CrossEncoderScorer::new(config).unwrap();

    let candidates = vec![
        (
            create_test_entry("Rust is a systems programming language"),
            0.85,
        ),
        (
            create_test_entry("Python is used for machine learning"),
            0.82,
        ),
    ];

    let (result, verification) = scorer
        .verify_candidates("What is Rust?", candidates)
        .unwrap();

    if let VerificationResult::Verified { score } = verification {
        assert!(result.is_some());
        assert!(score > 0.5);
    }
}

#[test]
fn test_verify_candidates_miss_when_below_threshold() {
    let config = RerankerConfig::stub().with_threshold(0.99);
    let scorer = CrossEncoderScorer::new(config).unwrap();

    let candidates = vec![
        (
            create_test_entry("completely unrelated content xyz abc"),
            0.50,
        ),
        (create_test_entry("more unrelated stuff def ghi"), 0.45),
    ];

    let (result, verification) = scorer
        .verify_candidates("What is Rust?", candidates)
        .unwrap();

    assert!(result.is_none());
    assert!(matches!(verification, VerificationResult::Rejected { .. }));
}

#[test]
fn test_verify_candidates_sorting() {
    let config = RerankerConfig::stub().with_threshold(0.0);
    let scorer = CrossEncoderScorer::new(config).unwrap();

    let candidates = vec![
        (create_test_entry("poor match xyz"), 0.90),
        (create_test_entry("Rust Rust Rust Rust"), 0.50),
        (create_test_entry("medium match"), 0.70),
    ];

    let (scored, _) = scorer
        .verify_candidates_with_details("What is Rust?", candidates)
        .unwrap();

    for i in 1..scored.len() {
        assert!(
            scored[i - 1].cross_encoder_score >= scored[i].cross_encoder_score,
            "Candidates should be sorted by cross-encoder score descending"
        );
    }
}

#[test]
fn test_score_candidates_preserves_original_scores() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let candidates = vec![
        (create_test_entry("content one"), 0.85),
        (create_test_entry("content two"), 0.75),
    ];

    let scored = scorer.score_candidates("query", candidates).unwrap();

    assert_eq!(scored.len(), 2);
    assert_eq!(scored[0].original_score, 0.85);
    assert_eq!(scored[1].original_score, 0.75);
}

#[test]
fn test_score_candidates_all_valid_range() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let candidates: Vec<_> = (0..10)
        .map(|i| {
            (
                create_test_entry(&format!("candidate {}", i)),
                i as f32 / 10.0,
            )
        })
        .collect();

    let scored = scorer.score_candidates("test query", candidates).unwrap();

    for candidate in &scored {
        assert!(
            candidate.cross_encoder_score >= 0.0,
            "Score {} should be >= 0.0",
            candidate.cross_encoder_score
        );
        assert!(
            candidate.cross_encoder_score <= 1.0,
            "Score {} should be <= 1.0",
            candidate.cross_encoder_score
        );
    }
}

#[test]
fn test_rerank_top_n_limits_results() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let candidates: Vec<_> = (0..20)
        .map(|i| (create_test_entry(&format!("candidate {}", i)), 0.5))
        .collect();

    let top5 = scorer.rerank_top_n("query", candidates, 5).unwrap();

    assert_eq!(top5.len(), 5);
}

#[test]
fn test_rerank_top_n_sorted() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let candidates = vec![
        (create_test_entry("Rust is great"), 0.8),
        (create_test_entry("Python rules"), 0.7),
        (create_test_entry("Rust Rust Rust"), 0.6),
    ];

    let top = scorer.rerank_top_n("What is Rust?", candidates, 3).unwrap();

    for i in 1..top.len() {
        assert!(top[i - 1].cross_encoder_score >= top[i].cross_encoder_score);
    }
}

#[test]
fn test_rerank_top_n_handles_small_input() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let candidates = vec![(create_test_entry("only one"), 0.5)];

    let top = scorer.rerank_top_n("query", candidates, 10).unwrap();

    assert_eq!(top.len(), 1);
}

#[test]
fn test_verify_with_details_returns_all() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let candidates = vec![
        (create_test_entry("one"), 0.9),
        (create_test_entry("two"), 0.8),
        (create_test_entry("three"), 0.7),
    ];

    let (scored, _) = scorer
        .verify_candidates_with_details("query", candidates)
        .unwrap();

    assert_eq!(scored.len(), 3);
}

#[test]
fn test_verify_with_details_correct_result() {
    let config = RerankerConfig::stub().with_threshold(0.5);
    let scorer = CrossEncoderScorer::new(config).unwrap();

    let candidates = vec![(create_test_entry("Rust Rust Rust Rust Rust"), 0.9)];

    let (_, result) = scorer
        .verify_candidates_with_details("What is Rust?", candidates)
        .unwrap();

    match result {
        VerificationResult::Verified { score } => {
            assert!(score > 0.5);
        }
        VerificationResult::Rejected { top_score } => {
            assert!(top_score <= 0.5);
        }
        VerificationResult::NoCandidates => {
            panic!("Should have candidates");
        }
    }
}

#[test]
fn test_scoring_error_from_reranker() {
    let err = ScoringError::from(RerankerError::InferenceFailed {
        reason: "test error".to_string(),
    });

    assert!(err.to_string().contains("reranker"));
}

#[test]
fn test_scoring_error_invalid_input() {
    let err = ScoringError::InvalidInput {
        reason: "empty query".to_string(),
    };

    assert!(err.to_string().contains("invalid input"));
    assert!(err.to_string().contains("empty query"));
}

#[test]
fn test_empty_payload_blob() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let entry = CacheEntry {
        tenant_id: 1,
        context_hash: 2,
        timestamp: 1000,
        embedding: vec![],
        payload_blob: vec![],
    };

    let candidates = vec![(entry, 0.5)];
    let result = scorer.verify_candidates("query", candidates);

    assert!(result.is_ok());
}

#[test]
fn test_unicode_payload() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let entry = create_test_entry("Rust ist eine Programmiersprache");
    let candidates = vec![(entry, 0.5)];

    let result = scorer.verify_candidates("What is Rust?", candidates);
    assert!(result.is_ok());
}

#[test]
fn test_very_long_payload() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let long_text = "Rust ".repeat(1000);
    let entry = create_test_entry(&long_text);
    let candidates = vec![(entry, 0.5)];

    let result = scorer.verify_candidates("What is Rust?", candidates);
    assert!(result.is_ok());
}

#[test]
fn test_special_characters_in_query() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let entry = create_test_entry("Hello world!");
    let candidates = vec![(entry, 0.5)];

    let result = scorer.verify_candidates("Hello! @#$% world?", candidates);
    assert!(result.is_ok());
}

#[test]
fn test_full_l3_verification_flow() {
    let config = RerankerConfig::stub().with_threshold(0.7);
    let scorer = CrossEncoderScorer::new(config).unwrap();

    let candidates = vec![
        (
            create_test_entry("Rust is a systems programming language focused on safety"),
            0.92,
        ),
        (create_test_entry("Python is an interpreted language"), 0.88),
        (
            create_test_entry("Rust prevents memory bugs at compile time"),
            0.85,
        ),
        (create_test_entry("JavaScript runs in web browsers"), 0.82),
        (create_test_entry("Rust has a steep learning curve"), 0.80),
    ];

    let (result, verification) = scorer
        .verify_candidates("What is Rust?", candidates)
        .unwrap();

    println!("L3 Verification Result: {:?}", verification);

    assert!(
        matches!(
            verification,
            VerificationResult::Verified { .. } | VerificationResult::Rejected { .. }
        ),
        "Should have a verification result with score"
    );

    if verification.is_verified() {
        assert!(result.is_some());
        let entry = result.unwrap();
        let text = String::from_utf8_lossy(&entry.payload_blob);
        assert!(text.to_lowercase().contains("rust"));

        assert_eq!(verification.to_cache_status(), ReflexStatus::HitL3Verified);
    } else {
        assert!(result.is_none());
        assert_eq!(verification.to_cache_status(), ReflexStatus::Miss);
    }
}

#[test]
fn test_multi_tenant_candidates() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let candidates = vec![
        (create_test_entry_with_context("Rust is great", 100, 1), 0.9),
        (
            create_test_entry_with_context("Rust is awesome", 200, 2),
            0.85,
        ),
    ];

    let scored = scorer
        .score_candidates("What is Rust?", candidates)
        .unwrap();

    assert_eq!(scored.len(), 2);
    assert_eq!(scored[0].entry.tenant_id, 100);
    assert_eq!(scored[1].entry.tenant_id, 200);
}

#[test]
fn test_scorer_debug_format() {
    let scorer = CrossEncoderScorer::stub().unwrap();
    let debug_str = format!("{:?}", scorer);

    assert!(debug_str.contains("CrossEncoderScorer"));
    assert!(debug_str.contains("reranker"));
}

#[test]
fn test_scorer_reranker_accessor() {
    let scorer = CrossEncoderScorer::stub().unwrap();
    let reranker = scorer.reranker();

    assert!(!reranker.is_model_loaded());
    assert_eq!(reranker.threshold(), scorer.threshold());
}

#[test]
fn test_verify_candidates_with_details_empty() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let (scored, result) = scorer
        .verify_candidates_with_details("query", vec![])
        .unwrap();

    assert!(scored.is_empty());
    assert_eq!(result, VerificationResult::NoCandidates);
}

#[test]
fn test_rerank_top_n_empty_candidates() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let result = scorer.rerank_top_n("query", vec![], 5).unwrap();

    assert!(result.is_empty());
}

#[test]
fn test_rerank_top_n_zero_requested() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let candidates = vec![
        (create_test_entry("content one"), 0.8),
        (create_test_entry("content two"), 0.7),
    ];

    let result = scorer.rerank_top_n("query", candidates, 0).unwrap();

    assert!(result.is_empty());
}

#[test]
fn test_score_candidates_empty() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let result = scorer.score_candidates("query", vec![]).unwrap();

    assert!(result.is_empty());
}

#[test]
fn test_verify_candidates_with_details_rejected() {
    let config = RerankerConfig::stub().with_threshold(0.99);
    let scorer = CrossEncoderScorer::new(config).unwrap();

    let candidates = vec![(create_test_entry("unrelated content xyz"), 0.5)];

    let (scored, result) = scorer
        .verify_candidates_with_details("What is Rust?", candidates)
        .unwrap();

    assert_eq!(scored.len(), 1);
    assert!(matches!(result, VerificationResult::Rejected { .. }));
}

#[test]
fn test_verify_candidates_with_details_verified() {
    let config = RerankerConfig::stub().with_threshold(0.3);
    let scorer = CrossEncoderScorer::new(config).unwrap();

    let candidates = vec![(create_test_entry("Rust Rust Rust Rust Rust"), 0.9)];

    let (scored, result) = scorer
        .verify_candidates_with_details("Rust", candidates)
        .unwrap();

    assert_eq!(scored.len(), 1);
    assert!(matches!(result, VerificationResult::Verified { .. }));
}

#[test]
fn test_verify_candidates_single_candidate_verified() {
    let config = RerankerConfig::stub().with_threshold(0.3);
    let scorer = CrossEncoderScorer::new(config).unwrap();

    let candidates = vec![(create_test_entry("Rust Rust Rust Rust"), 0.9)];

    let (result, verification) = scorer.verify_candidates("Rust", candidates).unwrap();

    assert!(result.is_some());
    assert!(matches!(verification, VerificationResult::Verified { .. }));
}

#[test]
fn test_verify_candidates_single_candidate_rejected() {
    let config = RerankerConfig::stub().with_threshold(0.99);
    let scorer = CrossEncoderScorer::new(config).unwrap();

    let candidates = vec![(create_test_entry("unrelated xyz abc"), 0.5)];

    let (result, verification) = scorer
        .verify_candidates("What is Rust?", candidates)
        .unwrap();

    assert!(result.is_none());
    assert!(matches!(verification, VerificationResult::Rejected { .. }));
}

#[test]
fn test_verify_candidates_verified_path_explicitly() {
    let config = RerankerConfig::stub().with_threshold(0.0);
    let scorer = CrossEncoderScorer::new(config).unwrap();

    let candidates = vec![(create_test_entry("Rust programming language"), 0.9)];

    let (result, verification) = scorer.verify_candidates("Rust", candidates).unwrap();

    assert!(result.is_some(), "Verified path should return Some(entry)");
    match verification {
        VerificationResult::Verified { score } => {
            assert!(score > 0.0, "Score should be positive");
        }
        _ => panic!("Expected Verified result"),
    }

    let entry = result.unwrap();
    let text = String::from_utf8_lossy(&entry.payload_blob);
    assert!(text.contains("Rust"), "Entry payload should contain 'Rust'");
}

#[test]
fn test_verify_candidates_rejected_path_explicitly() {
    let config = RerankerConfig::stub().with_threshold(1.0);
    let scorer = CrossEncoderScorer::new(config).unwrap();

    let candidates = vec![(create_test_entry("completely unrelated"), 0.5)];

    let (result, verification) = scorer.verify_candidates("query", candidates).unwrap();

    assert!(result.is_none(), "Rejected path should return None");
    match verification {
        VerificationResult::Rejected { top_score } => {
            assert!(
                (0.0..=1.0).contains(&top_score),
                "Score should be in valid range"
            );
        }
        _ => panic!("Expected Rejected result"),
    }
}

#[test]
fn test_verify_candidates_with_details_verified_path() {
    let config = RerankerConfig::stub().with_threshold(0.0);
    let scorer = CrossEncoderScorer::new(config).unwrap();

    let candidates = vec![(create_test_entry("matching content"), 0.8)];

    let (scored, result) = scorer
        .verify_candidates_with_details("matching", candidates)
        .unwrap();

    assert!(!scored.is_empty());
    assert!(matches!(result, VerificationResult::Verified { .. }));
}

#[test]
fn test_verify_candidates_with_details_rejected_path() {
    let config = RerankerConfig::stub().with_threshold(1.0);
    let scorer = CrossEncoderScorer::new(config).unwrap();

    let candidates = vec![(create_test_entry("some content"), 0.5)];

    let (scored, result) = scorer
        .verify_candidates_with_details("query", candidates)
        .unwrap();

    assert!(!scored.is_empty());
    assert!(matches!(result, VerificationResult::Rejected { .. }));
}

#[test]
fn test_score_candidates_processes_payload_blob() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let entry = CacheEntry {
        tenant_id: 1,
        context_hash: 2,
        timestamp: 1000,
        embedding: vec![],
        payload_blob: "test payload content".as_bytes().to_vec(),
    };

    let candidates = vec![(entry, 0.75)];
    let scored = scorer.score_candidates("test query", candidates).unwrap();

    assert_eq!(scored.len(), 1);
    assert_eq!(scored[0].original_score, 0.75);
    assert!(scored[0].cross_encoder_score >= 0.0);
    assert!(scored[0].cross_encoder_score <= 1.0);
}

#[test]
fn test_rerank_top_n_truncation() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let candidates: Vec<_> = (0..10)
        .map(|i| {
            (
                create_test_entry(&format!("candidate {}", i)),
                0.5 + i as f32 * 0.01,
            )
        })
        .collect();

    let top3 = scorer.rerank_top_n("query", candidates.clone(), 3).unwrap();
    assert_eq!(top3.len(), 3, "Should truncate to exactly 3");

    let top1 = scorer.rerank_top_n("query", candidates, 1).unwrap();
    assert_eq!(top1.len(), 1, "Should truncate to exactly 1");
}

#[test]
fn test_sorting_with_equal_scores() {
    let config = RerankerConfig::stub().with_threshold(0.5);
    let scorer = CrossEncoderScorer::new(config).unwrap();

    // Create candidates with content that will produce identical scores
    let candidates = vec![
        (create_test_entry("identical"), 0.9),
        (create_test_entry("identical"), 0.8),
        (create_test_entry("identical"), 0.7),
    ];

    let (scored, _) = scorer
        .verify_candidates_with_details("identical", candidates)
        .unwrap();

    assert_eq!(scored.len(), 3);
    assert_eq!(scored[0].cross_encoder_score, scored[1].cross_encoder_score);
    assert_eq!(scored[1].cross_encoder_score, scored[2].cross_encoder_score);
}

#[test]
fn test_verify_candidates_multiple_above_threshold() {
    let config = RerankerConfig::stub().with_threshold(0.0);
    let scorer = CrossEncoderScorer::new(config).unwrap();

    let candidates = vec![
        (create_test_entry("Rust language"), 0.95),
        (create_test_entry("Rust code"), 0.90),
        (create_test_entry("Rust programming"), 0.85),
    ];

    let (result, verification) = scorer.verify_candidates("Rust", candidates).unwrap();

    assert!(result.is_some());
    assert!(matches!(verification, VerificationResult::Verified { .. }));
}

#[test]
fn test_verify_candidates_original_scores_preserved() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let candidates = vec![
        (create_test_entry("first"), 0.123),
        (create_test_entry("second"), 0.456),
        (create_test_entry("third"), 0.789),
    ];

    let scored = scorer.score_candidates("query", candidates).unwrap();

    let original_scores: Vec<f32> = scored.iter().map(|c| c.original_score).collect();
    assert!(original_scores.contains(&0.123));
    assert!(original_scores.contains(&0.456));
    assert!(original_scores.contains(&0.789));
}

#[test]
fn test_binary_payload_blob() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let entry = CacheEntry {
        tenant_id: 1,
        context_hash: 2,
        timestamp: 1000,
        embedding: vec![],
        payload_blob: vec![0xFF, 0xFE, 0x00, 0x01],
    };

    let candidates = vec![(entry, 0.5)];
    let result = scorer.verify_candidates("query", candidates);
    assert!(result.is_ok());
}

#[test]
fn test_score_method_error_propagation() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let result = scorer.score("query", "candidate");
    assert!(result.is_ok());

    let score = result.unwrap();
    assert!((0.0..=1.0).contains(&score));
}

#[test]
fn test_new_with_invalid_config() {
    let config = RerankerConfig {
        model_path: Some(std::path::PathBuf::from("/nonexistent/path/to/model")),
        threshold: 0.8,
    };

    let result = CrossEncoderScorer::new(config);
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(err.to_string().contains("reranker"));
}

#[test]
fn test_debug_impl_contains_expected_fields() {
    let scorer = CrossEncoderScorer::stub().unwrap();
    let debug_output = format!("{:?}", scorer);

    assert!(debug_output.contains("CrossEncoderScorer"));
    assert!(debug_output.contains("reranker"));
}

#[test]
fn test_reranker_accessor_returns_reference() {
    let scorer = CrossEncoderScorer::stub().unwrap();
    let reranker = scorer.reranker();

    assert_eq!(reranker.is_model_loaded(), scorer.is_model_loaded());
    assert!((reranker.threshold() - scorer.threshold()).abs() < f32::EPSILON);
}

#[test]
fn test_threshold_accessor() {
    let config = RerankerConfig::stub().with_threshold(0.42);
    let scorer = CrossEncoderScorer::new(config).unwrap();

    assert!((scorer.threshold() - 0.42).abs() < f32::EPSILON);
}

#[test]
fn test_is_model_loaded_stub() {
    let scorer = CrossEncoderScorer::stub().unwrap();
    assert!(!scorer.is_model_loaded());
}

#[test]
fn test_verify_candidates_entry_clone() {
    let config = RerankerConfig::stub().with_threshold(0.0);
    let scorer = CrossEncoderScorer::new(config).unwrap();

    let original_payload = "unique test payload 12345";
    let entry = create_test_entry(original_payload);
    let original_tenant_id = entry.tenant_id;
    let original_context_hash = entry.context_hash;

    let candidates = vec![(entry, 0.9)];

    let (result, _) = scorer.verify_candidates("test", candidates).unwrap();

    let returned_entry = result.expect("Should have entry");
    assert_eq!(returned_entry.tenant_id, original_tenant_id);
    assert_eq!(returned_entry.context_hash, original_context_hash);
    assert_eq!(
        String::from_utf8_lossy(&returned_entry.payload_blob),
        original_payload
    );
}

#[test]
fn test_scoring_error_computation_failed() {
    let err = ScoringError::ComputationFailed {
        reason: "test computation error".to_string(),
    };

    let err_str = err.to_string();
    assert!(err_str.contains("computation failed"));
    assert!(err_str.contains("test computation error"));
}

#[test]
fn test_scoring_error_debug_format() {
    let err = ScoringError::InvalidInput {
        reason: "debug test".to_string(),
    };

    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("InvalidInput"));
    assert!(debug_str.contains("debug test"));
}

#[test]
fn test_rerank_top_n_with_various_n_values() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let candidates: Vec<_> = (0..5)
        .map(|i| (create_test_entry(&format!("candidate {}", i)), 0.5))
        .collect();

    let top0 = scorer.rerank_top_n("query", candidates.clone(), 0).unwrap();
    assert_eq!(top0.len(), 0);

    let top1 = scorer.rerank_top_n("query", candidates.clone(), 1).unwrap();
    assert_eq!(top1.len(), 1);

    let top5 = scorer.rerank_top_n("query", candidates.clone(), 5).unwrap();
    assert_eq!(top5.len(), 5);

    let top10 = scorer.rerank_top_n("query", candidates, 10).unwrap();
    assert_eq!(top10.len(), 5);
}

#[test]
fn test_score_candidates_single_entry() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let candidates = vec![(create_test_entry("single entry"), 0.77)];
    let scored = scorer.score_candidates("test query", candidates).unwrap();

    assert_eq!(scored.len(), 1);
    assert_eq!(scored[0].original_score, 0.77);
}

#[test]
fn test_verify_candidates_preserves_entry_fields() {
    let config = RerankerConfig::stub().with_threshold(0.0);
    let scorer = CrossEncoderScorer::new(config).unwrap();

    let entry = CacheEntry {
        tenant_id: 42,
        context_hash: 12345,
        timestamp: 1702500000,
        embedding: vec![1, 2, 3],
        payload_blob: "test content".as_bytes().to_vec(),
    };

    let candidates = vec![(entry, 0.9)];
    let (result, _) = scorer.verify_candidates("test", candidates).unwrap();

    let returned_entry = result.unwrap();
    assert_eq!(returned_entry.tenant_id, 42);
    assert_eq!(returned_entry.context_hash, 12345);
    assert_eq!(returned_entry.timestamp, 1702500000);
    assert_eq!(returned_entry.embedding, vec![1u8, 2u8, 3u8]);
}

#[test]
fn test_verified_candidate_fields() {
    let entry = create_test_entry("payload");
    let candidate = VerifiedCandidate::new(entry.clone(), 0.88, 0.77);

    assert_eq!(candidate.cross_encoder_score, 0.88);
    assert_eq!(candidate.original_score, 0.77);
    assert_eq!(candidate.entry.payload_blob, entry.payload_blob);
}

#[test]
fn test_stub_creates_valid_scorer() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let score = scorer.score("test query", "test candidate").unwrap();
    assert!((0.0..=1.0).contains(&score));

    let candidates = vec![(create_test_entry("test"), 0.5)];
    let result = scorer.verify_candidates("test", candidates);
    assert!(result.is_ok());
}

#[test]
fn test_score_with_empty_strings() {
    let scorer = CrossEncoderScorer::stub().unwrap();

    let score1 = scorer.score("", "candidate").unwrap();
    assert!((0.0..=1.0).contains(&score1));

    let score2 = scorer.score("query", "").unwrap();
    assert!((0.0..=1.0).contains(&score2));

    let score3 = scorer.score("", "").unwrap();
    assert!((0.0..=1.0).contains(&score3));
}

#[test]
fn test_verify_candidates_with_large_candidate_set() {
    let config = RerankerConfig::stub().with_threshold(0.0);
    let scorer = CrossEncoderScorer::new(config).unwrap();

    let candidates: Vec<_> = (0..100)
        .map(|i| {
            (
                create_test_entry(&format!("candidate number {}", i)),
                i as f32 / 100.0,
            )
        })
        .collect();

    let (result, verification) = scorer.verify_candidates("candidate", candidates).unwrap();

    assert!(result.is_some());
    assert!(matches!(verification, VerificationResult::Verified { .. }));
}

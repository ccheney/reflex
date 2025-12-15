use super::*;

fn f16_vec(values: &[f32]) -> Vec<f16> {
    values.iter().map(|&v| f16::from_f32(v)).collect()
}

fn create_test_entry(embedding_f32: &[f32]) -> CacheEntry {
    let embedding_f16: Vec<f16> = embedding_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let embedding_bytes: Vec<u8> = embedding_f16.iter().flat_map(|v| v.to_le_bytes()).collect();

    CacheEntry {
        tenant_id: 1,
        context_hash: 2,
        timestamp: 1702500000,
        embedding: embedding_bytes,
        payload_blob: vec![],
    }
}

fn create_full_dim_entry(seed: u64) -> CacheEntry {
    let embedding: Vec<f32> = (0..DEFAULT_EMBEDDING_DIM)
        .map(|i| {
            let mixed = (seed.wrapping_mul(31).wrapping_add(i as u64)) % 1000;
            mixed as f32 / 1000.0
        })
        .collect();
    create_test_entry(&embedding)
}

#[test]
fn test_cosine_identical_vectors() {
    let v = f16_vec(&[1.0, 2.0, 3.0]);
    let similarity = cosine_similarity_f16(&v, &v);
    assert!(
        (similarity - 1.0).abs() < 0.001,
        "Identical vectors should have similarity ~1.0"
    );
}

#[test]
fn test_cosine_orthogonal_vectors() {
    let v1 = f16_vec(&[1.0, 0.0]);
    let v2 = f16_vec(&[0.0, 1.0]);
    let similarity = cosine_similarity_f16(&v1, &v2);
    assert!(
        similarity.abs() < 0.001,
        "Orthogonal vectors should have similarity ~0.0"
    );
}

#[test]
fn test_cosine_opposite_vectors() {
    let v1 = f16_vec(&[1.0, 0.0]);
    let v2 = f16_vec(&[-1.0, 0.0]);
    let similarity = cosine_similarity_f16(&v1, &v2);
    assert!(
        (similarity - (-1.0)).abs() < 0.001,
        "Opposite vectors should have similarity ~-1.0"
    );
}

#[test]
fn test_cosine_scaled_vectors() {
    let v1 = f16_vec(&[1.0, 2.0, 3.0]);
    let v2 = f16_vec(&[2.0, 4.0, 6.0]);
    let similarity = cosine_similarity_f16(&v1, &v2);
    assert!(
        (similarity - 1.0).abs() < 0.001,
        "Scaled vectors should have similarity ~1.0"
    );
}

#[test]
fn test_cosine_different_lengths() {
    let v1 = f16_vec(&[1.0, 2.0]);
    let v2 = f16_vec(&[1.0, 2.0, 3.0]);
    let similarity = cosine_similarity_f16(&v1, &v2);
    assert_eq!(
        similarity, 0.0,
        "Different length vectors should return 0.0"
    );
}

#[test]
fn test_cosine_empty_vectors() {
    let v1: Vec<f16> = vec![];
    let v2: Vec<f16> = vec![];
    let similarity = cosine_similarity_f16(&v1, &v2);
    assert_eq!(similarity, 0.0, "Empty vectors should return 0.0");
}

#[test]
fn test_cosine_zero_vector() {
    let v1 = f16_vec(&[0.0, 0.0, 0.0]);
    let v2 = f16_vec(&[1.0, 2.0, 3.0]);
    let similarity = cosine_similarity_f16(&v1, &v2);
    assert_eq!(similarity, 0.0, "Zero vector should return 0.0");
}

#[test]
fn test_cosine_full_dimension() {
    let v1: Vec<f16> = (0..DEFAULT_EMBEDDING_DIM)
        .map(|i| f16::from_f32(i as f32 / 1000.0))
        .collect();
    let v2: Vec<f16> = (0..DEFAULT_EMBEDDING_DIM)
        .map(|i| f16::from_f32(i as f32 / 1000.0))
        .collect();

    let similarity = cosine_similarity_f16(&v1, &v2);
    assert!(
        (similarity - 1.0).abs() < 0.001,
        "Same vectors should be ~1.0"
    );
}

#[test]
fn test_cosine_f16_f32_mixed() {
    let v_f16 = f16_vec(&[1.0, 2.0, 3.0]);
    let v_f32 = vec![1.0f32, 2.0, 3.0];
    let similarity = cosine_similarity_f16_f32(&v_f16, &v_f32);
    assert!((similarity - 1.0).abs() < 0.001);
}

#[test]
fn test_bytes_to_f16_slice() {
    let values = f16_vec(&[1.0, 0.5, -0.25]);
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let slice = bytes_to_f16_slice(&bytes).expect("Should convert successfully");

    assert_eq!(slice.len(), 3);
    assert!((slice[0].to_f32() - 1.0).abs() < 0.01);
    assert!((slice[1].to_f32() - 0.5).abs() < 0.01);
    assert!((slice[2].to_f32() - (-0.25)).abs() < 0.01);
}

#[test]
fn test_bytes_to_f16_odd_length() {
    let bytes = vec![0u8; 3];
    assert!(bytes_to_f16_slice(&bytes).is_none());
}

#[test]
fn test_f16_slice_to_bytes_roundtrip() {
    let original = f16_vec(&[1.0, 2.0, 3.0]);
    let bytes = f16_slice_to_bytes(&original);
    let roundtrip = bytes_to_f16_slice(bytes).unwrap();

    assert_eq!(original.len(), roundtrip.len());
    for (a, b) in original.iter().zip(roundtrip.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn test_f32_to_f16_conversion() {
    let f32_values = vec![1.0f32, 0.5, -0.25, 0.0, 100.0];
    let f16_values = f32_to_f16_vec(&f32_values);
    let back = f16_to_f32_vec(&f16_values);

    assert_eq!(f32_values.len(), back.len());
    for (orig, rt) in f32_values.iter().zip(back.iter()) {
        assert!((orig - rt).abs() < 0.1, "{} vs {}", orig, rt);
    }
}

#[test]
fn test_candidate_entry_new() {
    let entry = create_test_entry(&[1.0, 2.0, 3.0]);
    let candidate = CandidateEntry::new(42, entry.clone());

    assert_eq!(candidate.id, 42);
    assert_eq!(candidate.entry.tenant_id, 1);
    assert!(candidate.bq_score.is_none());
}

#[test]
fn test_candidate_entry_with_bq_score() {
    let entry = create_test_entry(&[1.0, 2.0, 3.0]);
    let candidate = CandidateEntry::with_bq_score(42, entry, 0.85);

    assert_eq!(candidate.id, 42);
    assert_eq!(candidate.bq_score, Some(0.85));
}

#[test]
fn test_candidate_embedding_as_f16() {
    let values = [1.0f32, 2.0, 3.0];
    let entry = create_test_entry(&values);
    let candidate = CandidateEntry::new(1, entry);

    let embedding = candidate.embedding_as_f16().expect("Should get f16 slice");
    assert_eq!(embedding.len(), 3);
}

#[test]
fn test_scored_candidate_delta() {
    let entry = create_test_entry(&[1.0]);
    let scored = ScoredCandidate {
        id: 1,
        entry,
        score: 0.95,
        bq_score: Some(0.80),
    };

    let delta = scored.score_delta().unwrap();
    assert!((delta - 0.15).abs() < 0.001);
}

#[test]
fn test_scored_candidate_no_bq_score() {
    let entry = create_test_entry(&[1.0]);
    let scored = ScoredCandidate {
        id: 1,
        entry,
        score: 0.95,
        bq_score: None,
    };

    assert!(scored.score_delta().is_none());
}

#[test]
fn test_rescorer_default() {
    let rescorer = VectorRescorer::new();
    assert_eq!(rescorer.config().top_k, DEFAULT_TOP_K);
    assert!(rescorer.config().validate_dimensions);
}

#[test]
fn test_rescorer_with_top_k() {
    let rescorer = VectorRescorer::with_top_k(10);
    assert_eq!(rescorer.config().top_k, 10);
}

#[test]
fn test_rescorer_basic() {
    let query = f16_vec(&[1.0, 0.0, 0.0]);

    let candidate1 = CandidateEntry::new(1, create_test_entry(&[1.0, 0.0, 0.0]));
    let candidate2 = CandidateEntry::new(2, create_test_entry(&[0.0, 1.0, 0.0]));
    let candidate3 = CandidateEntry::new(3, create_test_entry(&[0.5, 0.5, 0.0]));

    let config = RescorerConfig {
        validate_dimensions: false,
        top_k: 2,
    };
    let rescorer = VectorRescorer::with_config(config);

    let results = rescorer
        .rescore(&query, vec![candidate1, candidate2, candidate3])
        .expect("Should rescore");

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, 1);
    assert!((results[0].score - 1.0).abs() < 0.01);
    assert_eq!(results[1].id, 3);
}

#[test]
fn test_rescorer_returns_top_k() {
    let rescorer = VectorRescorer::with_top_k(5);

    let query: Vec<f16> = (0..DEFAULT_EMBEDDING_DIM)
        .map(|i| f16::from_f32(i as f32 / 1000.0))
        .collect();

    let candidates: Vec<CandidateEntry> = (0..10)
        .map(|i| CandidateEntry::new(i, create_full_dim_entry(i)))
        .collect();

    let results = rescorer
        .rescore(&query, candidates)
        .expect("Should rescore");

    assert_eq!(results.len(), 5, "Should return top 5");
}

#[test]
fn test_rescorer_sorted_descending() {
    let config = RescorerConfig {
        validate_dimensions: false,
        top_k: 10,
    };
    let rescorer = VectorRescorer::with_config(config);

    let query = f16_vec(&[1.0, 0.0]);

    let candidates = vec![
        CandidateEntry::new(1, create_test_entry(&[0.5, 0.5])),
        CandidateEntry::new(2, create_test_entry(&[1.0, 0.0])),
        CandidateEntry::new(3, create_test_entry(&[0.0, 1.0])),
    ];

    let results = rescorer
        .rescore(&query, candidates)
        .expect("Should rescore");

    for i in 1..results.len() {
        assert!(
            results[i - 1].score >= results[i].score,
            "Results should be sorted descending"
        );
    }
}

#[test]
fn test_rescorer_empty_candidates() {
    let rescorer = VectorRescorer::new();
    let query: Vec<f16> = vec![f16::from_f32(1.0); DEFAULT_EMBEDDING_DIM];

    let result = rescorer.rescore(&query, vec![]);

    assert!(matches!(result, Err(RescoringError::NoCandidates)));
}

#[test]
fn test_rescorer_invalid_query_dimension() {
    let rescorer = VectorRescorer::new();
    let query = f16_vec(&[1.0, 2.0, 3.0]);

    let candidates = vec![CandidateEntry::new(1, create_full_dim_entry(1))];

    let result = rescorer.rescore(&query, candidates);

    assert!(matches!(
        result,
        Err(RescoringError::InvalidQueryDimension { .. })
    ));
}

#[test]
fn test_rescorer_skips_invalid_embeddings() {
    let config = RescorerConfig {
        validate_dimensions: true,
        top_k: 10,
    };
    let rescorer = VectorRescorer::with_config(config);

    let query: Vec<f16> = vec![f16::from_f32(0.1); DEFAULT_EMBEDDING_DIM];

    let valid_entry = create_full_dim_entry(1);
    let mut invalid_entry = create_full_dim_entry(2);
    invalid_entry.embedding = vec![0u8; 100];

    let candidates = vec![
        CandidateEntry::new(1, valid_entry),
        CandidateEntry::new(2, invalid_entry),
    ];

    let results = rescorer
        .rescore(&query, candidates)
        .expect("Should rescore");

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 1);
}

#[test]
fn test_rescorer_from_bytes() {
    let config = RescorerConfig {
        validate_dimensions: false,
        top_k: 5,
    };
    let rescorer = VectorRescorer::with_config(config);

    let query_f16 = f16_vec(&[1.0, 0.0, 0.0]);
    let query_bytes: Vec<u8> = query_f16.iter().flat_map(|v| v.to_le_bytes()).collect();

    let candidates = vec![CandidateEntry::new(1, create_test_entry(&[1.0, 0.0, 0.0]))];

    let results = rescorer
        .rescore_from_bytes(&query_bytes, candidates)
        .expect("Should rescore");

    assert_eq!(results.len(), 1);
    assert!((results[0].score - 1.0).abs() < 0.01);
}

#[test]
fn test_rescorer_preserves_bq_score() {
    let config = RescorerConfig {
        validate_dimensions: false,
        ..RescorerConfig::default()
    };
    let rescorer = VectorRescorer::with_config(config);

    let query = f16_vec(&[1.0, 0.0]);
    let entry = create_test_entry(&[1.0, 0.0]);
    let candidate = CandidateEntry::with_bq_score(1, entry, 0.75);

    let results = rescorer
        .rescore(&query, vec![candidate])
        .expect("Should rescore");

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].bq_score, Some(0.75));
}

#[test]
fn test_error_messages() {
    let err = RescoringError::InvalidQueryDimension {
        expected: crate::constants::DEFAULT_EMBEDDING_DIM,
        actual: 768,
    };
    let msg = err.to_string();
    assert!(msg.contains(&crate::constants::DEFAULT_EMBEDDING_DIM.to_string()));
    assert!(msg.contains("768"));

    let err = RescoringError::InvalidEmbeddingSize {
        id: 42,
        expected: crate::constants::EMBEDDING_F16_BYTES,
        actual: 100,
    };
    let msg = err.to_string();
    assert!(msg.contains("42"));
    assert!(msg.contains(&crate::constants::EMBEDDING_F16_BYTES.to_string()));
    assert!(msg.contains("100"));

    let err = RescoringError::NoCandidates;
    assert!(err.to_string().contains("no candidates"));
}

#[test]
fn test_rescorer_large_candidate_set() {
    let rescorer = VectorRescorer::with_top_k(5);

    let query: Vec<f16> = vec![f16::from_f32(0.1); DEFAULT_EMBEDDING_DIM];

    let candidates: Vec<CandidateEntry> = (0..50)
        .map(|i| CandidateEntry::new(i, create_full_dim_entry(i)))
        .collect();

    let results = rescorer
        .rescore(&query, candidates)
        .expect("Should rescore");

    assert_eq!(results.len(), 5);
    for i in 1..results.len() {
        assert!(results[i - 1].score >= results[i].score);
    }
}

#[test]
fn test_rescorer_single_candidate() {
    let rescorer = VectorRescorer::with_top_k(5);

    let query: Vec<f16> = vec![f16::from_f32(0.1); DEFAULT_EMBEDDING_DIM];
    let candidates = vec![CandidateEntry::new(1, create_full_dim_entry(1))];

    let results = rescorer
        .rescore(&query, candidates)
        .expect("Should rescore");

    assert_eq!(results.len(), 1);
}

#[test]
fn test_rescorer_fewer_than_top_k() {
    let rescorer = VectorRescorer::with_top_k(10);

    let query: Vec<f16> = vec![f16::from_f32(0.1); DEFAULT_EMBEDDING_DIM];

    let candidates: Vec<CandidateEntry> = (0..3)
        .map(|i| CandidateEntry::new(i, create_full_dim_entry(i)))
        .collect();

    let results = rescorer
        .rescore(&query, candidates)
        .expect("Should rescore");

    assert_eq!(
        results.len(),
        3,
        "Should return all candidates when fewer than top_k"
    );
}

#[test]
fn test_rescorer_default_trait() {
    let rescorer = VectorRescorer::default();
    assert_eq!(rescorer.config().top_k, DEFAULT_TOP_K);
    assert!(rescorer.config().validate_dimensions);
}

#[test]
fn test_cosine_f16_f32_different_lengths() {
    let v_f16 = f16_vec(&[1.0, 2.0]);
    let v_f32 = vec![1.0f32, 2.0, 3.0];
    let similarity = cosine_similarity_f16_f32(&v_f16, &v_f32);
    assert_eq!(
        similarity, 0.0,
        "Different length vectors should return 0.0"
    );
}

#[test]
fn test_cosine_f16_f32_empty_vectors() {
    let v_f16: Vec<f16> = vec![];
    let v_f32: Vec<f32> = vec![];
    let similarity = cosine_similarity_f16_f32(&v_f16, &v_f32);
    assert_eq!(similarity, 0.0, "Empty vectors should return 0.0");
}

#[test]
fn test_cosine_f16_f32_zero_vector_a() {
    let v_f16 = f16_vec(&[0.0, 0.0, 0.0]);
    let v_f32 = vec![1.0f32, 2.0, 3.0];
    let similarity = cosine_similarity_f16_f32(&v_f16, &v_f32);
    assert_eq!(similarity, 0.0, "Zero vector should return 0.0");
}

#[test]
fn test_cosine_f16_f32_zero_vector_b() {
    let v_f16 = f16_vec(&[1.0, 2.0, 3.0]);
    let v_f32 = vec![0.0f32, 0.0, 0.0];
    let similarity = cosine_similarity_f16_f32(&v_f16, &v_f32);
    assert_eq!(similarity, 0.0, "Zero vector should return 0.0");
}

#[test]
fn test_cosine_f16_f32_orthogonal() {
    let v_f16 = f16_vec(&[1.0, 0.0]);
    let v_f32 = vec![0.0f32, 1.0];
    let similarity = cosine_similarity_f16_f32(&v_f16, &v_f32);
    assert!(
        similarity.abs() < 0.001,
        "Orthogonal vectors should have similarity ~0.0"
    );
}

#[test]
fn test_cosine_f16_f32_opposite() {
    let v_f16 = f16_vec(&[1.0, 0.0]);
    let v_f32 = vec![-1.0f32, 0.0];
    let similarity = cosine_similarity_f16_f32(&v_f16, &v_f32);
    assert!(
        (similarity - (-1.0)).abs() < 0.001,
        "Opposite vectors should have similarity ~-1.0"
    );
}

#[test]
fn test_rescorer_drops_invalid_f16_parse() {
    let config = RescorerConfig {
        validate_dimensions: false,
        top_k: 10,
    };
    let rescorer = VectorRescorer::with_config(config);

    let query = f16_vec(&[1.0, 0.0, 0.0]);

    // Create a valid entry
    let valid_entry = create_test_entry(&[1.0, 0.0, 0.0]);

    // Create an entry with odd-length bytes that cannot be parsed as f16
    let invalid_entry = CacheEntry {
        tenant_id: 1,
        context_hash: 2,
        timestamp: 1702500000,
        embedding: vec![0u8; 5], // Odd length - cannot be cast to f16 slice
        payload_blob: vec![],
    };

    let candidates = vec![
        CandidateEntry::new(1, valid_entry),
        CandidateEntry::new(2, invalid_entry),
    ];

    let results = rescorer
        .rescore(&query, candidates)
        .expect("Should rescore");

    // The invalid entry should be dropped
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 1);
}

#[test]
fn test_rescore_from_bytes_invalid_query() {
    let rescorer = VectorRescorer::new();

    // Odd-length bytes cannot be cast to f16
    let invalid_query_bytes = vec![0u8; 5];
    let candidates = vec![CandidateEntry::new(1, create_full_dim_entry(1))];

    let result = rescorer.rescore_from_bytes(&invalid_query_bytes, candidates);

    assert!(matches!(
        result,
        Err(RescoringError::InvalidQueryDimension { .. })
    ));
}

#[test]
fn test_bytes_to_f16_slice_empty() {
    let bytes: &[u8] = &[];
    let result = bytes_to_f16_slice(bytes);
    // bytemuck::try_cast_slice returns Err for empty slices
    // This is documented bytemuck behavior
    assert!(result.is_none());
}

#[test]
fn test_f16_slice_to_bytes_empty() {
    let values: &[f16] = &[];
    let bytes = f16_slice_to_bytes(values);
    assert!(bytes.is_empty());
}

#[test]
fn test_rescorer_drops_wrong_dimension_embedding() {
    // With validate_dimensions enabled, wrong dimension embeddings should be dropped
    let rescorer = VectorRescorer::new();

    // Create query with proper dimension
    let query: Vec<f16> = vec![f16::from_f32(0.1); DEFAULT_EMBEDDING_DIM];

    // Create a valid entry (correct dimension)
    let valid_entry = create_full_dim_entry(1);

    // Create an entry with wrong dimension (different from DEFAULT_EMBEDDING_DIM)
    // We need a valid f16 byte slice but with wrong dimension
    let wrong_dim_embedding: Vec<f16> = vec![f16::from_f32(0.5); 100]; // Wrong dimension
    let wrong_dim_bytes: Vec<u8> = wrong_dim_embedding
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let wrong_dim_entry = CacheEntry {
        tenant_id: 2,
        context_hash: 3,
        timestamp: 1702500000,
        embedding: wrong_dim_bytes,
        payload_blob: vec![],
    };

    let candidates = vec![
        CandidateEntry::new(1, valid_entry),
        CandidateEntry::new(2, wrong_dim_entry),
    ];

    let results = rescorer
        .rescore(&query, candidates)
        .expect("Should rescore");

    // The wrong dimension entry should be dropped
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 1);
}

#[test]
fn test_rescorer_config_with_top_k() {
    let config = RescorerConfig::with_top_k(15);
    assert_eq!(config.top_k, 15);
    assert!(config.validate_dimensions); // default should be true
}

#[test]
fn test_rescorer_config_clone() {
    let config = RescorerConfig {
        top_k: 20,
        validate_dimensions: false,
    };
    let cloned = config.clone();
    assert_eq!(cloned.top_k, 20);
    assert!(!cloned.validate_dimensions);
}

#[test]
fn test_rescorer_config_debug() {
    let config = RescorerConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("RescorerConfig"));
    assert!(debug_str.contains("top_k"));
}

#[test]
fn test_candidate_entry_clone() {
    let entry = create_test_entry(&[1.0, 2.0, 3.0]);
    let candidate = CandidateEntry::with_bq_score(42, entry, 0.85);
    let cloned = candidate.clone();

    assert_eq!(cloned.id, 42);
    assert_eq!(cloned.bq_score, Some(0.85));
    assert_eq!(cloned.entry.tenant_id, candidate.entry.tenant_id);
}

#[test]
fn test_candidate_entry_debug() {
    let entry = create_test_entry(&[1.0, 2.0, 3.0]);
    let candidate = CandidateEntry::new(42, entry);
    let debug_str = format!("{:?}", candidate);
    assert!(debug_str.contains("CandidateEntry"));
    assert!(debug_str.contains("42"));
}

#[test]
fn test_scored_candidate_clone() {
    let entry = create_test_entry(&[1.0]);
    let scored = ScoredCandidate {
        id: 1,
        entry,
        score: 0.95,
        bq_score: Some(0.80),
    };
    let cloned = scored.clone();

    assert_eq!(cloned.id, 1);
    assert_eq!(cloned.score, 0.95);
    assert_eq!(cloned.bq_score, Some(0.80));
}

#[test]
fn test_scored_candidate_debug() {
    let entry = create_test_entry(&[1.0]);
    let scored = ScoredCandidate {
        id: 1,
        entry,
        score: 0.95,
        bq_score: None,
    };
    let debug_str = format!("{:?}", scored);
    assert!(debug_str.contains("ScoredCandidate"));
    assert!(debug_str.contains("0.95"));
}

#[test]
fn test_vector_rescorer_clone() {
    let rescorer = VectorRescorer::with_top_k(10);
    let cloned = rescorer.clone();
    assert_eq!(cloned.config().top_k, 10);
}

#[test]
fn test_vector_rescorer_debug() {
    let rescorer = VectorRescorer::new();
    let debug_str = format!("{:?}", rescorer);
    assert!(debug_str.contains("VectorRescorer"));
}

#[test]
fn test_bytes_to_f16_slice_valid_non_empty() {
    // Test with valid, non-empty f16 bytes
    let values = f16_vec(&[1.5, -2.5, 0.0, 3.15]);
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let slice = bytes_to_f16_slice(&bytes).expect("Should convert");
    assert_eq!(slice.len(), 4);
}

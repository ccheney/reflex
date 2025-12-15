use super::*;
use std::path::PathBuf;

mod config_tests {
    use super::*;
    use serial_test::serial;
    use std::env;

    #[test]
    fn test_sinter_config_default() {
        let config = SinterConfig::default();
        assert_eq!(config.embedding_dim, SINTER_EMBEDDING_DIM);
        assert_eq!(config.max_seq_len, SINTER_MAX_SEQ_LEN);
        assert!(!config.testing_stub);
        assert!(config.model_path.as_os_str().is_empty());
        assert!(config.tokenizer_path.as_os_str().is_empty());
    }

    #[test]
    fn test_sinter_config_new_with_parent() {
        let config = SinterConfig::new("/models/qwen3-embedding-8b.gguf");
        assert_eq!(
            config.model_path,
            PathBuf::from("/models/qwen3-embedding-8b.gguf")
        );
        assert_eq!(
            config.tokenizer_path,
            PathBuf::from("/models/tokenizer.json")
        );
        assert_eq!(config.embedding_dim, SINTER_EMBEDDING_DIM);
        assert_eq!(config.max_seq_len, SINTER_MAX_SEQ_LEN);
        assert!(!config.testing_stub);
    }

    #[test]
    fn test_sinter_config_new_without_parent() {
        let config = SinterConfig::new("model.gguf");
        assert_eq!(config.model_path, PathBuf::from("model.gguf"));
        assert_eq!(config.tokenizer_path, PathBuf::from("tokenizer.json"));
    }

    #[test]
    fn test_sinter_config_stub() {
        let config = SinterConfig::stub();
        assert!(config.testing_stub);
        assert!(config.model_path.as_os_str().is_empty());
        assert!(config.tokenizer_path.as_os_str().is_empty());
        assert_eq!(config.embedding_dim, SINTER_EMBEDDING_DIM);
        assert_eq!(config.max_seq_len, SINTER_MAX_SEQ_LEN);
    }

    #[test]
    fn test_sinter_config_debug() {
        let config = SinterConfig::stub();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("SinterConfig"));
        assert!(debug_str.contains("testing_stub: true"));
    }

    #[test]
    fn test_sinter_config_clone() {
        let config = SinterConfig::new("/models/model.gguf");
        let cloned = config.clone();
        assert_eq!(cloned.model_path, config.model_path);
        assert_eq!(cloned.tokenizer_path, config.tokenizer_path);
        assert_eq!(cloned.embedding_dim, config.embedding_dim);
    }

    #[test]
    fn test_sinter_config_validation_with_stub() {
        let config = SinterConfig::stub();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_sinter_config_validation_empty_path_no_stub() {
        let config = SinterConfig {
            testing_stub: false,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err,
            crate::embedding::error::EmbeddingError::InvalidConfig { .. }
        ));
    }

    #[test]
    fn test_sinter_config_validation_nonexistent_path() {
        let config = SinterConfig {
            model_path: PathBuf::from("/nonexistent/path/model.gguf"),
            tokenizer_path: PathBuf::from("/nonexistent/path/tokenizer.json"),
            testing_stub: false,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err,
            crate::embedding::error::EmbeddingError::ModelNotFound { .. }
        ));
    }

    #[test]
    fn test_sinter_config_model_available_false_empty() {
        let config = SinterConfig::default();
        assert!(!config.model_available());
    }

    #[test]
    fn test_sinter_config_model_available_false_nonexistent() {
        let config = SinterConfig::new("/nonexistent/model.gguf");
        assert!(!config.model_available());
    }

    #[test]
    fn test_sinter_config_tokenizer_available_false_empty() {
        let config = SinterConfig::default();
        assert!(!config.tokenizer_available());
    }

    #[test]
    fn test_sinter_config_tokenizer_available_false_nonexistent() {
        let config = SinterConfig::new("/nonexistent/model.gguf");
        assert!(!config.tokenizer_available());
    }

    #[test]
    fn test_sinter_config_env_constants() {
        assert_eq!(SinterConfig::ENV_MODEL_PATH, "REFLEX_MODEL_PATH");
        assert_eq!(SinterConfig::ENV_TOKENIZER_PATH, "REFLEX_TOKENIZER_PATH");
    }

    #[test]
    #[serial]
    fn test_sinter_config_from_env_empty() {
        unsafe {
            env::remove_var(SinterConfig::ENV_MODEL_PATH);
            env::remove_var(SinterConfig::ENV_TOKENIZER_PATH);
        }

        let config = SinterConfig::from_env().expect("Should parse empty env");
        assert!(config.model_path.as_os_str().is_empty());
        assert!(config.tokenizer_path.as_os_str().is_empty());
    }

    #[test]
    #[serial]
    fn test_sinter_config_from_env_with_model_path() {
        unsafe {
            env::set_var(SinterConfig::ENV_MODEL_PATH, "/custom/model.gguf");
            env::remove_var(SinterConfig::ENV_TOKENIZER_PATH);
        }

        let config = SinterConfig::from_env().expect("Should parse env");
        assert_eq!(config.model_path, PathBuf::from("/custom/model.gguf"));
        assert_eq!(
            config.tokenizer_path,
            PathBuf::from("/custom/tokenizer.json")
        );

        unsafe {
            env::remove_var(SinterConfig::ENV_MODEL_PATH);
        }
    }

    #[test]
    #[serial]
    fn test_sinter_config_from_env_with_both_paths() {
        unsafe {
            env::set_var(SinterConfig::ENV_MODEL_PATH, "/model/path.gguf");
            env::set_var(SinterConfig::ENV_TOKENIZER_PATH, "/tokenizer/custom.json");
        }

        let config = SinterConfig::from_env().expect("Should parse env");
        assert_eq!(config.model_path, PathBuf::from("/model/path.gguf"));
        assert_eq!(
            config.tokenizer_path,
            PathBuf::from("/tokenizer/custom.json")
        );

        unsafe {
            env::remove_var(SinterConfig::ENV_MODEL_PATH);
            env::remove_var(SinterConfig::ENV_TOKENIZER_PATH);
        }
    }

    #[test]
    #[serial]
    fn test_sinter_config_from_env_whitespace_only() {
        unsafe {
            env::set_var(SinterConfig::ENV_MODEL_PATH, "   ");
            env::set_var(SinterConfig::ENV_TOKENIZER_PATH, "\t\n");
        }

        let config = SinterConfig::from_env().expect("Should parse env");
        assert!(config.model_path.as_os_str().is_empty());
        assert!(config.tokenizer_path.as_os_str().is_empty());

        unsafe {
            env::remove_var(SinterConfig::ENV_MODEL_PATH);
            env::remove_var(SinterConfig::ENV_TOKENIZER_PATH);
        }
    }

    #[test]
    #[serial]
    fn test_sinter_config_from_env_with_trimming() {
        unsafe {
            env::set_var(SinterConfig::ENV_MODEL_PATH, "  /path/model.gguf  ");
        }

        let config = SinterConfig::from_env().expect("Should parse env");
        assert_eq!(config.model_path, PathBuf::from("/path/model.gguf"));

        unsafe {
            env::remove_var(SinterConfig::ENV_MODEL_PATH);
        }
    }
}

mod model_tests {
    use super::super::model::{Qwen2Config, RotaryEmbedding, create_causal_mask};
    use candle_core::{Device, Tensor};

    #[test]
    fn test_qwen2_config_fields() {
        let config = Qwen2Config {
            hidden_size: 4096,
            num_layers: 36,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 12288,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_seq_len: 8192,
            vocab_size: 152064,
        };

        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_layers, 36);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.intermediate_size, 12288);
        assert!((config.rms_norm_eps - 1e-6).abs() < 1e-10);
        assert!((config.rope_theta - 1_000_000.0).abs() < 1.0);
        assert_eq!(config.max_seq_len, 8192);
        assert_eq!(config.vocab_size, 152064);
    }

    #[test]
    fn test_qwen2_config_head_dim_calculation() {
        let config = Qwen2Config {
            hidden_size: 4096,
            num_layers: 36,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 12288,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_seq_len: 8192,
            vocab_size: 152064,
        };

        assert_eq!(
            config.head_dim,
            config.hidden_size / config.num_attention_heads
        );
    }

    #[test]
    fn test_qwen2_config_gqa_ratio() {
        let config = Qwen2Config {
            hidden_size: 4096,
            num_layers: 36,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 12288,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_seq_len: 8192,
            vocab_size: 152064,
        };

        let gqa_ratio = config.num_attention_heads / config.num_kv_heads;
        assert_eq!(gqa_ratio, 4);
    }

    #[test]
    fn test_qwen2_config_debug() {
        let config = Qwen2Config {
            hidden_size: 4096,
            num_layers: 36,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 12288,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_seq_len: 8192,
            vocab_size: 152064,
        };

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("Qwen2Config"));
        assert!(debug_str.contains("hidden_size: 4096"));
        assert!(debug_str.contains("num_layers: 36"));
        assert!(debug_str.contains("num_attention_heads: 32"));
    }

    #[test]
    fn test_qwen2_config_clone() {
        let config = Qwen2Config {
            hidden_size: 4096,
            num_layers: 36,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 12288,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_seq_len: 8192,
            vocab_size: 152064,
        };

        let cloned = config.clone();
        assert_eq!(cloned.hidden_size, config.hidden_size);
        assert_eq!(cloned.num_layers, config.num_layers);
        assert_eq!(cloned.num_attention_heads, config.num_attention_heads);
        assert_eq!(cloned.num_kv_heads, config.num_kv_heads);
        assert_eq!(cloned.head_dim, config.head_dim);
        assert_eq!(cloned.intermediate_size, config.intermediate_size);
        assert!((cloned.rms_norm_eps - config.rms_norm_eps).abs() < 1e-10);
        assert!((cloned.rope_theta - config.rope_theta).abs() < 1.0);
        assert_eq!(cloned.max_seq_len, config.max_seq_len);
        assert_eq!(cloned.vocab_size, config.vocab_size);
    }

    #[test]
    fn test_qwen2_config_small_model() {
        let config = Qwen2Config {
            hidden_size: 1024,
            num_layers: 24,
            num_attention_heads: 16,
            num_kv_heads: 4,
            head_dim: 64,
            intermediate_size: 2816,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_seq_len: 4096,
            vocab_size: 151936,
        };

        assert_eq!(
            config.head_dim,
            config.hidden_size / config.num_attention_heads
        );
        assert_eq!(config.num_attention_heads / config.num_kv_heads, 4);
    }

    #[test]
    fn test_qwen2_config_7b_variant() {
        let config = Qwen2Config {
            hidden_size: 4096,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate_size: 11008,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_seq_len: 32768,
            vocab_size: 151936,
        };

        assert_eq!(config.num_attention_heads, config.num_kv_heads);
        assert_eq!(config.num_attention_heads / config.num_kv_heads, 1);
    }

    #[test]
    fn test_qwen2_config_extreme_values() {
        let config = Qwen2Config {
            hidden_size: 256,
            num_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: 1,
            head_dim: 64,
            intermediate_size: 512,
            rms_norm_eps: 1e-12,
            rope_theta: 1.0,
            max_seq_len: 1,
            vocab_size: 1,
        };

        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_layers, 1);
        assert_eq!(
            config.head_dim,
            config.hidden_size / config.num_attention_heads
        );
    }

    #[test]
    fn test_qwen2_config_large_rope_theta() {
        let config = Qwen2Config {
            hidden_size: 4096,
            num_layers: 36,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 12288,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0, // 10M for extended context
            max_seq_len: 131072,      // 128K context
            vocab_size: 152064,
        };

        assert!((config.rope_theta - 10_000_000.0).abs() < 1.0);
        assert_eq!(config.max_seq_len, 131072);
    }

    fn make_test_config(head_dim: usize) -> Qwen2Config {
        Qwen2Config {
            hidden_size: head_dim * 4,
            num_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: 4,
            head_dim,
            intermediate_size: head_dim * 8,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_seq_len: 128,
            vocab_size: 1000,
        }
    }

    #[test]
    fn test_rotary_embedding_creation() {
        let config = make_test_config(64);
        let device = Device::Cpu;

        let rope = RotaryEmbedding::new(&config, 32, &device).expect("create rope");

        assert_eq!(rope.cos.dims(), &[32, 64]);
        assert_eq!(rope.sin.dims(), &[32, 64]);
    }

    #[test]
    fn test_rotary_embedding_values_bounded() {
        let config = make_test_config(64);
        let device = Device::Cpu;

        let rope = RotaryEmbedding::new(&config, 32, &device).expect("create rope");

        let cos_values: Vec<f32> = rope.cos.flatten_all().unwrap().to_vec1().unwrap();
        let sin_values: Vec<f32> = rope.sin.flatten_all().unwrap().to_vec1().unwrap();

        for val in &cos_values {
            assert!(val.abs() <= 1.0 + 1e-6, "cos value {} out of bounds", val);
        }
        for val in &sin_values {
            assert!(val.abs() <= 1.0 + 1e-6, "sin value {} out of bounds", val);
        }
    }

    #[test]
    fn test_rotary_embedding_pythagorean_identity() {
        let config = make_test_config(64);
        let device = Device::Cpu;

        let rope = RotaryEmbedding::new(&config, 16, &device).expect("create rope");

        let cos_values: Vec<f32> = rope.cos.flatten_all().unwrap().to_vec1().unwrap();
        let sin_values: Vec<f32> = rope.sin.flatten_all().unwrap().to_vec1().unwrap();

        // cos^2 + sin^2 should equal 1 for each position
        for (c, s) in cos_values.iter().zip(sin_values.iter()) {
            let identity = c * c + s * s;
            assert!(
                (identity - 1.0).abs() < 1e-5,
                "Pythagorean identity failed: cos^2 + sin^2 = {}, expected 1.0",
                identity
            );
        }
    }

    #[test]
    fn test_rotary_embedding_different_seq_lengths() {
        let config = make_test_config(64);
        let device = Device::Cpu;

        // Create with different sequence lengths
        for max_seq in [1, 8, 32, 128] {
            let rope = RotaryEmbedding::new(&config, max_seq, &device).expect("create rope");
            assert_eq!(rope.cos.dims()[0], max_seq);
            assert_eq!(rope.sin.dims()[0], max_seq);
        }
    }

    #[test]
    fn test_rotary_embedding_apply() {
        let config = make_test_config(8); // Small head_dim for easier testing
        let device = Device::Cpu;

        let rope = RotaryEmbedding::new(&config, 16, &device).expect("create rope");

        // Create a test tensor [batch=1, heads=2, seq_len=4, head_dim=8]
        let x_data: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let x = Tensor::from_vec(x_data, (1, 2, 4, 8), &device).expect("create tensor");

        let result = rope.apply(&x, 0).expect("apply rope");

        // Output should have same shape as input
        assert_eq!(result.dims(), x.dims());

        // Values should be different from input (rotation applied)
        let result_values: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        let x_values: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();

        // At least some values should be different
        let different_count = result_values
            .iter()
            .zip(x_values.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-6)
            .count();
        assert!(
            different_count > 0,
            "RoPE should modify at least some values"
        );
    }

    #[test]
    fn test_rotary_embedding_apply_with_offset() {
        let config = make_test_config(8);
        let device = Device::Cpu;

        let rope = RotaryEmbedding::new(&config, 32, &device).expect("create rope");

        let x_data: Vec<f32> = (0..32).map(|i| i as f32 / 32.0).collect();
        let x = Tensor::from_vec(x_data.clone(), (1, 1, 4, 8), &device).expect("create tensor");

        // Apply at position 0
        let result0 = rope.apply(&x, 0).expect("apply at 0");

        // Apply at position 4
        let result4 = rope.apply(&x, 4).expect("apply at 4");

        // Results should be different due to different position embeddings
        let r0_vals: Vec<f32> = result0.flatten_all().unwrap().to_vec1().unwrap();
        let r4_vals: Vec<f32> = result4.flatten_all().unwrap().to_vec1().unwrap();

        assert_ne!(
            r0_vals, r4_vals,
            "Different positions should give different results"
        );
    }

    #[test]
    fn test_rotary_embedding_preserves_norm() {
        let config = make_test_config(8);
        let device = Device::Cpu;

        let rope = RotaryEmbedding::new(&config, 16, &device).expect("create rope");

        // Create a unit vector for testing
        let x_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let x = Tensor::from_vec(x_data, (1, 1, 1, 8), &device).expect("create tensor");

        let result = rope.apply(&x, 0).expect("apply rope");
        let result_values: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();

        // RoPE is a rotation, so norm should be preserved (approximately)
        let norm: f32 = result_values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.1,
            "RoPE should approximately preserve norm, got {}",
            norm
        );
    }

    #[test]
    fn test_causal_mask_shape() {
        let device = Device::Cpu;

        for seq_len in [1, 4, 8, 16] {
            let mask = create_causal_mask(seq_len, &device).expect("create mask");
            assert_eq!(mask.dims(), &[1, 1, seq_len, seq_len]);
        }
    }

    #[test]
    fn test_causal_mask_values() {
        let device = Device::Cpu;
        let mask = create_causal_mask(4, &device).expect("create mask");
        let values: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

        // Expected causal mask for seq_len=4:
        // Position (i, j): can attend if j <= i
        // Row 0: [0, -inf, -inf, -inf]
        // Row 1: [0, 0, -inf, -inf]
        // Row 2: [0, 0, 0, -inf]
        // Row 3: [0, 0, 0, 0]
        let expected = [
            0.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            0.0,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            0.0,
            0.0,
        ];

        for (i, (actual, exp)) in values.iter().zip(expected.iter()).enumerate() {
            if exp.is_infinite() {
                assert!(
                    actual.is_infinite() && actual.is_sign_negative(),
                    "Position {} should be -inf, got {}",
                    i,
                    actual
                );
            } else {
                assert!(
                    (actual - exp).abs() < 1e-6,
                    "Position {} should be {}, got {}",
                    i,
                    exp,
                    actual
                );
            }
        }
    }

    #[test]
    fn test_causal_mask_single_token() {
        let device = Device::Cpu;
        let mask = create_causal_mask(1, &device).expect("create mask");
        let values: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

        // Single token can attend to itself
        assert_eq!(values.len(), 1);
        assert!((values[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_causal_mask_lower_triangular() {
        let device = Device::Cpu;
        let seq_len = 8;
        let mask = create_causal_mask(seq_len, &device).expect("create mask");
        let values: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

        // Check that it's lower triangular (0s) with upper triangular -inf
        for i in 0..seq_len {
            for j in 0..seq_len {
                let idx = i * seq_len + j;
                let val = values[idx];
                if j <= i {
                    assert!(
                        (val - 0.0).abs() < 1e-6,
                        "Position ({}, {}) should be 0, got {}",
                        i,
                        j,
                        val
                    );
                } else {
                    assert!(
                        val.is_infinite() && val.is_sign_negative(),
                        "Position ({}, {}) should be -inf, got {}",
                        i,
                        j,
                        val
                    );
                }
            }
        }
    }
}

mod embedder_tests {
    use super::*;

    #[test]
    fn test_sinter_load_stub() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load in stub mode");
        assert!(embedder.is_stub());
        assert!(!embedder.has_model());
    }

    #[test]
    fn test_sinter_load_validation_fails() {
        let config = SinterConfig {
            testing_stub: false,
            model_path: PathBuf::new(),
            tokenizer_path: PathBuf::new(),
            ..Default::default()
        };
        let result = SinterEmbedder::load(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_sinter_load_model_not_available() {
        let config = SinterConfig {
            testing_stub: false,
            model_path: PathBuf::from("/nonexistent/model.gguf"),
            tokenizer_path: PathBuf::from("/nonexistent/tokenizer.json"),
            ..Default::default()
        };
        let result = SinterEmbedder::load(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_sinter_embed_stub_determinism() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let text = "Hello, world!";
        let emb1 = embedder.embed(text).expect("Should embed");
        let emb2 = embedder.embed(text).expect("Should embed");

        assert_eq!(emb1, emb2, "Same text should produce same embedding");
    }

    #[test]
    fn test_sinter_embed_stub_uniqueness() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let emb1 = embedder.embed("Hello").expect("Should embed");
        let emb2 = embedder.embed("World").expect("Should embed");

        assert_ne!(
            emb1, emb2,
            "Different text should produce different embedding"
        );
    }

    #[test]
    fn test_sinter_embed_stub_dimension() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let emb = embedder.embed("Test").expect("Should embed");
        assert_eq!(emb.len(), SINTER_EMBEDDING_DIM);
    }

    #[test]
    fn test_sinter_embed_stub_normalized() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let emb = embedder.embed("Test").expect("Should embed");

        let norm: f32 = emb
            .iter()
            .map(|x| x.to_f32() * x.to_f32())
            .sum::<f32>()
            .sqrt();

        assert!(
            (norm - 1.0).abs() < 0.01,
            "Embedding should be normalized, got norm = {}",
            norm
        );
    }

    #[test]
    fn test_sinter_embed_stub_empty_string() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let emb = embedder.embed("").expect("Should embed empty string");
        assert_eq!(emb.len(), SINTER_EMBEDDING_DIM);

        // Even empty string should produce normalized embedding
        let norm: f32 = emb
            .iter()
            .map(|x| x.to_f32() * x.to_f32())
            .sum::<f32>()
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Empty string embedding should be normalized, got norm = {}",
            norm
        );
    }

    #[test]
    fn test_sinter_embed_stub_whitespace() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let emb = embedder
            .embed("   \t\n  ")
            .expect("Should embed whitespace");
        assert_eq!(emb.len(), SINTER_EMBEDDING_DIM);
    }

    #[test]
    fn test_sinter_embed_stub_unicode() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let emb = embedder
            .embed("Hello, World, 123")
            .expect("Should embed unicode");
        assert_eq!(emb.len(), SINTER_EMBEDDING_DIM);
    }

    #[test]
    fn test_sinter_embed_stub_long_text() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let long_text = "a".repeat(10000);
        let emb = embedder.embed(&long_text).expect("Should embed long text");
        assert_eq!(emb.len(), SINTER_EMBEDDING_DIM);
    }

    #[test]
    fn test_sinter_embed_batch_stub() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let texts = vec!["Query 1", "Query 2", "Query 3"];
        let embeddings = embedder.embed_batch(&texts).expect("Should embed batch");

        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), SINTER_EMBEDDING_DIM);
        }
    }

    #[test]
    fn test_sinter_embed_batch_empty() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let embeddings = embedder.embed_batch(&[]).expect("Should handle empty");
        assert!(embeddings.is_empty());
    }

    #[test]
    fn test_sinter_embed_batch_single() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let embeddings = embedder
            .embed_batch(&["Single query"])
            .expect("Should embed single");
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), SINTER_EMBEDDING_DIM);
    }

    #[test]
    fn test_sinter_embed_batch_determinism() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let texts = vec!["Hello", "World"];
        let emb1 = embedder.embed_batch(&texts).expect("Should embed");
        let emb2 = embedder.embed_batch(&texts).expect("Should embed");

        assert_eq!(emb1, emb2, "Same batch should produce same embeddings");
    }

    #[test]
    fn test_sinter_embedding_dim_accessor() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        assert_eq!(embedder.embedding_dim(), SINTER_EMBEDDING_DIM);
    }

    #[test]
    fn test_sinter_config_accessor() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let config_ref = embedder.config();
        assert!(config_ref.testing_stub);
        assert_eq!(config_ref.embedding_dim, SINTER_EMBEDDING_DIM);
    }

    #[test]
    fn test_sinter_is_stub() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");
        assert!(embedder.is_stub());
    }

    #[test]
    fn test_sinter_has_model_false_in_stub() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");
        assert!(!embedder.has_model());
    }

    #[test]
    fn test_sinter_debug_impl_stub() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let debug_str = format!("{:?}", embedder);
        assert!(debug_str.contains("SinterEmbedder"));
        assert!(debug_str.contains("Stub"));
        assert!(debug_str.contains("embedding_dim"));
        assert!(debug_str.contains("max_seq_len"));
    }
}

mod normalization_tests {
    use super::*;

    #[test]
    fn test_normalize_standard_vector() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let emb = embedder.embed("test").expect("Should embed");

        let norm: f32 = emb
            .iter()
            .map(|x| x.to_f32() * x.to_f32())
            .sum::<f32>()
            .sqrt();

        assert!(
            (norm - 1.0).abs() < 0.01,
            "Expected norm ~1.0, got {}",
            norm
        );
    }

    #[test]
    fn test_normalize_different_inputs() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let inputs = vec!["a", "hello world", "1234567890", "!@#$%^&*()"];

        for input in inputs {
            let emb = embedder.embed(input).expect("Should embed");
            let norm: f32 = emb
                .iter()
                .map(|x| x.to_f32() * x.to_f32())
                .sum::<f32>()
                .sqrt();

            assert!(
                (norm - 1.0).abs() < 0.01,
                "Input '{}' should produce normalized embedding, got norm = {}",
                input,
                norm
            );
        }
    }

    #[test]
    fn test_f16_precision() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let emb = embedder.embed("precision test").expect("Should embed");

        // Verify all values are valid f16
        for val in &emb {
            let f32_val = val.to_f32();
            assert!(f32_val.is_finite(), "All f16 values should be finite");
            assert!(
                f32_val.abs() <= 1.1,
                "Normalized values should be <= 1.0 (with some tolerance), got {}",
                f32_val.abs()
            );
        }
    }

    #[test]
    fn test_normalize_preserves_direction() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        // Same text should produce the same normalized direction
        let emb1 = embedder.embed("direction test").expect("embed");
        let emb2 = embedder.embed("direction test").expect("embed");

        // Cosine similarity should be 1.0 (since normalized, dot product = cosine)
        let dot_product: f32 = emb1
            .iter()
            .zip(emb2.iter())
            .map(|(a, b)| a.to_f32() * b.to_f32())
            .sum();

        assert!(
            (dot_product - 1.0).abs() < 0.01,
            "Same input should have cosine similarity ~1.0, got {}",
            dot_product
        );
    }

    #[test]
    fn test_f16_range_bounds() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        // Test various inputs to ensure f16 values stay within valid range
        let test_inputs = vec![
            "short",
            "a moderately long sentence with several words",
            "UPPERCASE TEXT",
            "mixed Case Text With Numbers 12345",
            "special!@#$%chars",
            "\t\n whitespace \r\n",
        ];

        for input in test_inputs {
            let emb = embedder.embed(input).expect("embed");

            for (i, val) in emb.iter().enumerate() {
                let f32_val = val.to_f32();
                assert!(
                    f32_val.is_finite(),
                    "Index {} should be finite for input '{}', got {:?}",
                    i,
                    input,
                    val
                );
                assert!(
                    !f32_val.is_nan(),
                    "Index {} should not be NaN for input '{}'",
                    i,
                    input
                );
            }
        }
    }

    #[test]
    fn test_embedding_values_in_unit_ball() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let emb = embedder.embed("unit ball test").expect("embed");

        // After L2 normalization, all component magnitudes should be <= 1
        for val in &emb {
            let f32_val = val.to_f32();
            assert!(
                f32_val.abs() <= 1.0 + 0.001, // Small tolerance for f16 precision
                "Normalized component should have magnitude <= 1.0, got {}",
                f32_val.abs()
            );
        }
    }
}

mod stub_hash_tests {
    use super::*;

    #[test]
    fn test_stub_hash_determinism() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        // Same input should always produce same output
        let text = "deterministic input";
        let emb1 = embedder.embed(text).expect("embed");
        let emb2 = embedder.embed(text).expect("embed");

        assert_eq!(emb1, emb2);
    }

    #[test]
    fn test_stub_hash_uniqueness() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let texts = vec![
            "text1", "text2", "text3", "text4", "text5", "a", "ab", "abc", "abcd",
        ];

        let embeddings: Vec<_> = texts
            .iter()
            .map(|t| embedder.embed(t).expect("embed"))
            .collect();

        // All embeddings should be unique
        for i in 0..embeddings.len() {
            for j in (i + 1)..embeddings.len() {
                assert_ne!(
                    embeddings[i], embeddings[j],
                    "Embeddings for '{}' and '{}' should be different",
                    texts[i], texts[j]
                );
            }
        }
    }

    #[test]
    fn test_stub_hash_distribution() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let emb = embedder.embed("distribution test").expect("embed");

        // Check that values are distributed (not all zeros or same value)
        let mut unique_vals = std::collections::HashSet::new();
        for val in &emb {
            unique_vals.insert(val.to_bits());
        }

        // Should have many unique values (at least 80% of dimension)
        // The PRNG may produce some collisions, especially after f16 conversion
        assert!(
            unique_vals.len() > SINTER_EMBEDDING_DIM * 8 / 10,
            "Stub embedding should have diverse values, got {} unique out of {}",
            unique_vals.len(),
            SINTER_EMBEDDING_DIM
        );
    }

    #[test]
    fn test_stub_prng_state_progression() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        // Each call to embed should use independent state
        let emb1 = embedder.embed("state1").expect("embed");
        let emb2 = embedder.embed("state2").expect("embed");

        // But repeated calls should be deterministic
        let emb1_again = embedder.embed("state1").expect("embed");

        assert_eq!(emb1, emb1_again, "Same input should give same output");
        assert_ne!(emb1, emb2, "Different inputs should give different outputs");
    }
}

mod error_tests {
    use super::*;
    use crate::embedding::error::EmbeddingError;
    use std::fs::File;
    use tempfile::TempDir;

    #[test]
    fn test_load_with_invalid_config() {
        let config = SinterConfig {
            model_path: PathBuf::new(),
            tokenizer_path: PathBuf::new(),
            testing_stub: false,
            ..Default::default()
        };

        let result = SinterEmbedder::load(config);
        assert!(result.is_err());

        match result.unwrap_err() {
            EmbeddingError::InvalidConfig { reason } => {
                assert!(reason.contains("model_path"));
            }
            other => panic!("Expected InvalidConfig error, got {:?}", other),
        }
    }

    #[test]
    fn test_load_with_nonexistent_model() {
        let config = SinterConfig {
            model_path: PathBuf::from("/definitely/nonexistent/path/model.gguf"),
            tokenizer_path: PathBuf::from("/definitely/nonexistent/path/tokenizer.json"),
            testing_stub: false,
            ..Default::default()
        };

        let result = SinterEmbedder::load(config);
        assert!(result.is_err());

        match result.unwrap_err() {
            EmbeddingError::ModelNotFound { path } => {
                assert!(path.to_string_lossy().contains("nonexistent"));
            }
            other => panic!("Expected ModelNotFound error, got {:?}", other),
        }
    }

    #[test]
    fn test_load_model_exists_but_tokenizer_missing() {
        // Tests the path where model exists but tokenizer doesn't
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.gguf");
        let tokenizer_path = temp_dir.path().join("tokenizer.json");

        // Create only the model file, not the tokenizer
        File::create(&model_path).expect("create model file");

        let config = SinterConfig {
            model_path: model_path.clone(),
            tokenizer_path: tokenizer_path.clone(),
            testing_stub: false,
            ..Default::default()
        };

        // model_available() returns true, tokenizer_available() returns false
        assert!(config.model_available());
        assert!(!config.tokenizer_available());

        let result = SinterEmbedder::load(config);
        assert!(result.is_err());

        match result.unwrap_err() {
            EmbeddingError::ModelNotFound { path } => {
                assert_eq!(path, model_path);
            }
            other => panic!("Expected ModelNotFound error, got {:?}", other),
        }
    }

    #[test]
    fn test_load_tokenizer_exists_but_model_missing() {
        // Tests the path where tokenizer exists but model doesn't
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.gguf");
        let tokenizer_path = temp_dir.path().join("tokenizer.json");

        // Create only the tokenizer file, not the model
        File::create(&tokenizer_path).expect("create tokenizer file");

        let config = SinterConfig {
            model_path: model_path.clone(),
            tokenizer_path: tokenizer_path.clone(),
            testing_stub: false,
            ..Default::default()
        };

        // model_available() returns false, tokenizer_available() returns true
        assert!(!config.model_available());
        assert!(config.tokenizer_available());

        let result = SinterEmbedder::load(config);
        assert!(result.is_err());

        match result.unwrap_err() {
            EmbeddingError::ModelNotFound { path } => {
                assert_eq!(path, model_path);
            }
            other => panic!("Expected ModelNotFound error, got {:?}", other),
        }
    }

    #[test]
    fn test_load_both_exist_but_invalid_model_format() {
        // Tests the path where both files exist but model is invalid
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.gguf");
        let tokenizer_path = temp_dir.path().join("tokenizer.json");

        // Create both files with invalid content
        std::fs::write(&model_path, b"not a valid gguf file").expect("write model");
        std::fs::write(&tokenizer_path, r#"{"not": "valid tokenizer"}"#).expect("write tokenizer");

        let config = SinterConfig {
            model_path: model_path.clone(),
            tokenizer_path: tokenizer_path.clone(),
            testing_stub: false,
            ..Default::default()
        };

        assert!(config.model_available());
        assert!(config.tokenizer_available());

        let result = SinterEmbedder::load(config);
        assert!(result.is_err());

        // Should fail during model loading (either tokenization or GGUF parsing)
        match result.unwrap_err() {
            EmbeddingError::TokenizationFailed { reason } => {
                assert!(!reason.is_empty());
            }
            EmbeddingError::ModelLoadFailed { reason } => {
                assert!(!reason.is_empty());
            }
            other => panic!(
                "Expected TokenizationFailed or ModelLoadFailed error, got {:?}",
                other
            ),
        }
    }
}

mod constants_tests {
    use super::*;

    #[test]
    fn test_sinter_embedding_dim_constant() {
        assert_eq!(SINTER_EMBEDDING_DIM, 1536);
    }

    #[test]
    fn test_sinter_max_seq_len_constant() {
        assert_eq!(SINTER_MAX_SEQ_LEN, 8192);
    }

    #[test]
    fn test_constants_match_defaults() {
        let config = SinterConfig::default();
        assert_eq!(config.embedding_dim, SINTER_EMBEDDING_DIM);
        assert_eq!(config.max_seq_len, SINTER_MAX_SEQ_LEN);
    }
}

mod config_file_tests {
    use super::*;
    use std::fs::File;
    use tempfile::TempDir;

    #[test]
    fn test_model_available_with_real_file() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("test_model.gguf");

        // Create a dummy file
        File::create(&model_path).expect("create file");

        let config = SinterConfig {
            model_path: model_path.clone(),
            tokenizer_path: PathBuf::new(),
            ..Default::default()
        };

        assert!(config.model_available());
    }

    #[test]
    fn test_tokenizer_available_with_real_file() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let tokenizer_path = temp_dir.path().join("tokenizer.json");

        // Create a dummy file
        File::create(&tokenizer_path).expect("create file");

        let config = SinterConfig {
            model_path: PathBuf::new(),
            tokenizer_path: tokenizer_path.clone(),
            ..Default::default()
        };

        assert!(config.tokenizer_available());
    }

    #[test]
    fn test_both_available_with_real_files() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.gguf");
        let tokenizer_path = temp_dir.path().join("tokenizer.json");

        File::create(&model_path).expect("create model file");
        File::create(&tokenizer_path).expect("create tokenizer file");

        let config = SinterConfig {
            model_path: model_path.clone(),
            tokenizer_path: tokenizer_path.clone(),
            ..Default::default()
        };

        assert!(config.model_available());
        assert!(config.tokenizer_available());
    }

    #[test]
    fn test_validation_passes_with_real_model_file() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.gguf");

        File::create(&model_path).expect("create file");

        let config = SinterConfig {
            model_path: model_path.clone(),
            tokenizer_path: temp_dir.path().join("tokenizer.json"),
            testing_stub: false,
            ..Default::default()
        };

        // Validation should pass (file exists)
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_new_derives_tokenizer_from_model_path() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.gguf");

        let config = SinterConfig::new(&model_path);

        // Tokenizer should be in same directory
        assert_eq!(
            config.tokenizer_path,
            temp_dir.path().join("tokenizer.json")
        );
    }

    #[test]
    fn test_config_with_nested_model_path() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let nested_dir = temp_dir.path().join("models").join("qwen3");
        std::fs::create_dir_all(&nested_dir).expect("create dirs");

        let model_path = nested_dir.join("embedding.gguf");
        let config = SinterConfig::new(&model_path);

        assert_eq!(config.tokenizer_path, nested_dir.join("tokenizer.json"));
    }

    #[test]
    fn test_config_custom_embedding_dim() {
        let config = SinterConfig {
            embedding_dim: 768,
            ..Default::default()
        };
        assert_eq!(config.embedding_dim, 768);
    }

    #[test]
    fn test_config_custom_max_seq_len() {
        let config = SinterConfig {
            max_seq_len: 2048,
            ..Default::default()
        };
        assert_eq!(config.max_seq_len, 2048);
    }
}

mod custom_dimension_tests {
    use super::*;

    #[test]
    fn test_stub_with_custom_embedding_dim() {
        let config = SinterConfig {
            testing_stub: true,
            embedding_dim: 768,
            ..Default::default()
        };
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let emb = embedder.embed("test").expect("embed");
        assert_eq!(emb.len(), 768);
        assert_eq!(embedder.embedding_dim(), 768);
    }

    #[test]
    fn test_stub_with_small_embedding_dim() {
        let config = SinterConfig {
            testing_stub: true,
            embedding_dim: 64,
            ..Default::default()
        };
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let emb = embedder.embed("small dim test").expect("embed");
        assert_eq!(emb.len(), 64);

        // Should still be normalized
        let norm: f32 = emb
            .iter()
            .map(|x| x.to_f32() * x.to_f32())
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_stub_with_large_embedding_dim() {
        let config = SinterConfig {
            testing_stub: true,
            embedding_dim: 4096,
            ..Default::default()
        };
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let emb = embedder.embed("large dim test").expect("embed");
        assert_eq!(emb.len(), 4096);

        // Should still be normalized
        let norm: f32 = emb
            .iter()
            .map(|x| x.to_f32() * x.to_f32())
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_with_custom_dim() {
        let config = SinterConfig {
            testing_stub: true,
            embedding_dim: 512,
            ..Default::default()
        };
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let texts = vec!["query 1", "query 2"];
        let embeddings = embedder.embed_batch(&texts).expect("embed batch");

        assert_eq!(embeddings.len(), 2);
        for emb in &embeddings {
            assert_eq!(emb.len(), 512);
        }
    }
}

mod batch_edge_cases {
    use super::*;

    #[test]
    fn test_batch_large_number_of_texts() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let texts: Vec<&str> = (0..100).map(|_| "test text").collect();
        let embeddings = embedder.embed_batch(&texts).expect("embed batch");

        assert_eq!(embeddings.len(), 100);
    }

    #[test]
    fn test_batch_mixed_lengths() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let texts = vec![
            "a",
            "medium length",
            "a very long text that contains many words and should still work correctly",
            "",
            "   ",
        ];
        let embeddings = embedder.embed_batch(&texts).expect("embed batch");

        assert_eq!(embeddings.len(), 5);
        for emb in &embeddings {
            assert_eq!(emb.len(), SINTER_EMBEDDING_DIM);
        }
    }

    #[test]
    fn test_batch_all_same_text() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let texts = vec!["same"; 5];
        let embeddings = embedder.embed_batch(&texts).expect("embed batch");

        // All embeddings should be identical
        for i in 1..embeddings.len() {
            assert_eq!(embeddings[0], embeddings[i]);
        }
    }

    #[test]
    fn test_batch_all_unique() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let texts = vec!["text1", "text2", "text3", "text4", "text5"];
        let embeddings = embedder.embed_batch(&texts).expect("embed batch");

        // All embeddings should be unique
        for i in 0..embeddings.len() {
            for j in (i + 1)..embeddings.len() {
                assert_ne!(embeddings[i], embeddings[j]);
            }
        }
    }

    #[test]
    fn test_batch_vs_single_consistency() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let texts = vec!["hello", "world", "test"];

        // Embed as batch
        let batch_embeddings = embedder.embed_batch(&texts).expect("embed batch");

        // Embed individually
        let individual_embeddings: Vec<_> = texts
            .iter()
            .map(|t| embedder.embed(t).expect("embed"))
            .collect();

        // Results should be identical
        assert_eq!(batch_embeddings.len(), individual_embeddings.len());
        for (batch, individual) in batch_embeddings.iter().zip(individual_embeddings.iter()) {
            assert_eq!(batch, individual, "Batch and individual should match");
        }
    }
}

mod additional_edge_cases {
    use super::*;

    #[test]
    fn test_embed_very_short_text() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        // Single character
        let emb = embedder.embed("x").expect("embed");
        assert_eq!(emb.len(), SINTER_EMBEDDING_DIM);

        // Verify normalized
        let norm: f32 = emb
            .iter()
            .map(|x| x.to_f32() * x.to_f32())
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_embed_numeric_text() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let emb = embedder.embed("12345").expect("embed");
        assert_eq!(emb.len(), SINTER_EMBEDDING_DIM);
    }

    #[test]
    fn test_embed_special_characters() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let special_texts = vec![
            "!@#$%^&*()",
            "<script>alert('xss')</script>",
            "SELECT * FROM users",
            "{\"json\": \"value\"}",
            "path/to/file.txt",
            "user@email.com",
            "https://example.com",
        ];

        for text in special_texts {
            let emb = embedder
                .embed(text)
                .unwrap_or_else(|_| panic!("embed '{}'", text));
            assert_eq!(emb.len(), SINTER_EMBEDDING_DIM);
        }
    }

    #[test]
    fn test_embed_multiline_text() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let multiline = "Line 1\nLine 2\nLine 3";
        let emb = embedder.embed(multiline).expect("embed");
        assert_eq!(emb.len(), SINTER_EMBEDDING_DIM);
    }

    #[test]
    fn test_embed_repeated_character() {
        let config = SinterConfig::stub();
        let embedder = SinterEmbedder::load(config).expect("Should load");

        let repeated = "a".repeat(1000);
        let emb = embedder.embed(&repeated).expect("embed");
        assert_eq!(emb.len(), SINTER_EMBEDDING_DIM);
    }

    #[test]
    fn test_stub_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let config = SinterConfig::stub();
        let embedder = Arc::new(SinterEmbedder::load(config).expect("Should load"));

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let embedder = Arc::clone(&embedder);
                thread::spawn(move || {
                    let text = format!("thread {} text", i);
                    let emb = embedder.embed(&text).expect("embed");
                    assert_eq!(emb.len(), SINTER_EMBEDDING_DIM);
                    emb
                })
            })
            .collect();

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All results should be unique
        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                assert_ne!(results[i], results[j]);
            }
        }
    }

    #[test]
    fn test_config_accessors_consistency() {
        let config = SinterConfig {
            testing_stub: true,
            embedding_dim: 512,
            max_seq_len: 4096,
            ..Default::default()
        };
        let embedder = SinterEmbedder::load(config).expect("Should load");

        assert_eq!(embedder.embedding_dim(), 512);
        assert_eq!(embedder.config().embedding_dim, 512);
        assert_eq!(embedder.config().max_seq_len, 4096);
        assert!(embedder.config().testing_stub);
    }
}

/// Integration test for full transformer inference.
/// Run with: cargo test --lib sinter -- --ignored
#[test]
#[ignore]
fn test_sinter_transformer_embedding_dimension() {
    let model_path = std::env::var("QWEN3_EMBEDDING_MODEL_PATH")
        .unwrap_or_else(|_| "/models/qwen3-embedding-8b-q4_k_m.gguf".to_string());
    let tokenizer_path = std::env::var("QWEN3_TOKENIZER_PATH")
        .unwrap_or_else(|_| "/models/tokenizer.json".to_string());

    let config = SinterConfig {
        model_path: PathBuf::from(model_path),
        tokenizer_path: PathBuf::from(tokenizer_path),
        embedding_dim: SINTER_EMBEDDING_DIM,
        max_seq_len: SINTER_MAX_SEQ_LEN,
        testing_stub: false,
    };

    let embedder = SinterEmbedder::load(config).expect("Should load model");
    assert!(embedder.has_model());

    let embedding = embedder.embed("Test sentence").expect("Should embed");
    assert_eq!(embedding.len(), SINTER_EMBEDDING_DIM);
}

#[test]
#[ignore]
fn test_sinter_transformer_normalized_output() {
    let model_path = std::env::var("QWEN3_EMBEDDING_MODEL_PATH")
        .unwrap_or_else(|_| "/models/qwen3-embedding-8b-q4_k_m.gguf".to_string());
    let tokenizer_path = std::env::var("QWEN3_TOKENIZER_PATH")
        .unwrap_or_else(|_| "/models/tokenizer.json".to_string());

    let config = SinterConfig {
        model_path: PathBuf::from(model_path),
        tokenizer_path: PathBuf::from(tokenizer_path),
        embedding_dim: SINTER_EMBEDDING_DIM,
        max_seq_len: SINTER_MAX_SEQ_LEN,
        testing_stub: false,
    };

    let embedder = SinterEmbedder::load(config).expect("Should load model");
    let embedding = embedder.embed("Test sentence").expect("Should embed");

    let norm: f32 = embedding
        .iter()
        .map(|x| x.to_f32() * x.to_f32())
        .sum::<f32>()
        .sqrt();

    assert!(
        (norm - 1.0).abs() < 0.01,
        "Embedding should be L2 normalized, got norm = {}",
        norm
    );
}

#[test]
#[ignore]
fn test_sinter_transformer_determinism() {
    let model_path = std::env::var("QWEN3_EMBEDDING_MODEL_PATH")
        .unwrap_or_else(|_| "/models/qwen3-embedding-8b-q4_k_m.gguf".to_string());
    let tokenizer_path = std::env::var("QWEN3_TOKENIZER_PATH")
        .unwrap_or_else(|_| "/models/tokenizer.json".to_string());

    let config = SinterConfig {
        model_path: PathBuf::from(model_path),
        tokenizer_path: PathBuf::from(tokenizer_path),
        embedding_dim: SINTER_EMBEDDING_DIM,
        max_seq_len: SINTER_MAX_SEQ_LEN,
        testing_stub: false,
    };

    let embedder = SinterEmbedder::load(config).expect("Should load model");

    let text = "The quick brown fox jumps over the lazy dog";
    let emb1 = embedder.embed(text).expect("Should embed");
    let emb2 = embedder.embed(text).expect("Should embed");

    assert_eq!(emb1, emb2, "Same text should produce identical embeddings");
}

#[test]
#[ignore]
fn test_sinter_transformer_semantic_similarity() {
    let model_path = std::env::var("QWEN3_EMBEDDING_MODEL_PATH")
        .unwrap_or_else(|_| "/models/qwen3-embedding-8b-q4_k_m.gguf".to_string());
    let tokenizer_path = std::env::var("QWEN3_TOKENIZER_PATH")
        .unwrap_or_else(|_| "/models/tokenizer.json".to_string());

    let config = SinterConfig {
        model_path: PathBuf::from(model_path),
        tokenizer_path: PathBuf::from(tokenizer_path),
        embedding_dim: SINTER_EMBEDDING_DIM,
        max_seq_len: SINTER_MAX_SEQ_LEN,
        testing_stub: false,
    };

    let embedder = SinterEmbedder::load(config).expect("Should load model");

    let emb1 = embedder.embed("The cat sat on the mat").expect("embed");
    let emb2 = embedder.embed("A feline rested on the rug").expect("embed");
    let emb3 = embedder
        .embed("Quantum physics explains wave functions")
        .expect("embed");

    // Cosine similarity (embeddings are normalized)
    let sim_12: f32 = emb1
        .iter()
        .zip(emb2.iter())
        .map(|(a, b)| a.to_f32() * b.to_f32())
        .sum();

    let sim_13: f32 = emb1
        .iter()
        .zip(emb3.iter())
        .map(|(a, b)| a.to_f32() * b.to_f32())
        .sum();

    assert!(
        sim_12 > sim_13,
        "Semantically similar texts should have higher similarity: sim(cat,feline)={} vs sim(cat,quantum)={}",
        sim_12,
        sim_13
    );
}

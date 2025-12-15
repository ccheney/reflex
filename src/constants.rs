//! Cross-cutting, shared constants.
//!
//! Prefer deriving secondary constants (e.g. byte sizes) from primary ones to avoid drift.
//!
//! # Dimension Invariants
//!
//! The embedding dimension values are treated as compile-time invariants across many modules
//! (embedding, vectordb, cache, scoring). If you need runtime-configurable dimensions:
//!
//! 1. Use [`DimConfig`] to pass dimensions through initialization
//! 2. Use [`validate_embedding_dim`] at module boundaries to catch mismatches early
//! 3. The compile-time constants remain as defaults and for static size calculations

pub const DEFAULT_EMBEDDING_DIM: usize = 1536;
pub const EMBEDDING_F16_BYTES: usize = DEFAULT_EMBEDDING_DIM * 2;
pub const EMBEDDING_F32_BYTES: usize = DEFAULT_EMBEDDING_DIM * 4;
pub const EMBEDDING_BQ_BYTES: usize = DEFAULT_EMBEDDING_DIM / 8;

pub const DEFAULT_VECTOR_SIZE_U64: u64 = DEFAULT_EMBEDDING_DIM as u64;

pub const DEFAULT_VERIFICATION_THRESHOLD: f32 = 0.70;

pub const DEFAULT_MAX_SEQ_LEN: usize = 8192;

/// Runtime dimension configuration for modules that support dynamic embedding sizes.
///
/// Use this when initializing modules that need to agree on vector dimensions at runtime.
/// The [`validate`](DimConfig::validate) method ensures consistency with the default constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimConfig {
    /// The embedding vector dimension (number of floats).
    pub embedding_dim: usize,
}

impl Default for DimConfig {
    fn default() -> Self {
        Self {
            embedding_dim: DEFAULT_EMBEDDING_DIM,
        }
    }
}

impl DimConfig {
    /// Creates a new dimension configuration with the specified embedding dimension.
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim }
    }

    /// Validates that this configuration is internally consistent and reasonable.
    ///
    /// Returns an error if:
    /// - `embedding_dim` is zero
    /// - `embedding_dim` is not divisible by 8 (required for binary quantization)
    pub fn validate(&self) -> Result<(), DimValidationError> {
        if self.embedding_dim == 0 {
            return Err(DimValidationError::ZeroDimension);
        }
        if !self.embedding_dim.is_multiple_of(8) {
            return Err(DimValidationError::NotDivisibleBy8 {
                dim: self.embedding_dim,
            });
        }
        Ok(())
    }

    /// Returns the number of bytes needed for F16 representation.
    pub fn f16_bytes(&self) -> usize {
        self.embedding_dim * 2
    }

    /// Returns the number of bytes needed for F32 representation.
    pub fn f32_bytes(&self) -> usize {
        self.embedding_dim * 4
    }

    /// Returns the number of bytes needed for binary quantization.
    pub fn bq_bytes(&self) -> usize {
        self.embedding_dim / 8
    }
}

/// Error returned when dimension validation fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimValidationError {
    /// Embedding dimension cannot be zero.
    ZeroDimension,
    /// Embedding dimension must be divisible by 8 for binary quantization.
    NotDivisibleBy8 { dim: usize },
    /// Runtime dimension does not match expected dimension.
    DimensionMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for DimValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "embedding dimension cannot be zero"),
            Self::NotDivisibleBy8 { dim } => {
                write!(
                    f,
                    "embedding dimension {} is not divisible by 8 (required for BQ)",
                    dim
                )
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "dimension mismatch: expected {}, got {}",
                    expected, actual
                )
            }
        }
    }
}

impl std::error::Error for DimValidationError {}

/// Validates that a runtime embedding dimension matches the expected dimension.
///
/// Use this at module boundaries to catch dimension mismatches early, rather than
/// encountering silent data corruption or panics deep in the processing pipeline.
///
/// # Example
///
/// ```
/// use reflex::constants::{validate_embedding_dim, DEFAULT_EMBEDDING_DIM};
///
/// // At module boundary, validate incoming dimension matches expected
/// let embedder_dim = 1536;
/// validate_embedding_dim(embedder_dim, DEFAULT_EMBEDDING_DIM).unwrap();
/// ```
pub fn validate_embedding_dim(actual: usize, expected: usize) -> Result<(), DimValidationError> {
    if actual != expected {
        return Err(DimValidationError::DimensionMismatch { expected, actual });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dim_config_default() {
        let config = DimConfig::default();
        assert_eq!(config.embedding_dim, DEFAULT_EMBEDDING_DIM);
    }

    #[test]
    fn test_dim_config_validate_success() {
        let config = DimConfig::new(1536);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_dim_config_validate_zero() {
        let config = DimConfig::new(0);
        assert_eq!(config.validate(), Err(DimValidationError::ZeroDimension));
    }

    #[test]
    fn test_dim_config_validate_not_divisible_by_8() {
        let config = DimConfig::new(1537);
        assert_eq!(
            config.validate(),
            Err(DimValidationError::NotDivisibleBy8 { dim: 1537 })
        );
    }

    #[test]
    fn test_dim_config_byte_calculations() {
        let config = DimConfig::new(1536);
        assert_eq!(config.f16_bytes(), EMBEDDING_F16_BYTES);
        assert_eq!(config.f32_bytes(), EMBEDDING_F32_BYTES);
        assert_eq!(config.bq_bytes(), EMBEDDING_BQ_BYTES);
    }

    #[test]
    fn test_validate_embedding_dim_match() {
        assert!(validate_embedding_dim(1536, 1536).is_ok());
    }

    #[test]
    fn test_validate_embedding_dim_mismatch() {
        assert_eq!(
            validate_embedding_dim(768, 1536),
            Err(DimValidationError::DimensionMismatch {
                expected: 1536,
                actual: 768
            })
        );
    }

    #[test]
    fn test_error_display() {
        let err = DimValidationError::ZeroDimension;
        assert_eq!(err.to_string(), "embedding dimension cannot be zero");

        let err = DimValidationError::NotDivisibleBy8 { dim: 1537 };
        assert!(err.to_string().contains("1537"));
        assert!(err.to_string().contains("divisible by 8"));

        let err = DimValidationError::DimensionMismatch {
            expected: 1536,
            actual: 768,
        };
        assert!(err.to_string().contains("1536"));
        assert!(err.to_string().contains("768"));
    }
}

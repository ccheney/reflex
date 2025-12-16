use candle_core::Device;
use tracing::warn;

#[cfg(any(feature = "metal", feature = "cuda"))]
use tracing::info;

#[cfg(not(any(feature = "metal", feature = "cuda")))]
use tracing::debug;

use super::error::EmbeddingError;

/// Selects the compute device based on enabled features (falls back to CPU).
pub fn select_device() -> Result<Device, EmbeddingError> {
    #[cfg(any(feature = "metal", feature = "cuda"))]
    let mut failures: Vec<String> = Vec::new();

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    let failures: Vec<String> = Vec::new();

    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(device) => {
                info!("Using Metal GPU acceleration");
                return Ok(device);
            }
            Err(e) => {
                let msg = e.to_string();
                if cfg!(feature = "cuda") {
                    warn!(error = %msg, "Metal device unavailable, trying CUDA");
                } else {
                    warn!(error = %msg, "Metal device unavailable");
                }
                failures.push(format!("metal failed: {msg}"));
            }
        }
    }

    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(device) => {
                info!("Using CUDA GPU acceleration");
                return Ok(device);
            }
            Err(e) => {
                let msg = e.to_string();
                warn!(error = %msg, "CUDA device unavailable");
                failures.push(format!("cuda failed: {msg}"));
            }
        }
    }

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        debug!("No GPU features enabled");
    }

    let reason = if !cfg!(any(feature = "metal", feature = "cuda")) {
        "no GPU backend compiled".to_string()
    } else if failures.is_empty() {
        "no GPU device available".to_string()
    } else {
        failures.join("; ")
    };

    warn!(reason = %reason, "Falling back to CPU device");
    Ok(Device::Cpu)
}

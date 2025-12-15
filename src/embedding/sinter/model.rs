//! Qwen2 model implementation for embeddings.
//!
//! This module provides a custom Qwen2 forward pass that returns hidden states
//! (not logits) for use with embedding models like Qwen3-Embedding-8B.

use std::sync::Arc;

use candle_core::quantized::{QMatMul, gguf_file};
use candle_core::{D, Device, Module, Result, Tensor};
use candle_nn::RmsNorm;

/// Qwen2 model configuration extracted from GGUF metadata.
#[derive(Debug, Clone)]
#[allow(dead_code)] // intermediate_size and vocab_size are parsed but not used in forward pass
pub struct Qwen2Config {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_seq_len: usize,
    pub vocab_size: usize,
}

impl Qwen2Config {
    /// Extract configuration from GGUF metadata.
    pub fn from_gguf(content: &gguf_file::Content) -> Result<Self> {
        let get_u64 = |key: &str, default: u64| -> u64 {
            content
                .metadata
                .get(key)
                .and_then(|v| v.to_u64().ok())
                .unwrap_or(default)
        };

        let get_f64 = |key: &str, default: f64| -> f64 {
            content
                .metadata
                .get(key)
                .and_then(|v| v.to_f64().ok())
                .unwrap_or(default)
        };

        let hidden_size = get_u64("qwen2.embedding_length", 4096) as usize;
        let num_layers = get_u64("qwen2.block_count", 36) as usize;
        let num_attention_heads = get_u64("qwen2.attention.head_count", 32) as usize;
        let num_kv_heads = get_u64("qwen2.attention.head_count_kv", 8) as usize;
        let intermediate_size = get_u64("qwen2.feed_forward_length", 12288) as usize;
        let rms_norm_eps = get_f64("qwen2.attention.layer_norm_rms_epsilon", 1e-6);
        let rope_theta = get_f64("qwen2.rope.freq_base", 1_000_000.0);
        let max_seq_len = get_u64("qwen2.context_length", 32768) as usize;
        let vocab_size = get_u64("qwen2.vocab_size", 152064) as usize;

        let head_dim = hidden_size / num_attention_heads;

        Ok(Self {
            hidden_size,
            num_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            rms_norm_eps,
            rope_theta,
            max_seq_len,
            vocab_size,
        })
    }
}

/// Precomputed rotary position embeddings.
pub(crate) struct RotaryEmbedding {
    pub(crate) cos: Tensor,
    pub(crate) sin: Tensor,
}

impl RotaryEmbedding {
    pub(crate) fn new(config: &Qwen2Config, max_seq_len: usize, device: &Device) -> Result<Self> {
        let half_dim = config.head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (config.rope_theta as f32).powf((2 * i) as f32 / config.head_dim as f32))
            .collect();

        let inv_freq = Tensor::new(inv_freq, device)?;
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions, device)?;

        // Outer product: [max_seq_len] x [half_dim] -> [max_seq_len, half_dim]
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;

        // Duplicate for full head_dim: [max_seq_len, head_dim]
        let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;

        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        Ok(Self { cos, sin })
    }

    pub(crate) fn apply(&self, x: &Tensor, seq_start: usize) -> Result<Tensor> {
        let (_batch, _heads, seq_len, head_dim) = x.dims4()?;

        // Get cos/sin for this sequence range
        let cos = self.cos.narrow(0, seq_start, seq_len)?;
        let sin = self.sin.narrow(0, seq_start, seq_len)?;

        // Reshape for broadcasting: [1, 1, seq_len, head_dim]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        // Rotate half: split x into first and second half
        let half = head_dim / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;

        // Rotated version: [-x2, x1]
        let x_rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;

        // Apply rotation: x * cos + x_rotated * sin
        let result = (x.broadcast_mul(&cos)? + x_rotated.broadcast_mul(&sin)?)?;
        Ok(result)
    }
}

/// Single transformer layer with attention and MLP.
struct Qwen2Layer {
    attn_q: QMatMul,
    attn_k: QMatMul,
    attn_v: QMatMul,
    attn_o: QMatMul,
    attn_q_bias: Option<Tensor>,
    attn_k_bias: Option<Tensor>,
    attn_v_bias: Option<Tensor>,
    attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    ffn_gate: QMatMul,
    ffn_up: QMatMul,
    ffn_down: QMatMul,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Qwen2Layer {
    fn load(
        content: &gguf_file::Content,
        file: &mut std::fs::File,
        device: &Device,
        config: &Qwen2Config,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        // Attention weights
        let attn_q = Self::load_qmatmul(content, file, &format!("{prefix}.attn_q.weight"), device)?;
        let attn_k = Self::load_qmatmul(content, file, &format!("{prefix}.attn_k.weight"), device)?;
        let attn_v = Self::load_qmatmul(content, file, &format!("{prefix}.attn_v.weight"), device)?;
        let attn_o = Self::load_qmatmul(
            content,
            file,
            &format!("{prefix}.attn_output.weight"),
            device,
        )?;

        // Attention biases (Qwen2 has them)
        let attn_q_bias = Self::load_bias(content, file, &format!("{prefix}.attn_q.bias"), device);
        let attn_k_bias = Self::load_bias(content, file, &format!("{prefix}.attn_k.bias"), device);
        let attn_v_bias = Self::load_bias(content, file, &format!("{prefix}.attn_v.bias"), device);

        // Layer norms
        let attn_norm = Self::load_rms_norm(
            content,
            file,
            &format!("{prefix}.attn_norm.weight"),
            device,
            config.rms_norm_eps,
        )?;
        let ffn_norm = Self::load_rms_norm(
            content,
            file,
            &format!("{prefix}.ffn_norm.weight"),
            device,
            config.rms_norm_eps,
        )?;

        // MLP (SwiGLU)
        let ffn_gate =
            Self::load_qmatmul(content, file, &format!("{prefix}.ffn_gate.weight"), device)?;
        let ffn_up = Self::load_qmatmul(content, file, &format!("{prefix}.ffn_up.weight"), device)?;
        let ffn_down =
            Self::load_qmatmul(content, file, &format!("{prefix}.ffn_down.weight"), device)?;

        Ok(Self {
            attn_q,
            attn_k,
            attn_v,
            attn_o,
            attn_q_bias,
            attn_k_bias,
            attn_v_bias,
            attn_norm,
            ffn_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
        })
    }

    fn load_qmatmul(
        content: &gguf_file::Content,
        file: &mut std::fs::File,
        name: &str,
        device: &Device,
    ) -> Result<QMatMul> {
        let qtensor = content.tensor(file, name, device)?;
        QMatMul::from_arc(Arc::new(qtensor))
    }

    fn load_bias(
        content: &gguf_file::Content,
        file: &mut std::fs::File,
        name: &str,
        device: &Device,
    ) -> Option<Tensor> {
        content
            .tensor(file, name, device)
            .ok()
            .and_then(|qt| qt.dequantize(device).ok())
    }

    fn load_rms_norm(
        content: &gguf_file::Content,
        file: &mut std::fs::File,
        name: &str,
        device: &Device,
        eps: f64,
    ) -> Result<RmsNorm> {
        let weight = content.tensor(file, name, device)?.dequantize(device)?;
        Ok(RmsNorm::new(weight, eps))
    }

    fn forward(&self, x: &Tensor, mask: &Tensor, rope: &RotaryEmbedding) -> Result<Tensor> {
        let residual = x;

        // Pre-norm for attention
        let x = self.attn_norm.forward(x)?;
        let x = self.self_attention(&x, mask, rope)?;
        let x = (residual + x)?;

        // Pre-norm for MLP
        let residual = &x;
        let h = self.ffn_norm.forward(&x)?;

        // SwiGLU: down(silu(gate(x)) * up(x))
        let gate = self.ffn_gate.forward(&h)?;
        let up = self.ffn_up.forward(&h)?;
        let h = (candle_nn::ops::silu(&gate)? * up)?;
        let h = self.ffn_down.forward(&h)?;

        residual + h
    }

    fn self_attention(&self, x: &Tensor, mask: &Tensor, rope: &RotaryEmbedding) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = x.dims3()?;

        // QKV projections
        let mut q = self.attn_q.forward(x)?;
        let mut k = self.attn_k.forward(x)?;
        let mut v = self.attn_v.forward(x)?;

        // Add biases if present
        if let Some(ref bias) = self.attn_q_bias {
            q = q.broadcast_add(bias)?;
        }
        if let Some(ref bias) = self.attn_k_bias {
            k = k.broadcast_add(bias)?;
        }
        if let Some(ref bias) = self.attn_v_bias {
            v = v.broadcast_add(bias)?;
        }

        // Reshape for multi-head attention: [batch, seq, heads * head_dim] -> [batch, heads, seq, head_dim]
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embeddings
        let q = rope.apply(&q, 0)?;
        let k = rope.apply(&k, 0)?;

        // Group Query Attention: repeat KV heads to match query heads
        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let attn = attn.broadcast_add(mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let out = out
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        self.attn_o.forward(&out)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            return Ok(x);
        }
        let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;
        x.unsqueeze(2)?
            .expand((batch, num_kv_heads, n_rep, seq_len, head_dim))?
            .reshape((batch, num_kv_heads * n_rep, seq_len, head_dim))
    }
}

/// Qwen2 model for embedding extraction.
///
/// Unlike the standard Qwen2 model that returns logits for next-token prediction,
/// this model returns the final hidden states for use as embeddings.
pub struct Qwen2ForEmbedding {
    tok_embeddings: Tensor,
    layers: Vec<Qwen2Layer>,
    final_norm: RmsNorm,
    rope: RotaryEmbedding,
    config: Qwen2Config,
    device: Device,
}

impl Qwen2ForEmbedding {
    /// Load model from GGUF file.
    pub fn from_gguf(
        content: gguf_file::Content,
        file: &mut std::fs::File,
        device: &Device,
        max_seq_len: usize,
    ) -> Result<Self> {
        let config = Qwen2Config::from_gguf(&content)?;

        // Load and dequantize token embeddings for efficient lookup
        let tok_embeddings = content
            .tensor(file, "token_embd.weight", device)?
            .dequantize(device)?;

        // Load transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            layers.push(Qwen2Layer::load(
                &content, file, device, &config, layer_idx,
            )?);
        }

        // Load final normalization (NOT the output projection - we want hidden states)
        let final_norm_weight = content
            .tensor(file, "output_norm.weight", device)?
            .dequantize(device)?;
        let final_norm = RmsNorm::new(final_norm_weight, config.rms_norm_eps);

        // Precompute rotary embeddings
        let rope = RotaryEmbedding::new(&config, max_seq_len.min(config.max_seq_len), device)?;

        Ok(Self {
            tok_embeddings,
            layers,
            final_norm,
            rope,
            config,
            device: device.clone(),
        })
    }

    /// Run forward pass and return hidden states (not logits).
    ///
    /// Returns tensor of shape `[batch, seq_len, hidden_size]`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // For 2D input [batch, seq], flatten for index_select then reshape
        let (batch, seq_len) = input_ids.dims2()?;
        let flat_ids = input_ids.flatten_all()?;
        let mut hidden = self.tok_embeddings.index_select(&flat_ids, 0)?;
        hidden = hidden.reshape((batch, seq_len, self.config.hidden_size))?;

        // Create causal attention mask
        let mask = self.causal_mask(seq_len)?;

        // Run through all transformer layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &mask, &self.rope)?;
        }

        // Final normalization - returns hidden states for embedding
        self.final_norm.forward(&hidden)
    }

    /// Create causal attention mask for self-attention.
    fn causal_mask(&self, seq_len: usize) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
            .collect();

        Tensor::from_vec(mask, (1, 1, seq_len, seq_len), &self.device)
    }

    /// Get model configuration.
    pub fn config(&self) -> &Qwen2Config {
        &self.config
    }
}

/// Create a causal attention mask for self-attention (test helper).
///
/// Returns a tensor of shape `[1, 1, seq_len, seq_len]` where positions
/// that can attend are 0.0 and positions that cannot attend are -inf.
#[cfg(test)]
pub(crate) fn create_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
        .collect();

    Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)
}

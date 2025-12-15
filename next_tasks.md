# Reflex: Next Tasks

Technical debt, bugs, and improvement opportunities identified through code review.

---

## Critical Issues

### 1. Lifecycle Reaper Never Records Activity

**Severity:** Critical
**Impact:** Instance will shut down after idle timeout even under active load

The `LifecycleManager::record_activity()` method exists (`src/lifecycle/manager.rs:55-57`) but is **never called** from the request path. The reaper thread (`start_reaper_thread()` at line 119) checks `idle_duration()` against `idle_timeout`, but since no activity is recorded, the instance will always appear idle.

**References:**
- `src/main.rs:67` — Starts reaper thread
- `src/lifecycle/manager.rs:55-57` — `record_activity()` method (unused)
- `src/lifecycle/manager.rs:119-167` — Reaper thread logic
- `src/gateway/handler.rs` — Request handler (never calls `record_activity`)

**Fix:** Call `lifecycle.record_activity().await` in `chat_completions_handler` or via middleware.

---

### 2. Embedding Dimension Invariants Are Inconsistent

**Severity:** Critical
**Impact:** Non-1536 embedder will break or silently corrupt data

Runtime-configurable dimension plumbing exists (`L2Config::vector_size`, `SinterConfig::embedding_dim`, `DimConfig`) but multiple paths hard-code `DEFAULT_EMBEDDING_DIM = 1536`:

| File | Line | Issue |
|------|------|-------|
| `src/constants.rs` | 14-18 | `DEFAULT_EMBEDDING_DIM`, `EMBEDDING_F16_BYTES` defined as const |
| `src/vectordb/model.rs` | 104 | `embedding_bytes_to_f32` requires exactly `EMBEDDING_F16_BYTES` |
| `src/vectordb/rescoring.rs` | 12-14, 132, 157 | `DEFAULT_EMBEDDING_DIM` used for validation |
| `src/main.rs` | 85 | `l2_config.vector_size(embedder.embedding_dim())` suggests runtime support |

**References:**
- `src/constants.rs:14` — `DEFAULT_EMBEDDING_DIM = 1536`
- `src/vectordb/model.rs:103-109` — Hard-coded byte length check
- `src/vectordb/rescoring.rs:132-136` — Validates against `DEFAULT_EMBEDDING_DIM`

**Fix:** Either enforce 1536 everywhere, or propagate `DimConfig` through all paths consistently.

---

### 3. Dangerous Defaults for Non-GCE Environments

**Severity:** Critical
**Impact:** Local dev/test may attempt GCP operations or instance stop

`LifecycleConfig::default()` sets:
- `cloud_provider: CloudProviderType::Gcp` (line 48)
- `enable_instance_stop: true` (line 47)

**References:**
- `src/lifecycle/config.rs:40-52` — Default implementation

**Fix:** Default to `Local` provider and `enable_instance_stop: false`, or require explicit opt-in.

---

## High Priority

### 4. SinterEmbedder Serializes All Inference Behind Single Mutex

**Severity:** High
**Impact:** Throughput bottleneck under concurrent requests

All embedding requests contend on a single `Mutex<Qwen2ForEmbedding>`:

```rust
model: Arc<Mutex<Qwen2ForEmbedding>>,
```

**References:**
- `src/embedding/sinter/mod.rs:23-27` — `EmbedderBackend::Model` definition
- `src/embedding/sinter/mod.rs:88` — `Arc::new(Mutex::new(model))`
- `src/embedding/sinter/mod.rs:211-216` — `model.lock().forward()`

**Fix:** Consider model sharding, batching queue, or async inference pool.

---

### 5. Alignment/Endianness Risks in f16 Byte Reinterpretation

**Severity:** High
**Impact:** Silent data corruption or dropped candidates on some platforms

`bytes_to_f16_slice()` uses `bytemuck::try_cast_slice` which requires proper alignment. `Vec<u8>` from arbitrary sources may not be 2-byte aligned.

**References:**
- `src/vectordb/rescoring.rs:260-262` — `bytes_to_f16_slice()`
- `src/vectordb/model.rs:103-125` — Manual little-endian conversion (safer approach)

**Fix:** Use the manual byte-to-f16 approach (`from_le_bytes`) consistently, or guarantee alignment at storage layer.

---

### 6. Duplicate Embedding Work on Cache Miss

**Severity:** High
**Impact:** Wasted compute, doubled latency

On cache miss, embedding is computed twice:
1. In `tiered_cache.lookup_with_semantic_query()` for L2 search
2. Again at line 207-212 for storage

**References:**
- `src/gateway/handler.rs:72-76` — First embedding (inside lookup)
- `src/gateway/handler.rs:207-212` — Second embedding (for storage)

**Fix:** Return embedding from L2 lookup result and reuse it.

---

### 7. `join_all` Fan-Out Creates Unbounded Concurrency

**Severity:** High
**Impact:** Load spikes can overwhelm storage layer

L2 cache parallel loading spawns unbounded concurrent operations:

```rust
let load_results = join_all(load_futures).await;
```

**References:**
- `src/cache/l2/cache.rs:117-132` — Parallel storage loading

**Fix:** Use `futures::stream::iter(...).buffer_unordered(MAX_CONCURRENT)`.

---

## Medium Priority

### 8. Streaming Response Compatibility Issues

**Severity:** Medium
**Impact:** Some OpenAI-compatible clients may break

SSE chunks generate a new `id` per chunk (line 89). Many clients expect stable `id` across the stream.

**References:**
- `src/gateway/streaming.rs:87-106` — New UUID per chunk
- `src/gateway/streaming.rs:18-45` — Cache bypass rationale (documented)

**Fix:** Generate single `id` at stream start, reuse for all chunks.

---

### 9. `parse_u64_from_env` Silent Fallback

**Severity:** Medium
**Impact:** Configuration errors go unnoticed

```rust
fn parse_u64_from_env(var_name: &str, default: u64) -> u64 {
    env::var(var_name)
        .ok()
        .and_then(|v| v.parse().ok())  // Silent fallback on parse error
        .unwrap_or(default)
}
```

Contrast with `parse_port_from_env` which returns explicit errors.

**References:**
- `src/config/mod.rs:153-158` — Silent fallback
- `src/config/mod.rs:110-126` — Strict port parsing (inconsistent)

**Fix:** Return `Result` or log warning on parse failures.

---

### 10. Cross-Encoder Scoring Uses `partial_cmp` with `Equal` Fallback

**Severity:** Medium
**Impact:** NaN scores may cause incorrect ordering

```rust
scored.sort_by(|a, b| {
    b.cross_encoder_score
        .partial_cmp(&a.cross_encoder_score)
        .unwrap_or(Ordering::Equal)
});
```

**References:**
- `src/scoring/scorer.rs:68-72` — First occurrence
- `src/scoring/scorer.rs:139-143` — Second occurrence
- `src/scoring/scorer.rs:165-169` — Third occurrence
- `src/vectordb/rescoring.rs:178` — Same pattern

**Fix:** Filter NaN scores before sorting, or use `total_cmp` after converting to a sortable representation.

---

### 11. Reranker Config `with_threshold` Panics

**Severity:** Medium
**Impact:** Inconsistent error handling style

```rust
pub fn with_threshold(mut self, threshold: f32) -> Self {
    assert!(
        (0.0..=1.0).contains(&threshold),
        "threshold must be between 0.0 and 1.0"
    );
    ...
}
```

Other configs return `Result` from validation.

**References:**
- `src/embedding/reranker/config.rs:38-45` — Panic on invalid threshold
- `src/embedding/reranker/config.rs:47-62` — `validate()` returns `Result` (inconsistent)

**Fix:** Have `with_threshold` also return `Result`, or document panic behavior.

---

### 12. Gateway Adapter Uses `expect` for JSON Construction

**Severity:** Medium
**Impact:** Panic if upstream types change shape

```rust
serde_json::from_value(message_value).expect("constructed OpenAI message is valid");
...
serde_json::from_value(response_value).expect("constructed OpenAI response is valid");
```

**References:**
- `src/gateway/adapter.rs:94-95` — First expect
- `src/gateway/adapter.rs:117` — Second expect

**Fix:** Return `GatewayError` instead of panicking.

---

### 13. Repeated `ensure_collection` in `spawn_index_update`

**Severity:** Medium
**Impact:** Redundant network calls per write

Every cache miss calls `ensure_collection` before `upsert_points`, even though collection is ensured at startup.

**References:**
- `src/gateway/handler.rs:441-447` — `ensure_collection` in spawned task
- `src/main.rs:97` — Already ensured at startup

**Fix:** Remove `ensure_collection` from hot path, or add in-memory "ensured" flag.

---

## Low Priority

### 14. L1 Cache Key Allocates String Per Lookup

**Severity:** Low
**Impact:** Minor memory churn under high load

```rust
format!("{}:{}", tenant_id, exact_key)
```

**References:**
- `src/cache/l1.rs` — Key formatting (location varies)

**Fix:** Consider interning or pre-allocated key buffer.

---

### 15. `TieredCacheHandle` Uses `RwLock` But Only Reads

**Severity:** Low
**Impact:** None currently (future-proofing)

```rust
pub struct L2SemanticCacheHandle<B, S> {
    inner: Arc<RwLock<L2SemanticCache<B, S>>>,
}
```

All methods use `.read().await`.

**References:**
- `src/cache/l2/cache.rs:233-235` — Handle definition
- `src/cache/l2/cache.rs:244-260` — Only read locks taken

**Fix:** Simplify to `Arc<L2SemanticCache>` if no mutation needed, or document intent.

---

### 16. Storage `sync_all` May Be Expensive

**Severity:** Low
**Impact:** Write latency on durable storage

`sync_all()` guarantees durability but can be slow on some filesystems.

**References:**
- `src/storage/nvme/mod.rs` — `sync_all` usage

**Fix:** Consider `sync_data()` or batched fsync, depending on durability requirements.

---

### 17. `hamming_distance` Returns `u32::MAX` for Mismatched Lengths

**Severity:** Low
**Impact:** Easy to misuse in non-test code

**References:**
- `src/vectordb/bq/utils.rs` — `hamming_distance` function

**Fix:** Return `Option<u32>` or panic in debug builds.

---

### 18. L2Config Doesn't Validate `vector_size` Against Embedder

**Severity:** Low
**Impact:** Mismatch caught later with confusing errors

**References:**
- `src/cache/l2/config.rs` — No dimension reconciliation

**Fix:** Add `validate_with_embedder(dim: usize)` method.

---

### 19. BQ Oversampling Can Be Extreme

**Severity:** Low
**Impact:** Excessive search results if `limit` is small

Oversampling factor derived from `rescore_limit / limit` can be large.

**References:**
- `src/vectordb/bq/client.rs` — Oversampling calculation

**Fix:** Cap oversampling factor (e.g., max 10x).

---

### 20. Last-Token Pooling May Not Be Optimal

**Severity:** Low
**Impact:** Embedding quality trade-off

Current implementation uses last-token pooling:
```rust
hidden_states.i((0, last_idx, ..self.config.embedding_dim))
```

Mean pooling or CLS token may perform better depending on model.

**References:**
- `src/embedding/sinter/mod.rs:218-229` — Last-token extraction

**Fix:** Make pooling strategy configurable.

---

### 21. Embedding Error Collapses IO Errors

**Severity:** Low
**Impact:** Hard to debug model loading failures

`From<std::io::Error>` maps all IO errors to `ModelLoadFailed`.

**References:**
- `src/embedding/error.rs` — Error conversions

**Fix:** Preserve IO error kind or add specific variants.

---

## Housekeeping

### 22. `src/check_genai.rs` — Orphan Scratch File

**Severity:** Info
**Impact:** Confusion, clutters codebase

Appears to be a test/scratch file. Not declared in `Cargo.toml` as a binary.

**References:**
- `src/check_genai.rs:1-13` — Entire file

**Fix:** Delete or move to `examples/`.

---

## Summary

| Priority | Count |
|----------|-------|
| Critical | 3 |
| High | 4 |
| Medium | 6 |
| Low | 8 |
| Info | 1 |
| **Total** | **22** |

---

*Generated: 2025-12-14*

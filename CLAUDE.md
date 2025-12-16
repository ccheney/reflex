# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reflex is an episodic memory and semantic cache for LLM responses. It sits between coding agents (Claude Code, Aider, etc.) and LLM providers (OpenAI, Anthropic), caching responses to reduce redundant API calls. Responses are returned in Tauq format for ~40% token savings.

## Build Commands

```bash
# Build (CPU only - default)
cargo build -p reflex-server --release

# Build with Apple Silicon GPU acceleration
cargo build -p reflex-server --release --features metal

# Build with NVIDIA GPU acceleration
cargo build -p reflex-server --release --features cuda
```

## Running Tests

```bash
# Run all tests (uses mocks by default)
cargo test

# Run a specific test
cargo test test_name

# Run tests in a specific module
cargo test -p reflex-cache cache::l1_tests

# Run integration tests (requires Qdrant running)
docker compose up -d qdrant
REFLEX_QDRANT_URL=http://localhost:6334 cargo test -p reflex-server --test integration_real
```

## Linting and Formatting

```bash
# Check formatting
cargo fmt -- --check

# Apply formatting
cargo fmt

# Run clippy (CI uses -D warnings)
cargo clippy --all-targets -- -D warnings
```

## Running the Server

```bash
# Start dependencies
docker compose up -d

# Run with debug logging
RUST_LOG=debug cargo run -p reflex-server

# Production run
cargo run -p reflex-server --release
```

## Architecture

### Tiered Cache System

```
Request → L1 (Exact Hash) → L2 (Semantic Search) → L3 (Cross-Encoder) → Provider
              <1ms              ~5ms                  ~10ms
```

- **L1 Cache** (`crates/reflex-cache/src/cache/l1.rs`): Blake3 hash-based exact matching using Moka in-memory cache
- **L2 Cache** (`crates/reflex-cache/src/cache/l2/`): Vector similarity search via Qdrant with binary quantization
- **L3 Verification** (`crates/reflex-cache/src/scoring/`): ModernBERT cross-encoder reranking to filter false positives
- **Tiered Orchestration** (`crates/reflex-cache/src/cache/tiered.rs`): Coordinates all cache tiers

### Key Modules

| Module | Purpose |
|--------|---------|
| `crates/reflex-server/src/gateway/` | Axum HTTP server with OpenAI-compatible API (`POST /v1/chat/completions`) |
| `crates/reflex-cache/src/embedding/sinter/` | Qwen2-based embedder (1536-dim vectors); has `real.rs` and `stub.rs` variants |
| `crates/reflex-cache/src/embedding/reranker/` | ModernBERT cross-encoder for L3 verification |
| `crates/reflex-cache/src/vectordb/` | Qdrant client with binary quantization (`crates/reflex-cache/src/vectordb/bq/`) |
| `crates/reflex-cache/src/storage/` | Memory-mapped (rkyv) and NVMe storage backends |
| `crates/reflex-cache/src/payload/` | Tauq semantic encoding/decoding |
| `crates/reflex-cache/src/lifecycle/` | GCP spot instance lifecycle management (hydrate/dehydrate) |
| `crates/reflex-cache/src/config/` | Environment-based configuration |

### Cargo Features

- `default = []` - No GPU acceleration
- `metal` - Apple Silicon GPU via Metal
- `cuda` - NVIDIA GPU via CUDA
- `mock` - Mock providers for testing (used by dev-dependencies)

### Constants

Embedding dimension is 1536 (defined in `crates/reflex-cache/src/constants.rs`). This is used throughout the codebase for vector operations.

## Testing Patterns

- Unit tests are co-located with source files (e.g., `l1_tests.rs` alongside `l1.rs`)
- Integration tests are in `crates/*/tests/` directories
- Test helpers are in `crates/*/tests/common/`
- Set `REFLEX_MOCK_PROVIDER=1` to bypass real LLM calls in CI
- Tests use `serial_test` crate for tests that need exclusive resource access

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REFLEX_PORT` | 8080 | HTTP server port |
| `REFLEX_BIND_ADDR` | 127.0.0.1 | Bind address |
| `REFLEX_QDRANT_URL` | http://localhost:6334 | Qdrant gRPC endpoint |
| `REFLEX_STORAGE_PATH` | ./.data | Path for mmap storage |
| `REFLEX_MODEL_PATH` | None | Path to embedding model (stub mode if unset) |
| `REFLEX_RERANKER_PATH` | None | Path to reranker model (optional) |
| `REFLEX_L1_CAPACITY` | 10000 | Max L1 cache entries |
| `REFLEX_RERANKER_THRESHOLD` | 0.70 | L3 verification confidence threshold |

## Code Conventions

- Uses Rust 2024 edition with Rust 1.92.0
- Error handling via `thiserror` for library errors, `anyhow` for application errors
- Async runtime is Tokio with full features
- Memory allocator is mimalloc (see `crates/reflex-server/src/main.rs`)
- Build artifacts go to `.target/` (configured in `.cargo/config.toml`)

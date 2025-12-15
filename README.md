> **Under Construction**: This project is actively being developed and is not yet ready for production use. APIs and features may change without notice.
<p align="center">
  <strong>Episodic Memory & Semantic Cache for LLM Responses</strong>
</p>

<p align="center">
  <a href="https://github.com/ccheney/reflex/actions"><img src="https://img.shields.io/github/actions/workflow/status/ccheney/reflex/ci.yml?branch=master&style=flat-square" alt="Build Status"></a>
  <a href="https://github.com/ccheney/reflex/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square" alt="License"></a>
  <a href="https://crates.io/crates/reflex-cache"><img src="https://img.shields.io/crates/v/reflex-cache?style=flat-square" alt="Crates.io"></a>
  <a href="https://github.com/ccheney/reflex"><img src="https://img.shields.io/badge/rust-2024_edition-orange?style=flat-square" alt="Rust 2024"></a>
</p>

<p align="center">
  <em>Because nobody likes paying for the same token twice.</em>
</p>

<p>
  <pre align="center">
██████╗ ███████╗███████╗██╗     ███████╗██╗  ██╗
██╔══██╗██╔════╝██╔════╝██║     ██╔════╝╚██╗██╔╝
██████╔╝█████╗  █████╗  ██║     █████╗   ╚███╔╝
██╔══██╗██╔══╝  ██╔══╝  ██║     ██╔══╝   ██╔██╗
██║  ██║███████╗██║     ███████╗███████╗██╔╝ ██╗
╚═╝  ╚═╝╚══════╝╚═╝     ╚══════╝╚══════╝╚═╝  ╚═╝
  </pre>
</p>

---

## Why Reflex?

Every time your coding agent asks "How do I center a div?" it costs you tokens. Every time it asks "What's the syntax for centering a div in CSS?" it costs you tokens *again*---even though you already have a perfectly good answer sitting in your logs.

**The problem is threefold:**

| Pain Point | What Happens | The Cost |
|------------|--------------|----------|
| **Duplicate queries** | Agents ask semantically identical questions | You pay for the same answer repeatedly |
| **Context bloat** | JSON responses eat precious context window | Agents hit token limits faster |
| **Cold starts** | Every session starts from scratch | No institutional memory |

**Reflex solves this** by sitting between your agent and the LLM provider:

```
Agent  --->  Reflex  --->  OpenAI/Anthropic/etc.
               |
               v
         [Cache Hit?]
               |
      Yes: Return instantly (~1ms)
       No: Forward, cache response, return
```

**The killer feature:** Reflex doesn't just cache---it compresses. Responses are returned in **[Tauq](https://github.com/epistates/tauq) format**, a semantic encoding that cuts token overhead by ~40%. Your agent gets the same information in fewer tokens, extending its effective context window.

---

## Quick Start

```bash
# 1. Start Qdrant (vector database)
docker run -d -p 6334:6334 -p 6333:6333 qdrant/qdrant

# 2. Run Reflex
cargo run --release

# 3. Point your agent to localhost:8080
export OPENAI_BASE_URL=http://localhost:8080/v1
```

That's it. Your existing OpenAI-compatible code works unchanged.

---

## How It Works

Reflex uses a **tiered cache architecture** that balances speed with semantic understanding:

```
                    Request
                       │
                       ▼
              ┌────────────────┐
              │   L1: Exact    │  <-- Hash lookup (~0.1ms)
              │   (In-Memory)  │      Identical requests return instantly
              └───────┬────────┘
                      │ Miss
                      ▼
              ┌────────────────┐
              │  L2: Semantic  │  <-- Vector search (~5ms)
              │   (Qdrant)     │      "Center a div" ≈ "CSS centering"
              └───────┬────────┘
                      │ Candidates found
                      ▼
              ┌────────────────┐
              │ L3: Verification│ <-- Cross-encoder (~10ms)
              │  (ModernBERT)   │     Ensures intent actually matches
              └───────┬────────┘
                      │ Verified / No match
                      ▼
              ┌────────────────┐
              │   Provider     │  <-- Forward to OpenAI/Anthropic
              │   (Upstream)   │      Cache response for next time
              └────────────────┘
```

### Cache Tiers Explained

| Tier | What It Does | When It Hits | Latency |
|------|--------------|--------------|---------|
| **L1 Exact** | Blake3 hash match on serialized request | Identical JSON payloads | <1ms |
| **L2 Semantic** | Embedding similarity via Qdrant | "How to sort?" ~ "Sorting algorithm?" | ~5ms |
| **L3 Verification** | Cross-encoder reranking | Confirms semantic match is valid | ~10ms |

The L3 verification step is critical: it prevents false positives from L2. Just because two questions are *similar* doesn't mean they have the same *answer*. The cross-encoder ensures the cached response actually answers the new question.

---

## Features

- **OpenAI-Compatible API** --- Drop-in replacement. Change `base_url`, nothing else.
- **Tiered Caching** --- L1 exact + L2 semantic + L3 verification
- **[Tauq](https://github.com/epistates/tauq) Compression** --- ~40% token savings via semantic encoding
- **Multi-Tenant** --- Isolated caches per API key
- **GPU Acceleration** --- Metal (Apple Silicon) and CUDA support
- **Zero-Copy Storage** --- Memory-mapped files via `rkyv` for instant deserialization
- **Self-Hosted** --- Your data stays on your machine
- **Graceful Lifecycle** --- Automatic state hydration/dehydration for cloud deployments

---

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/ccheney/reflex.git
cd reflex

# CPU only
cargo build --release

# Apple Silicon (Metal)
cargo build --release --features metal

# NVIDIA GPU (CUDA)
cargo build --release --features cuda
```

### Docker Compose

```bash
# CPU
docker compose up -d

# With GPU backend
GPU_BACKEND=metal docker compose up -d
```

### Prerequisites

- **Rust 2024 Edition** (nightly or 1.85+)
- **Qdrant** vector database (included in docker-compose)
- **Embedding model** (Qwen2-based, 1536-dim)
- **Reranker model** (ModernBERT cross-encoder) [optional but recommended]

---

## Configuration

All configuration is via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `REFLEX_PORT` | HTTP port to bind | `8080` |
| `REFLEX_BIND_ADDR` | IP address to bind | `127.0.0.1` |
| `REFLEX_QDRANT_URL` | Qdrant gRPC endpoint | `http://localhost:6334` |
| `REFLEX_STORAGE_PATH` | Path for mmap storage | `./.data` |
| `REFLEX_L1_CAPACITY` | Max entries in L1 cache | `10000` |
| `REFLEX_MODEL_PATH` | Path to embedding model | *Required for production* |
| `REFLEX_RERANKER_PATH` | Path to reranker model | *Optional* |
| `REFLEX_RERANKER_THRESHOLD` | L3 verification confidence (0.0-1.0) | `0.70` |

### Cloud/Lifecycle Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `REFLEX_GCS_BUCKET` | GCS bucket for state persistence | *Empty* |
| `REFLEX_SNAPSHOT_PATH` | Local snapshot file path | `/mnt/nvme/reflex_data/snapshot.rkyv` |
| `REFLEX_IDLE_TIMEOUT_SECS` | Idle timeout before shutdown | `900` (15 min) |
| `REFLEX_CLOUD_PROVIDER` | Cloud provider (`gcp`, `local`) | `gcp` |

---

## API Reference

### `POST /v1/chat/completions`

Fully OpenAI-compatible. Your existing code works unchanged.

**Request:**
```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "user", "content": "How do I center a div in CSS?"}
  ]
}
```

**Response (Cache Hit):**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "gpt-4o",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "! def\n: semantic_request \"How do I center a div in CSS?\"\n: response !\n  : content \"Use flexbox: display: flex; justify-content: center; align-items: center;\""
    },
    "finish_reason": "stop"
  }]
}
```

The `content` field contains **[Tauq](https://github.com/epistates/tauq)-encoded** data. Your agent can parse this to extract:
- The original semantic request (for verification)
- The cached response content

**Response Headers:**
| Header | Values | Meaning |
|--------|--------|---------|
| `X-Reflex-Status` | `hit-l1-exact` | L1 cache hit (identical request) |
| | `hit-l3-verified` | L2 candidate verified by L3 |
| | `miss` | Cache miss, forwarded to provider |

### `GET /healthz`

Liveness probe. Returns `200 OK` if the server is running.

### `GET /ready`

Readiness probe. Returns `200 OK` when Qdrant is connected and storage is operational.

---

## Usage with Coding Agents

### Claude Code / Aider / Continue

```bash
# Set the base URL to Reflex
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=sk-your-key  # Still needed for cache misses

# Run your agent as normal
aider --model gpt-4o
```

### Python (openai SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-your-key"
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain quicksort"}]
)
```

### async-openai (Rust)

```rust
use async_openai::{Client, config::OpenAIConfig};

let config = OpenAIConfig::new()
    .with_api_base("http://localhost:8080/v1");
let client = Client::with_config(config);
```

---

## Architecture Deep Dive

### Storage Layer

Reflex uses **rkyv** for zero-copy deserialization. Cached responses are memory-mapped directly from disk---no parsing, no allocation, no delay.

```
.data/
├── {tenant_id}/
│   ├── {context_hash}.rkyv    # Archived CacheEntry
│   └── ...
```

### Embedding Pipeline

1. **Tokenization** --- Qwen2 tokenizer (8192 max sequence length)
2. **Embedding** --- Sinter embedder (1536-dim vectors)
3. **Indexing** --- Binary quantization in Qdrant for efficient ANN search

### Verification Pipeline

The L3 cross-encoder takes `(query, candidate)` pairs and outputs a relevance score. Only candidates exceeding the threshold (default 0.70) are considered valid cache hits.

This prevents:
- "How to sort ascending?" returning cached "How to sort descending?" answer
- "Python quicksort" returning cached "Rust quicksort" implementation
- False positives from embedding similarity alone

---

## Development

```bash
# Run tests
cargo test

# Run with debug logging
RUST_LOG=debug cargo run

# Run specific integration tests (requires Qdrant)
docker compose up -d qdrant
cargo test --test integration_real

# Lint
cargo clippy --all-targets -- -D warnings

# Format
cargo fmt
```

### Test Modes

| Environment Variable | Effect |
|---------------------|--------|
| `REFLEX_MOCK_PROVIDER=1` | Bypass real LLM calls (for CI) |

---

## Roadmap

- [ ] Streaming response caching
- [ ] Cache invalidation API
- [ ] Prometheus metrics endpoint
- [ ] Multi-model embedding support
- [ ] Redis L1 backend option

---

## License

[AGPL-3.0](LICENSE) --- Free as in freedom. If you modify Reflex and distribute it (including as a service), you must release your modifications under the same license.

---

<p align="center">
  <strong>Reflex: Stop paying for the same token twice.</strong>
</p>

<p align="center">
  <sub>Built with Rust, Qdrant, and a healthy disdain for redundant API calls.</sub>
</p>

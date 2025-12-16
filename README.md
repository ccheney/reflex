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

## What It Is

Reflex is an **OpenAI-compatible HTTP cache** for LLM responses: it sits between your agent/app and the provider, returning cached answers instantly and storing misses for later reuse. Cached responses are returned in [Tauq](https://github.com/epistates/tauq) format to reduce token overhead.

---

## Quick Start (Server)

```bash
# 1. Start Qdrant (vector database)
docker run -d -p 6334:6334 -p 6333:6333 qdrant/qdrant

# 2. Run Reflex (HTTP server)
cargo run -p reflex-server --release

# 3. Point your agent to localhost:8080
export OPENAI_BASE_URL=http://localhost:8080/v1
```

---

## Crates In This Repo

- **Server + binary (`reflex`)**: `crates/reflex-server/README.md`
- **Core library (embedded use)**: `crates/reflex-cache/README.md` (docs.rs: https://docs.rs/reflex-cache)

---

## How It Works (High Level)

```
Request → L1 (exact) → L2 (semantic) → L3 (rerank/verify) → Provider
```

- **L1**: exact match (fast, in-memory)
- **L2**: semantic retrieval (Qdrant vector search)
- **L3**: verification (cross-encoder rerank to avoid false positives)

---

## Configuration (Server)

| Variable | Default |
|----------|---------|
| `REFLEX_PORT` | `8080` |
| `REFLEX_BIND_ADDR` | `127.0.0.1` |
| `REFLEX_QDRANT_URL` | `http://localhost:6334` |
| `REFLEX_STORAGE_PATH` | `./.data` |
| `REFLEX_L1_CAPACITY` | `10000` |
| `REFLEX_MODEL_PATH` | *(unset → stub embedder)* |
| `REFLEX_RERANKER_PATH` | *(optional)* |
| `REFLEX_RERANKER_THRESHOLD` | `0.70` |

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
RUST_LOG=debug cargo run -p reflex-server

# Run specific integration tests (requires Qdrant)
docker compose up -d qdrant
cargo test -p reflex-server --test integration_real

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

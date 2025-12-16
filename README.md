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

## Quick Start (Library)

```bash
# Run the library example (no HTTP server)
cargo run -p reflex-cache --example basic_lookup --features mock
```

Embed in your own app:

```toml
[dependencies]
reflex = { package = "reflex-cache", version = "0.1.2" }
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

## Development

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt -- --check
```

## License

[AGPL-3.0](LICENSE) --- Free as in freedom. If you modify Reflex and distribute it (including as a service), you must release your modifications under the same license.

---

<p align="center">
  <strong>Reflex: Stop paying for the same token twice.</strong>
</p>

<p align="center">
  <sub>Built with Rust, Qdrant, and a healthy disdain for redundant API calls.</sub>
</p>

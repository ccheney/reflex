# reflex-cache

`reflex-cache` is the **core Reflex library**: tiered cache, storage, embeddings/reranking, and vector DB integration.

This crate is published to crates.io as `reflex-cache`, but exposes its library as `reflex` (so downstream code can `use reflex::...`).

Docs: https://docs.rs/reflex-cache

## Use As A Dependency

```toml
[dependencies]
reflex = { package = "reflex-cache", version = "0.1.2" }
```

## Quick Start

From this repo:

```bash
cargo run -p reflex-cache --example basic_lookup --features mock
```

## What’s Inside

- `cache`: tiered cache orchestration (L1 exact + L2 semantic)
- `storage`: rkyv-backed storage + mmap/NVMe loaders
- `vectordb`: Qdrant client + binary quantization helpers (and mocks behind `mock`)
- `embedding`: embedder + reranker wiring
- `scoring`: L3 verification (cross-encoder)
- `config`: env-backed configuration types used by the server

## Features

- `metal`: Apple Silicon acceleration (passes through to model deps)
- `cuda`: NVIDIA acceleration (passes through to model deps)
- `mock`: enables mock backends for tests/examples

## Build / Test

From repo root:

```bash
cargo test -p reflex-cache
cargo doc -p reflex-cache --no-deps
```

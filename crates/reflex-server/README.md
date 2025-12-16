# reflex-server

`reflex-server` owns the **HTTP gateway** (Axum) and builds the `reflex` binary.

It depends on the core library crate (`reflex-cache`) for the cache tiers, storage, embeddings, and vector DB integration.

## Run (Dev)

```bash
# Requires Qdrant running (gRPC on 6334)
docker run -d -p 6334:6334 -p 6333:6333 qdrant/qdrant

RUST_LOG=debug cargo run -p reflex-server
```

## Run (Release)

```bash
cargo run -p reflex-server --release
```

## Docker Compose

From repo root:

```bash
docker compose up -d
```

## Endpoints

- `GET /healthz`
- `GET /ready`
- `POST /v1/chat/completions` (OpenAI-compatible)

## Configuration

Most commonly used env vars:

| Variable | Default | Notes |
|----------|---------|------|
| `REFLEX_PORT` | `8080` | HTTP port |
| `REFLEX_BIND_ADDR` | `127.0.0.1` | Bind address |
| `REFLEX_QDRANT_URL` | `http://localhost:6334` | Qdrant gRPC |
| `REFLEX_STORAGE_PATH` | `./.data` | Storage base path |
| `REFLEX_L1_CAPACITY` | `10000` | L1 capacity |
| `REFLEX_MODEL_PATH` | *(unset)* | Unset = stub embedder |
| `REFLEX_RERANKER_PATH` | *(unset)* | Optional reranker |
| `REFLEX_RERANKER_THRESHOLD` | `0.70` | L3 threshold |
| `REFLEX_MOCK_PROVIDER` | *(unset)* | Set to bypass real provider calls |

## GPU Features

Build/run with one of:

- `--features metal` (Apple Silicon)
- `--features cuda` (NVIDIA)
- `--features cpu` (docs.rs-style CPU flag; usually you can omit)

Example:

```bash
cargo run -p reflex-server --release --no-default-features --features metal
```

## Tests

```bash
cargo test -p reflex-server
```

Real integration tests (needs Qdrant):

```bash
REFLEX_QDRANT_URL=http://localhost:6334 cargo test -p reflex-server --test integration_real
```


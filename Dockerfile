# syntax=docker/dockerfile:1.7

ARG RUST_VERSION=1.92.0
ARG GPU_BACKEND=cuda
ARG RELEASE=true

FROM rust:${RUST_VERSION} AS chef

ENV PATH=/usr/local/cargo/bin:$PATH

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo install cargo-chef --locked

WORKDIR /app

FROM chef AS planner

COPY Cargo.toml Cargo.lock* ./
COPY crates ./crates

RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder-env

ARG GPU_BACKEND
ARG RELEASE

FROM builder-env AS builder-cuda

ARG GPU_BACKEND
ARG RELEASE

ARG CUDA_KEYRING_DEB=cuda-keyring_1.1-1_all.deb
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    gnupg \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/${CUDA_KEYRING_DEB} \
    && dpkg -i ${CUDA_KEYRING_DEB} \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      cuda-nvcc-12-4 \
      libcublas-dev-12-4 \
    && rm -rf /var/lib/apt/lists/* ${CUDA_KEYRING_DEB}

COPY --from=planner /app/recipe.json recipe.json

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo chef cook \
    $(if [ "$RELEASE" = "true" ]; then echo "--release"; fi) \
    --no-default-features \
    --features "$GPU_BACKEND" \
    --recipe-path recipe.json

COPY Cargo.toml Cargo.lock* ./
COPY crates ./crates

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build \
    $(if [ "$RELEASE" = "true" ]; then echo "--release"; fi) \
    --no-default-features \
    --features "$GPU_BACKEND" \
    -p reflex-server \
    --bin reflex \
    && cp /app/target/$(if [ "$RELEASE" = "true" ]; then echo "release"; else echo "debug"; fi)/reflex /app/reflex-binary

FROM builder-env AS builder-metal

ARG GPU_BACKEND
ARG RELEASE

COPY --from=planner /app/recipe.json recipe.json

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo chef cook \
    $(if [ "$RELEASE" = "true" ]; then echo "--release"; fi) \
    --no-default-features \
    --features "$GPU_BACKEND" \
    --recipe-path recipe.json

COPY Cargo.toml Cargo.lock* ./
COPY crates ./crates

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build \
    $(if [ "$RELEASE" = "true" ]; then echo "--release"; fi) \
    --no-default-features \
    --features "$GPU_BACKEND" \
    -p reflex-server \
    --bin reflex \
    && cp /app/target/$(if [ "$RELEASE" = "true" ]; then echo "release"; else echo "debug"; fi)/reflex /app/reflex-binary

FROM builder-env AS builder-cpu

ARG GPU_BACKEND
ARG RELEASE

COPY --from=planner /app/recipe.json recipe.json

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo chef cook \
    $(if [ "$RELEASE" = "true" ]; then echo "--release"; fi) \
    --no-default-features \
    --features "$GPU_BACKEND" \
    --recipe-path recipe.json

COPY Cargo.toml Cargo.lock* ./
COPY crates ./crates

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build \
    $(if [ "$RELEASE" = "true" ]; then echo "--release"; fi) \
    --no-default-features \
    --features "$GPU_BACKEND" \
    -p reflex-server \
    --bin reflex \
    && cp /app/target/$(if [ "$RELEASE" = "true" ]; then echo "release"; else echo "debug"; fi)/reflex /app/reflex-binary

FROM builder-${GPU_BACKEND} AS builder

FROM rust:${RUST_VERSION}-slim-trixie AS runtime-base

ENV PATH=/usr/local/cargo/bin:$PATH

FROM runtime-base AS runtime-cuda
ARG CUDA_KEYRING_DEB=cuda-keyring_1.1-1_all.deb
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    gnupg \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/${CUDA_KEYRING_DEB} \
    && dpkg -i ${CUDA_KEYRING_DEB} \
    && apt-get update \
    && apt-get install -y --no-install-recommends libcublas-12-4 \
    && apt-get purge -y wget gnupg \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* ${CUDA_KEYRING_DEB}

FROM runtime-base AS runtime-metal
FROM runtime-base AS runtime-cpu

FROM runtime-${GPU_BACKEND} AS runtime

RUN useradd --create-home --user-group reflex
USER reflex
WORKDIR /home/reflex

COPY --from=builder --chown=reflex:reflex /app/reflex-binary /usr/local/bin/reflex

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["reflex", "--health-check"]

ENV RUST_LOG=info \
    REFLEX_PORT=8080

ENTRYPOINT ["reflex"]

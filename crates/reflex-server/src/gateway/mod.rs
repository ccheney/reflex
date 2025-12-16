//! HTTP gateway layer.

//! HTTP gateway (Axum) for cache lookup and storage.
//!
//! This module is primarily used by the `reflex` server binary.

#![allow(missing_docs)]

pub mod adapter;
pub mod error;
pub mod handler;
pub mod payload;
pub mod state;
pub mod streaming;

#[cfg(test)]
mod handler_tests;

use axum::{
    Json, Router,
    http::{HeaderMap, StatusCode, header::HeaderValue},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use tower_http::trace::TraceLayer;

pub use handler::chat_completions_handler;
pub use state::HandlerState;

use reflex::cache::{
    BqSearchBackend, REFLEX_STATUS_ERROR, REFLEX_STATUS_HEADER, REFLEX_STATUS_HEALTHY,
    REFLEX_STATUS_READY, StorageLoader,
};
use reflex::storage::StorageWriter;

pub fn create_router_with_state<B, S>(state: HandlerState<B, S>) -> Router
where
    B: BqSearchBackend + Clone + Send + Sync + 'static,
    S: StorageLoader + StorageWriter + Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/healthz", get(health_handler))
        .route("/ready", get(ready_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

#[derive(serde::Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

#[derive(serde::Serialize)]
pub struct ReadyResponse {
    pub status: &'static str,
    pub components: ComponentStatus,
}

#[derive(serde::Serialize)]
pub struct ComponentStatus {
    pub http: &'static str,
    pub storage: &'static str,
    pub vectordb: &'static str,
    pub embedding: &'static str,
    pub embedder_mode: &'static str,
}

#[tracing::instrument]
pub async fn health_handler() -> Response {
    let mut headers = HeaderMap::new();
    headers.insert(
        REFLEX_STATUS_HEADER,
        HeaderValue::from_static(REFLEX_STATUS_HEALTHY),
    );

    (
        StatusCode::OK,
        headers,
        Json(HealthResponse { status: "ok" }),
    )
        .into_response()
}

use axum::extract::State;

#[tracing::instrument(skip(state))]
pub async fn ready_handler<B, S>(State(state): State<HandlerState<B, S>>) -> Response
where
    B: BqSearchBackend + Clone + Send + Sync + 'static,
    S: StorageLoader + Clone + Send + Sync + 'static,
{
    let storage_status = if state.storage_path.exists() && state.storage_path.is_dir() {
        REFLEX_STATUS_READY
    } else {
        REFLEX_STATUS_ERROR
    };

    let vectordb_status = if state.tiered_cache.is_ready().await {
        REFLEX_STATUS_READY
    } else {
        "pending"
    };

    let embedding_status = REFLEX_STATUS_READY;

    let is_stub = state.tiered_cache.l2().is_embedder_stub();
    let embedder_mode = if is_stub { "stub" } else { "real" };

    let components = ComponentStatus {
        http: REFLEX_STATUS_READY,
        storage: storage_status,
        vectordb: vectordb_status,
        embedding: embedding_status,
        embedder_mode,
    };

    let is_ready = components.storage == REFLEX_STATUS_READY
        && components.vectordb == REFLEX_STATUS_READY
        && components.embedding == REFLEX_STATUS_READY;

    let status_code = if is_ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    let status_msg = if is_ready { "ok" } else { "pending" };

    let mut headers = HeaderMap::new();
    headers.insert(
        REFLEX_STATUS_HEADER,
        HeaderValue::from_str(status_msg).unwrap_or(HeaderValue::from_static("error")),
    );

    (
        status_code,
        headers,
        Json(ReadyResponse {
            status: status_msg,
            components,
        }),
    )
        .into_response()
}

//! The HTTP surface.
//!
//! `/v1/chat/completions` and `/v1/completions` generate, `/v1/models` describes
//! the model, `/health` and `/readyz` are probes, `/metrics` exports Prometheus
//! text. Errors use the OpenAI envelope with the status the situation deserves.

pub mod chat;
pub mod completions;
pub mod errors;
pub mod generate;
pub mod health;
pub mod models;

use std::sync::Arc;

use axum::Router;
use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use prometheus_client::encoding::prometheus_text;

use crate::api::errors::ApiError;
use crate::metrics::append_engine_metrics;
use crate::server::AppState;

/// Builds the router.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/completions", post(completions::completions))
        .route("/v1/models", get(models::list_models))
        .route("/health", get(health::health))
        .route("/readyz", get(health::ready))
        .route("/metrics", get(metrics))
        .with_state(state)
}

/// `GET /metrics`: the registry plus the engine's `inferno_gpu_*` counters.
pub async fn metrics(State(state): State<Arc<AppState>>) -> Result<Response, ApiError> {
    if !state.config.metrics.enabled {
        return Err(ApiError::not_found("metrics are disabled"));
    }
    let mut body = String::new();
    prometheus_text::encode(&mut body, &state.metrics.registry)
        .map_err(|error| ApiError::internal(format!("cannot encode metrics: {error}")))?;
    append_engine_metrics(&state.engine.metrics_json(), &mut body);
    Ok(([(CONTENT_TYPE, prometheus_text::CONTENT_TYPE)], body).into_response())
}

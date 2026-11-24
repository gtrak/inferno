//! `GET /health` (liveness) and `GET /readyz` (readiness).

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::server::AppState;

/// `GET /health`: the process is alive, whatever the engine is doing.
pub async fn health() -> Response {
    (StatusCode::OK, Json(serde_json::json!({"status": "ok"}))).into_response()
}

/// `GET /readyz`: 200 only when the engine can take work.
pub async fn ready(State(state): State<Arc<AppState>>) -> Response {
    if state.engine.ready() {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ready",
                "engine": state.engine.version(),
                "model": state.config.model_id,
            })),
        )
            .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "not_ready",
                "reason": "the engine is not initialised",
            })),
        )
            .into_response()
    }
}

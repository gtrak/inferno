//! `GET /v1/models`: the model this server serves.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;

use crate::server::AppState;

/// `GET /v1/models`: the single loaded model, with its context length.
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.config.model_id,
            "object": "model",
            "created": state.created,
            "owned_by": "inferno",
            "context_length": state.config.max_context,
        }],
    }))
}

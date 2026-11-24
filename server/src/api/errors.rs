//! The HTTP error envelope: OpenAI's `{"error": {"message", "type", "code"}}`
//! with the status code the situation deserves.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::engine::EngineError;

/// One client-visible failure.
#[derive(Debug)]
pub struct ApiError {
    /// HTTP status.
    pub status: StatusCode,
    /// OpenAI `type` field.
    pub error_type: &'static str,
    /// OpenAI `code` field, when there is one.
    pub code: Option<&'static str>,
    /// Human readable message.
    pub message: String,
}

impl ApiError {
    /// 400: the request is malformed or asks for something unsupported.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error_type: "invalid_request_error",
            code: None,
            message: message.into(),
        }
    }

    /// 429: the admission limit is reached; the client should retry.
    pub fn too_many_requests(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            error_type: "requests",
            code: Some("too_many_requests"),
            message: message.into(),
        }
    }

    /// 503: the engine cannot take work right now.
    pub fn not_ready(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            error_type: "engine_unavailable",
            code: Some("engine_not_ready"),
            message: message.into(),
        }
    }

    /// 500: the failure is ours.
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            error_type: "server_error",
            code: None,
            message: message.into(),
        }
    }

    /// 404: the endpoint is not exposed.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            error_type: "not_found_error",
            code: Some("not_found"),
            message: message.into(),
        }
    }

    /// Maps an engine failure onto a status: a rejected request is "we cannot
    /// serve this now" (503), anything else is our fault (500).
    pub fn from_engine(error: EngineError) -> Self {
        match error {
            EngineError::Rejected(code) => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                error_type: "engine_unavailable",
                code: Some("engine_rejected"),
                message: format!("the engine rejected the request with code {code}"),
            },
            EngineError::NotInitialised => Self::not_ready("the engine is not initialised"),
            EngineError::UnknownRequest(req_id) => {
                Self::internal(format!("the engine does not know request {req_id}"))
            }
            EngineError::Failed(message) => Self::internal(message),
        }
    }

    /// The envelope, as the JSON body.
    pub fn body(&self) -> serde_json::Value {
        serde_json::json!({
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.code,
            }
        })
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} {}: {}",
            self.status.as_u16(),
            self.error_type,
            self.message
        )
    }
}

impl std::error::Error for ApiError {}

impl From<crate::template::TemplateError> for ApiError {
    /// A template that refuses the messages is a bad request; anything else is ours.
    fn from(error: crate::template::TemplateError) -> Self {
        match error {
            crate::template::TemplateError::Render(message) => Self::invalid_request(message),
            other => Self::internal(other.to_string()),
        }
    }
}

impl From<crate::tokenizer::TokenizerError> for ApiError {
    fn from(error: crate::tokenizer::TokenizerError) -> Self {
        Self::internal(error.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status;
        let body = self.body();
        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_envelope_has_the_three_contract_fields() {
        let error = ApiError::invalid_request("messages must not be empty");
        let body = error.body();
        assert_eq!(body["error"]["message"], "messages must not be empty");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(body["error"]["code"].is_null());
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn an_engine_rejection_is_a_503_with_the_code() {
        let error = ApiError::from_engine(EngineError::Rejected(-7));
        assert_eq!(error.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error.code, Some("engine_rejected"));
        assert!(error.message.contains("-7"));
    }
}

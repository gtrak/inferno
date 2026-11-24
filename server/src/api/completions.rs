//! `POST /v1/completions`: the legacy text-completion surface, a thin wrapper
//! over the same generator (no chat template, the prompt is the prompt).

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use crate::api::chat::{CompletionIds, StopSequences, StreamOptions, sse_event, usage_json};
use crate::api::errors::ApiError;
use crate::api::generate::{self, FinishReason, GenerationEvent, GenerationPlan, SamplingParams};
use crate::server::AppState;

/// The prompt: a string, or a list of strings of which only one is served.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum CompletionPrompt {
    /// One prompt.
    One(String),
    /// A batch of prompts; only a single one is supported.
    Many(Vec<String>),
}

impl CompletionPrompt {
    /// The prompt text, refusing a batch.
    fn into_text(self) -> Result<String, ApiError> {
        match self {
            Self::One(text) => Ok(text),
            Self::Many(many) => match many.len() {
                1 => Ok(many.into_iter().next().expect("length checked")),
                _ => Err(ApiError::invalid_request(
                    "only a single prompt is supported (batch prompts are not served)",
                )),
            },
        }
    }
}

/// The legacy completion request.
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    /// Model id; echoed back.
    #[serde(default)]
    pub model: Option<String>,
    /// The prompt.
    pub prompt: CompletionPrompt,
    /// Ask for `text/event-stream`.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Streaming extras.
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Token budget of the answer.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus threshold.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-k cut-off.
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Min-p threshold.
    #[serde(default)]
    pub min_p: Option<f32>,
    /// Sampling seed.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Stop strings.
    #[serde(default)]
    pub stop: Option<StopSequences>,
    /// Number of choices; only 1 is served.
    #[serde(default)]
    pub n: Option<u32>,
}

/// `POST /v1/completions`.
pub async fn completions(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<CompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) =
        payload.map_err(|rejection| ApiError::invalid_request(rejection.body_text()))?;
    let plan = build_plan(&state, &request)?;
    let ids = CompletionIds {
        id: format!("cmpl-{}", state.next_request_id()),
        created: state.created,
        model: request
            .model
            .clone()
            .unwrap_or_else(|| state.config.model_id.clone()),
    };
    let include_usage = request
        .stream_options
        .map(|options| options.include_usage)
        .unwrap_or(false);

    let permit = generate::admit(&state)?;
    if request.stream.unwrap_or(false) {
        return Ok(stream_response(state, plan, ids, include_usage, permit));
    }

    let collected =
        generate::collect(generate::generate(Arc::clone(&state), plan.clone(), permit)).await?;
    let body = serde_json::json!({
        "id": ids.id,
        "object": "text_completion",
        "created": ids.created,
        "model": ids.model,
        "choices": [{
            "index": 0,
            "text": collected.text,
            "logprobs": serde_json::Value::Null,
            "finish_reason": collected.finish_reason.as_str(),
        }],
        "usage": usage_json(plan.prompt_tokens.len(), collected.completion_tokens),
    });
    Ok(Json(body).into_response())
}

/// Validates, tokenizes and budgets one legacy completion.
fn build_plan(
    state: &Arc<AppState>,
    request: &CompletionRequest,
) -> Result<GenerationPlan, ApiError> {
    if request.n.unwrap_or(1) != 1 {
        return Err(ApiError::invalid_request("only n = 1 is supported"));
    }
    if let Some(temperature) = request.temperature
        && !(0.0..=2.0).contains(&temperature)
    {
        return Err(ApiError::invalid_request(
            "temperature must be between 0 and 2",
        ));
    }
    if let Some(top_p) = request.top_p
        && !(top_p > 0.0 && top_p <= 1.0)
    {
        return Err(ApiError::invalid_request("top_p must be in (0, 1]"));
    }
    if let Some(min_p) = request.min_p
        && !(0.0..=1.0).contains(&min_p)
    {
        return Err(ApiError::invalid_request("min_p must be between 0 and 1"));
    }

    let prompt = request.prompt.clone().into_text()?;
    let prompt_tokens = state
        .tokenizer
        .encode(&prompt)
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let max_context = state.config.max_context as usize;
    if prompt_tokens.len() >= max_context {
        return Err(ApiError::invalid_request(format!(
            "the prompt is {} tokens, which does not fit the {max_context} token context",
            prompt_tokens.len()
        )));
    }
    let remaining = (max_context - prompt_tokens.len()) as u32;
    let max_tokens = match request.max_tokens {
        Some(0) => return Err(ApiError::invalid_request("max_tokens must be at least 1")),
        Some(max_tokens) => max_tokens,
        None => state.default_max_tokens(prompt_tokens.len()),
    };
    if max_tokens > remaining {
        return Err(ApiError::invalid_request(format!(
            "max_tokens {max_tokens} exceeds the {remaining} tokens left in the context"
        )));
    }

    let mut stop_token_ids = Vec::new();
    if let Some(eos) = state.tokenizer.eos_token_id() {
        stop_token_ids.push(eos);
    }
    Ok(GenerationPlan {
        prompt_tokens,
        max_tokens,
        sampling: SamplingParams {
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.unwrap_or(0),
            min_p: request.min_p.unwrap_or(0.0),
            seed: request.seed.unwrap_or(0),
        },
        stop_strings: request
            .stop
            .clone()
            .map(StopSequences::into_strings)
            .unwrap_or_default(),
        stop_token_ids,
        stream: request.stream.unwrap_or(false),
    })
}

/// Streams one legacy completion as server-sent events.
fn stream_response(
    state: Arc<AppState>,
    plan: GenerationPlan,
    ids: CompletionIds,
    include_usage: bool,
    permit: crate::admission::Admitted,
) -> Response {
    let prompt_tokens = plan.prompt_tokens.len();
    let generator = generate::generate(state, plan, permit);
    let events = async_stream::stream! {
        let mut completion_tokens = 0_u32;
        let mut finish_reason = FinishReason::Stop;
        let mut failed = false;
        for await event in generator {
            match event {
                Ok(GenerationEvent::Delta(text)) => {
                    let document = serde_json::json!({
                        "id": ids.id,
                        "object": "text_completion",
                        "created": ids.created,
                        "model": ids.model,
                        "choices": [{"index": 0, "text": text, "logprobs": serde_json::Value::Null, "finish_reason": serde_json::Value::Null}],
                    });
                    yield Ok::<_, std::convert::Infallible>(sse_event(&document));
                }
                Ok(GenerationEvent::Done { finish_reason: reason, completion_tokens: tokens }) => {
                    finish_reason = reason;
                    completion_tokens = tokens;
                }
                Err(error) => {
                    failed = true;
                    yield Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data(error.body().to_string()));
                }
            }
        }
        if !failed {
            let document = serde_json::json!({
                "id": ids.id,
                "object": "text_completion",
                "created": ids.created,
                "model": ids.model,
                "choices": [{"index": 0, "text": "", "logprobs": serde_json::Value::Null, "finish_reason": finish_reason.as_str()}],
            });
            yield Ok::<_, std::convert::Infallible>(sse_event(&document));
            if include_usage {
                yield Ok::<_, std::convert::Infallible>(sse_event(&serde_json::json!({
                    "id": ids.id,
                    "object": "text_completion",
                    "created": ids.created,
                    "model": ids.model,
                    "choices": [],
                    "usage": usage_json(prompt_tokens, completion_tokens),
                })));
            }
        }
        yield Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data("[DONE]"));
    };

    Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response()
}

//! `POST /v1/chat/completions`: the OpenAI chat surface, streaming and buffered.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use crate::api::errors::ApiError;
use crate::api::generate::{self, FinishReason, GenerationEvent, GenerationPlan, SamplingParams};
use crate::server::AppState;
use crate::template::{RenderContext, parse_tool_calls};

/// Longest `stop` list accepted; more than this is a client bug.
const MAX_STOP_SEQUENCES: usize = 8;

/// Message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Instructions for the whole conversation.
    System,
    /// The human turn.
    User,
    /// The model turn.
    Assistant,
    /// A tool result.
    Tool,
}

/// Message content: a plain string, or OpenAI's content-part array (text and
/// media parts are passed through to the template unchanged).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text.
    Text(String),
    /// Content parts.
    Parts(Vec<serde_json::Value>),
}

/// One conversation message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Who is speaking.
    pub role: Role,
    /// What was said.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    /// Optional speaker name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool calls the assistant asked for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    /// Which tool call a `tool` message answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Reasoning content of an earlier assistant turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// `stop`: one string or a list of them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopSequences {
    /// A single stop string.
    One(String),
    /// Several stop strings.
    Many(Vec<String>),
}

impl StopSequences {
    /// The stop strings, dropping empty ones.
    pub(crate) fn into_strings(self) -> Vec<String> {
        match self {
            Self::One(single) => vec![single],
            Self::Many(many) => many,
        }
    }
}

/// `stream_options`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamOptions {
    /// Adds a final SSE chunk carrying `usage`.
    #[serde(default)]
    pub include_usage: bool,
}

/// The chat completion request.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model id; echoed back, not used to route.
    #[serde(default)]
    pub model: Option<String>,
    /// The conversation.
    pub messages: Vec<ChatMessage>,
    /// Ask for `text/event-stream`.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Streaming extras.
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Token budget of the answer.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// OpenAI's newer spelling of `max_tokens`.
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
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
    /// Tools the model may call.
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    /// `false` renders the template's no-thinking prefix.
    #[serde(default)]
    pub enable_thinking: Option<bool>,
}

/// The identifiers every chunk of one completion repeats.
#[derive(Debug, Clone)]
pub(crate) struct CompletionIds {
    pub(crate) id: String,
    pub(crate) created: u64,
    pub(crate) model: String,
}

/// `POST /v1/chat/completions`.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) =
        payload.map_err(|rejection| ApiError::invalid_request(rejection.body_text()))?;
    let plan = build_plan(&state, &request)?;
    let ids = CompletionIds {
        id: format!("chatcmpl-{}", state.next_request_id()),
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
    let parsed = parse_tool_calls(&collected.text);
    let finish_reason = if parsed.tool_calls.is_empty() {
        collected.finish_reason
    } else {
        FinishReason::ToolCalls
    };
    let content = if parsed.content.is_empty() && !parsed.tool_calls.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::Value::String(parsed.content)
    };
    let mut message = serde_json::json!({
        "role": "assistant",
        "content": content,
    });
    if !parsed.tool_calls.is_empty() {
        let calls: Vec<serde_json::Value> = parsed
            .tool_calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                serde_json::json!({
                    "id": format!("call_{}_{index}", ids.id),
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                })
            })
            .collect();
        message["tool_calls"] = serde_json::Value::Array(calls);
    }

    let body = serde_json::json!({
        "id": ids.id,
        "object": "chat.completion",
        "created": ids.created,
        "model": ids.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason.as_str(),
        }],
        "usage": usage_json(plan.prompt_tokens.len(), collected.completion_tokens),
    });
    Ok(Json(body).into_response())
}

/// Builds the generation plan: validate, render, tokenize, budget.
pub(crate) fn build_plan(
    state: &Arc<AppState>,
    request: &ChatCompletionRequest,
) -> Result<GenerationPlan, ApiError> {
    validate(request)?;

    let messages = serde_json::to_value(&request.messages)
        .map_err(|error| ApiError::invalid_request(format!("messages: {error}")))?;
    let tools = match request.tools.as_ref() {
        Some(tools) => Some(
            serde_json::to_value(tools)
                .map_err(|error| ApiError::invalid_request(format!("tools: {error}")))?,
        ),
        None => None,
    };
    let prompt = state.template.render(&RenderContext {
        messages: &messages,
        tools: tools.as_ref(),
        add_generation_prompt: true,
        enable_thinking: request.enable_thinking,
    })?;
    let prompt_tokens = state.tokenizer.encode(&prompt)?;

    let max_context = state.config.max_context as usize;
    if prompt_tokens.len() >= max_context {
        return Err(ApiError::invalid_request(format!(
            "the prompt is {} tokens, which does not fit the {max_context} token context",
            prompt_tokens.len()
        )));
    }

    let requested = request.max_tokens.or(request.max_completion_tokens);
    let max_tokens = match requested {
        Some(max_tokens) => {
            if max_tokens == 0 {
                return Err(ApiError::invalid_request("max_tokens must be at least 1"));
            }
            max_tokens
        }
        None => state.default_max_tokens(prompt_tokens.len()),
    };
    let remaining = (max_context - prompt_tokens.len()) as u32;
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
        max_tokens: max_tokens.min(remaining),
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

/// Checks everything that does not need the tokenizer.
fn validate(request: &ChatCompletionRequest) -> Result<(), ApiError> {
    if request.messages.is_empty() {
        return Err(ApiError::invalid_request("messages must not be empty"));
    }
    if request.n.unwrap_or(1) != 1 {
        return Err(ApiError::invalid_request("only n = 1 is supported"));
    }
    if request.max_tokens == Some(0) || request.max_completion_tokens == Some(0) {
        return Err(ApiError::invalid_request("max_tokens must be at least 1"));
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
    if let Some(stop) = request.stop.as_ref() {
        let count = match stop {
            StopSequences::One(_) => 1,
            StopSequences::Many(many) => many.len(),
        };
        if count > MAX_STOP_SEQUENCES {
            return Err(ApiError::invalid_request(format!(
                "at most {MAX_STOP_SEQUENCES} stop sequences are supported"
            )));
        }
    }
    Ok(())
}

/// The `usage` block; `total_tokens` always adds up.
pub(crate) fn usage_json(prompt_tokens: usize, completion_tokens: u32) -> serde_json::Value {
    let prompt = prompt_tokens as u64;
    let completion = u64::from(completion_tokens);
    serde_json::json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    })
}

/// One `chat.completion.chunk` document.
fn chat_chunk(
    ids: &CompletionIds,
    delta: serde_json::Value,
    finish_reason: Option<&str>,
) -> serde_json::Value {
    serde_json::json!({
        "id": ids.id,
        "object": "chat.completion.chunk",
        "created": ids.created,
        "model": ids.model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    })
}

/// Wraps a JSON document into one SSE event.
pub(crate) fn sse_event(document: &serde_json::Value) -> Event {
    Event::default().data(document.to_string())
}

/// Streams one chat completion as server-sent events.
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
        yield Ok::<_, std::convert::Infallible>(sse_event(&chat_chunk(
            &ids,
            serde_json::json!({"role": "assistant", "content": ""}),
            None,
        )));

        let mut completion_tokens = 0_u32;
        let mut finish_reason = FinishReason::Stop;
        let mut failed = false;
        for await event in generator {
            match event {
                Ok(GenerationEvent::Delta(text)) => {
                    yield Ok::<_, std::convert::Infallible>(sse_event(&chat_chunk(
                        &ids,
                        serde_json::json!({"content": text}),
                        None,
                    )));
                }
                Ok(GenerationEvent::Done { finish_reason: reason, completion_tokens: tokens }) => {
                    finish_reason = reason;
                    completion_tokens = tokens;
                }
                Err(error) => {
                    // The status line is already on the wire; report in band.
                    failed = true;
                    yield Ok::<_, std::convert::Infallible>(Event::default().data(error.body().to_string()));
                }
            }
        }

        if !failed {
            yield Ok::<_, std::convert::Infallible>(sse_event(&chat_chunk(
                &ids,
                serde_json::json!({}),
                Some(finish_reason.as_str()),
            )));
            if include_usage {
                yield Ok::<_, std::convert::Infallible>(sse_event(&serde_json::json!({
                    "id": ids.id,
                    "object": "chat.completion.chunk",
                    "created": ids.created,
                    "model": ids.model,
                    "choices": [],
                    "usage": usage_json(prompt_tokens, completion_tokens),
                })));
            }
        }
        yield Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"));
    };

    Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_adds_up() {
        let usage = usage_json(7, 3);
        assert_eq!(usage["prompt_tokens"], 7);
        assert_eq!(usage["completion_tokens"], 3);
        assert_eq!(usage["total_tokens"], 10);
    }

    #[test]
    fn content_parts_and_strings_both_deserialise() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": "hi"},
            ],
            "stop": "\n\n",
        }))
        .expect("the request parses");
        assert_eq!(request.stop, Some(StopSequences::One("\n\n".to_owned())));
        assert!(matches!(
            request.messages[0].content,
            Some(MessageContent::Parts(_))
        ));
        assert!(matches!(
            request.messages[1].content,
            Some(MessageContent::Text(_))
        ));
    }

    #[test]
    fn bad_params_are_refused_before_any_work() {
        let cases = [
            (
                serde_json::json!({"messages": []}),
                "messages must not be empty",
            ),
            (
                serde_json::json!({"messages": [{"role": "user", "content": "x"}], "n": 2}),
                "only n = 1",
            ),
            (
                serde_json::json!({"messages": [{"role": "user", "content": "x"}], "temperature": 9.0}),
                "temperature",
            ),
            (
                serde_json::json!({"messages": [{"role": "user", "content": "x"}], "top_p": 0.0}),
                "top_p",
            ),
            (
                serde_json::json!({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}),
                "max_tokens",
            ),
        ];
        for (body, expected) in cases {
            let request: ChatCompletionRequest = serde_json::from_value(body).expect("parses");
            let error = validate(&request).expect_err("the request is refused");
            assert!(
                error.message.contains(expected),
                "expected {expected:?} in {:?}",
                error.message
            );
            assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
        }
    }

    #[test]
    fn a_valid_request_passes_validation() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
            "temperature": 0.0,
            "top_p": 0.9,
            "n": 1,
        }))
        .expect("parses");
        validate(&request).expect("valid");
    }
}

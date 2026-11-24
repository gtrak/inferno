//! One generation, shared by the streaming and the buffered response paths.
//!
//! Both paths poll the same stream, so a streamed answer and a buffered answer
//! cannot drift apart: same prompt, same stop handling, same accounting. The
//! stream stops when the engine says the request is done, when `max_tokens` is
//! reached, when a stop token arrives or when a stop string completes.

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::Stream;
use futures::StreamExt;
use tokio::sync::mpsc;

use crate::admission::Admitted;
use crate::api::errors::ApiError;
use crate::engine::{EngineToken, GenerateRequest, TOKEN_CHANNEL_DEPTH};
use crate::server::AppState;
use crate::stop::StopMatcher;
use crate::tokenizer::Tokenizer;

/// Sampling knobs for one request.
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    /// `0.0` means greedy.
    pub temperature: f32,
    /// Nucleus threshold.
    pub top_p: f32,
    /// Top-k cut-off, `0` disables it.
    pub top_k: u32,
    /// Min-p threshold.
    pub min_p: f32,
    /// Sampling seed.
    pub seed: u64,
}

/// Everything the generator needs about one request.
#[derive(Debug, Clone)]
pub struct GenerationPlan {
    /// Tokenized prompt.
    pub prompt_tokens: Vec<u32>,
    /// Upper bound on emitted tokens.
    pub max_tokens: u32,
    /// Sampling knobs.
    pub sampling: SamplingParams,
    /// Stop strings, matched on the decoded text.
    pub stop_strings: Vec<String>,
    /// Stop token ids, dropped from the output.
    pub stop_token_ids: Vec<u32>,
    /// True when the client asked for SSE.
    pub stream: bool,
}

/// Why the generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The model stopped, or a stop string/token matched.
    Stop,
    /// `max_tokens` was reached.
    Length,
    /// The output contains parsed tool calls.
    ToolCalls,
}

impl FinishReason {
    /// The OpenAI `finish_reason` string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
        }
    }
}

/// One step of a generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationEvent {
    /// Text safe to send to the client.
    Delta(String),
    /// The generation is over; `completion_tokens` counts what the engine emitted.
    Done {
        /// Why it ended.
        finish_reason: FinishReason,
        /// Tokens the engine produced for this request.
        completion_tokens: u32,
    },
}

/// Takes an admission permit, or reports the 429 the client should retry on.
///
/// This runs in the handler, before the response head is written, so a saturated
/// server answers `429` even when the client asked for SSE: a stream that starts
/// with an in-band error would be a 200 on the wire.
pub(crate) fn admit(state: &Arc<AppState>) -> Result<Admitted, ApiError> {
    state.admission.try_admit().ok_or_else(|| {
        state.metrics.reject();
        ApiError::too_many_requests(format!(
            "{} requests already in flight (max_concurrent_requests)",
            state.admission.limit()
        ))
    })
}

/// Runs one generation.
///
/// The returned stream holds `permit` and the cancellation guard, so dropping it
/// (a client that goes away mid-stream) cancels the request in the engine and
/// frees the permit.
pub fn generate(
    state: Arc<AppState>,
    plan: GenerationPlan,
    permit: Admitted,
) -> impl Stream<Item = Result<GenerationEvent, ApiError>> + Send + 'static {
    // The instant the request was accepted, before any of the body was polled:
    // `queue_ms` is measured from here to the engine owning the request.
    let accepted = Instant::now();
    let span = tracing::info_span!(
        "request",
        req_id = tracing::field::Empty,
        prompt_tokens = plan.prompt_tokens.len() as u64,
        completion_tokens = tracing::field::Empty,
        ttft_ms = tracing::field::Empty,
        decode_ms = tracing::field::Empty,
        queue_ms = tracing::field::Empty,
        finish_reason = tracing::field::Empty,
    );
    let stream = generation_stream(Arc::clone(&state), plan, permit, span, accepted);
    futures::StreamExt::boxed(stream)
}

/// The generator body.
///
/// `accepted` is the instant the request was accepted (taken in [`generate`]);
/// `queue_ms` is the time from it until the engine owns the request.
///
/// The span is not entered across `await` points (that would leak into whatever
/// the task polls next); it is closed with one event carrying the recorded
/// fields instead, which is what the JSON layer exports.
fn generation_stream(
    state: Arc<AppState>,
    plan: GenerationPlan,
    permit: Admitted,
    span: tracing::Span,
    accepted: Instant,
) -> impl Stream<Item = Result<GenerationEvent, ApiError>> + Send + 'static {
    let body = async_stream::try_stream! {
        // Held for the whole life of the stream: a client that walks away drops the
        // stream, which frees the permit and cancels the request.
        let _permit = permit;

        let req_id = state.next_request_id();
        span.record("req_id", req_id);
        let (sink, mut receiver) = mpsc::channel(TOKEN_CHANNEL_DEPTH);
        let request = GenerateRequest {
            prompt_tokens: plan.prompt_tokens.clone(),
            max_tokens: plan.max_tokens,
            temperature: plan.sampling.temperature,
            top_p: plan.sampling.top_p,
            top_k: plan.sampling.top_k,
            min_p: plan.sampling.min_p,
            seed: plan.sampling.seed,
            stop_token_ids: plan.stop_token_ids.clone(),
            stream: plan.stream,
            priority: 0,
            sink,
        };

        let submitted_at = Instant::now();
        state.metrics.enter_queue();
        // Recorded after `submit` returns: a core blocks inside `submit` while it
        // has no slot, and that block is the queue time the field names. Before,
        // `queue_ms` measured only the code between two clock reads (~4 us) and
        // stopped exactly where the wait began.
        let submitted = state.engine.submit(req_id, request);
        span.record("queue_ms", millis(accepted.elapsed()));
        if let Err(error) = submitted {
            state.metrics.start_of_stream();
            state.metrics.engine_error();
            Err(ApiError::from_engine(error))?;
        }

        state.register_request(req_id);
        let mut cancellation = Cancellation::new(Arc::clone(&state), req_id);
        let deadline = tokio::time::Instant::now()
            + Duration::from_millis(state.config.request_timeout_ms);
        let mut matcher = StopMatcher::new(plan.stop_strings.clone());
        let mut ids: Vec<u32> = Vec::with_capacity(plan.max_tokens.min(4096) as usize);
        let mut emitted_bytes = 0_usize;
        let mut tokens_generated = 0_u32;
        let mut first_token_at: Option<Instant> = None;
        let mut last_token_at: Option<Instant> = None;
        let finish_reason: FinishReason = loop {
            if tokens_generated >= plan.max_tokens {
                break FinishReason::Length;
            }

            let token = next_token(&state, &mut receiver, deadline)
                .await
                .inspect_err(|error| {
                    let status = if error.code == Some("timeout") {
                        "timeout"
                    } else {
                        "error"
                    };
                    cancellation.abandon(status);
                })?;

            let now = Instant::now();
            match first_token_at {
                None => {
                    first_token_at = Some(now);
                    last_token_at = Some(now);
                    state.metrics.start_of_stream();
                    cancellation.mark_streaming();
                    state.metrics.observe_ttft(now.duration_since(submitted_at).as_secs_f64());
                    span.record("ttft_ms", millis(now.duration_since(submitted_at)));
                }
                Some(_) => {
                    if let Some(previous) = last_token_at {
                        state
                            .metrics
                            .observe_inter_token(now.duration_since(previous).as_secs_f64());
                    }
                    last_token_at = Some(now);
                }
            }

            if plan.stop_token_ids.contains(&token.id) {
                break FinishReason::Stop;
            }

            ids.push(token.id);
            tokens_generated += 1;

            let delta = take_decoded_delta(&state.tokenizer, &ids, &mut emitted_bytes);
            if !delta.is_empty() {
                let (emit, stopped) = matcher.feed(&delta);
                if !emit.is_empty() {
                    yield GenerationEvent::Delta(emit);
                }
                if stopped {
                    break FinishReason::Stop;
                }
            }

            if token.is_final() {
                // Reaching the budget exactly ends the generation as "length": that is
                // what a client uses to decide whether to ask for more.
                break if tokens_generated >= plan.max_tokens {
                    FinishReason::Length
                } else {
                    FinishReason::Stop
                };
            }
        };

        let tail = matcher.flush();
        if !tail.is_empty() {
            yield GenerationEvent::Delta(tail);
        }

        let decode_ms = match (first_token_at, last_token_at) {
            (Some(first), Some(last)) => millis(last.duration_since(first)),
            _ => 0.0,
        };
        span.record("completion_tokens", tokens_generated as u64);
        span.record("decode_ms", decode_ms);
        span.record("finish_reason", finish_reason.as_str());

        state
            .metrics
            .finish_request("ok", u64::from(tokens_generated));
        cancellation.settle();
        span.in_scope(|| {
            tracing::info!(tokens = tokens_generated, "generation finished");
        });

        yield GenerationEvent::Done {
            finish_reason,
            completion_tokens: tokens_generated,
        };
    };
    body
}

/// Waits for the next token, bounded by `deadline`.
async fn next_token(
    state: &Arc<AppState>,
    receiver: &mut mpsc::Receiver<EngineToken>,
    deadline: tokio::time::Instant,
) -> Result<EngineToken, ApiError> {
    let received = tokio::time::timeout_at(deadline, receiver.recv()).await;
    let token = match received {
        Err(_) => {
            return Err(ApiError {
                status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                error_type: "server_error",
                code: Some("timeout"),
                message: format!(
                    "the request timed out after {} ms",
                    state.config.request_timeout_ms
                ),
            });
        }
        Ok(None) => {
            state.metrics.engine_error();
            return Err(ApiError::internal(
                "the engine closed the token stream without a final token",
            ));
        }
        Ok(Some(token)) => token,
    };

    if token.is_error() {
        state.metrics.engine_error();
        return Err(ApiError::internal(format!(
            "the engine failed with code {}",
            token.error_code()
        )));
    }
    Ok(token)
}

/// Converts a duration to fractional milliseconds.
fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// The newly decoded text of `ids`, never ending in a partial UTF-8 character.
///
/// A token can end in the middle of a multi-byte character; the decoder then ends
/// the text with a replacement character. Holding it back until the character is
/// complete is what keeps multi-byte output intact on the wire.
fn take_decoded_delta(tokenizer: &Tokenizer, ids: &[u32], emitted_bytes: &mut usize) -> String {
    let text = tokenizer.decode(ids);
    if text.len() <= *emitted_bytes {
        return String::new();
    }
    let mut end = text.len();
    while end > *emitted_bytes && text[..end].ends_with('\u{FFFD}') {
        end -= '\u{FFFD}'.len_utf8();
    }
    if end <= *emitted_bytes {
        return String::new();
    }
    let delta = text[*emitted_bytes..end].to_owned();
    *emitted_bytes = end;
    delta
}

/// Cancels the request in the engine unless the generation finished on its own,
/// and releases the queue-depth gauge if the request never reached its first token.
struct Cancellation {
    state: Arc<AppState>,
    req_id: u64,
    settled: bool,
    streaming: bool,
    drop_status: &'static str,
}

impl Cancellation {
    fn new(state: Arc<AppState>, req_id: u64) -> Self {
        assert!(req_id > 0, "request ids start at one");
        Self {
            state,
            req_id,
            settled: false,
            streaming: false,
            drop_status: "client_gone",
        }
    }

    /// Marks the request as finished, so dropping the guard does not cancel it.
    fn settle(&mut self) {
        self.settled = true;
    }

    /// Records that the request left the queue (the first token arrived).
    fn mark_streaming(&mut self) {
        self.streaming = true;
    }

    /// Keeps the guard armed but reports `status` when it cancels.
    fn abandon(&mut self, status: &'static str) {
        self.settled = false;
        self.drop_status = status;
    }
}

impl Drop for Cancellation {
    fn drop(&mut self) {
        let still_active = self.state.forget_request(self.req_id);
        if !self.streaming {
            self.state.metrics.start_of_stream();
        }
        if self.settled {
            return;
        }
        self.state.metrics.finish_request(self.drop_status, 0);
        if still_active && let Err(error) = self.state.engine.cancel(self.req_id) {
            tracing::warn!(req_id = self.req_id, %error, "cancel failed");
        }
        tracing::info!(
            req_id = self.req_id,
            status = self.drop_status,
            "request cancelled"
        );
    }
}

/// The payload of one non-streaming generation.
#[derive(Debug, Clone)]
pub struct CollectedGeneration {
    /// The decoded text, stop strings removed.
    pub text: String,
    /// Why the generation ended.
    pub finish_reason: FinishReason,
    /// Tokens the engine produced.
    pub completion_tokens: u32,
}

/// Drains a generation into one string, for the buffered (non-SSE) responses.
pub async fn collect(
    stream: impl Stream<Item = Result<GenerationEvent, ApiError>>,
) -> Result<CollectedGeneration, ApiError> {
    let mut stream = std::pin::pin!(stream);
    let mut text = String::new();
    let mut finish_reason = FinishReason::Stop;
    let mut completion_tokens = 0;

    while let Some(event) = stream.next().await {
        match event? {
            GenerationEvent::Delta(delta) => text.push_str(&delta),
            GenerationEvent::Done {
                finish_reason: reason,
                completion_tokens: tokens,
            } => {
                finish_reason = reason;
                completion_tokens = tokens;
            }
        }
    }

    Ok(CollectedGeneration {
        text,
        finish_reason,
        completion_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_tokenizer() -> Tokenizer {
        Tokenizer::from_file("tests/fixtures/model/tokenizer.json").expect("fixture tokenizer")
    }

    #[test]
    fn a_multi_byte_character_is_not_split_across_deltas() {
        let tokenizer = fixture_tokenizer();
        // "é" is two bytes and one byte is one token, so the first decode ends in a
        // replacement character and must emit nothing.
        let mut emitted = 0;
        let first = take_decoded_delta(&tokenizer, &[0xC3], &mut emitted);
        assert_eq!(first, "");
        let second = take_decoded_delta(&tokenizer, &[0xC3, 0xA9], &mut emitted);
        assert_eq!(second, "é");
    }

    #[test]
    fn deltas_are_the_growing_decoded_text() {
        let tokenizer = fixture_tokenizer();
        let ids = tokenizer.encode("Hi!").expect("encodes");
        let mut emitted = 0;
        let mut text = String::new();
        for (index, _) in ids.iter().enumerate() {
            text.push_str(&take_decoded_delta(
                &tokenizer,
                &ids[..=index],
                &mut emitted,
            ));
        }
        assert_eq!(text, "Hi!");
    }
}

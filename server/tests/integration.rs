//! End-to-end tests of the HTTP surface, against a real server on a real socket.
//!
//! The server runs in-process on port 0 with `MockEngine`, whose output is the
//! tokenization of [`MOCK_TEXT`]; every assertion is on *content* (the exact text,
//! the exact token counts), not just status codes. The HTTP client is hand-rolled
//! on a raw `TcpStream` so the tests add no dependency and so a stream can be
//! dropped mid-flight, which is how the disconnect test is possible at all.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use inferno_server::config::{Config, EngineKind, LogConfig, LogFormat, MetricsConfig};
use inferno_server::engine::{Engine, MockConfig, MockEngine};
use inferno_server::server::{AppState, Server};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::oneshot;

/// What `MockEngine` emits; with the fixture tokenizer one token is one byte, so
/// the expected text and the expected token count are both explicit.
const MOCK_TEXT: &str = "Hello world";

/// The fixture model directory (identity tokenizer + the model chat template).
fn fixture_model() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../host/tests/fixtures/model")
}

/// A configuration that serves the fixtures.
fn fixture_config(max_concurrent_requests: u32) -> Config {
    Config {
        listen: "127.0.0.1:0".to_owned(),
        model_path: fixture_model(),
        model_id: "inferno-fixture".to_owned(),
        max_concurrent_requests,
        max_context: 4096,
        kv_bytes: 1 << 30,
        engine: EngineKind::Mock,
        log: LogConfig {
            level: "warn".to_owned(),
            format: LogFormat::Pretty,
        },
        metrics: MetricsConfig { enabled: true },
        request_timeout_ms: 30_000,
        drain_timeout_ms: 2_000,
    }
}

/// A mock engine that emits `MOCK_TEXT`, with the delays a test asks for.
fn scripted_engine(ttft: Duration, per_token: Duration) -> MockEngine {
    let tokenizer = inferno_server::tokenizer::Tokenizer::from_model_directory(fixture_model())
        .expect("the fixture tokenizer loads");
    let tokens = tokenizer
        .encode(MOCK_TEXT)
        .expect("the mock text tokenizes");
    assert_eq!(tokens.len(), MOCK_TEXT.len(), "one token per byte");
    MockEngine::new(MockConfig {
        tokens,
        ttft,
        per_token,
        ..MockConfig::default()
    })
}

/// A running server plus the handles a test needs.
struct Harness {
    address: SocketAddr,
    shutdown: oneshot::Sender<()>,
    task: tokio::task::JoinHandle<()>,
    _state: Arc<AppState>,
}

impl Harness {
    /// Starts a server with `engine` and the given concurrency limit.
    async fn start(engine: Arc<dyn Engine>, max_concurrent_requests: u32) -> Self {
        let state = AppState::with_engine(fixture_config(max_concurrent_requests), engine)
            .expect("the fixture assets load");
        let server = Server::bind(Arc::clone(&state))
            .await
            .expect("the server binds port 0");
        let address = server.local_addr().expect("the bound address is known");
        let (shutdown, shutdown_signal) = oneshot::channel();
        let task = tokio::spawn(async move {
            let _ = server
                .run_until(async move {
                    let _ = shutdown_signal.await;
                })
                .await;
        });
        Self {
            address,
            shutdown,
            task,
            _state: state,
        }
    }

    /// A harness whose engine emits `MOCK_TEXT` quickly.
    async fn quick() -> Self {
        Self::start(
            Arc::new(scripted_engine(
                Duration::from_millis(2),
                Duration::from_millis(1),
            )),
            8,
        )
        .await
    }

    /// Stops the server and waits for it.
    async fn stop(self) {
        let _ = self.shutdown.send(());
        let _ = self.task.await;
    }
}

/// One HTTP response.
#[derive(Debug)]
struct Response {
    status: u16,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

impl Response {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body)
            .unwrap_or_else(|error| panic!("response is not JSON ({error}): {}", self.text()))
    }

    /// The payload of every `data:` line, without the `data: ` prefix.
    fn sse_events(&self) -> Vec<String> {
        self.text()
            .split("\n\n")
            .filter_map(|event| {
                event
                    .lines()
                    .find_map(|line| line.strip_prefix("data: ").map(str::to_owned))
            })
            .collect()
    }
}

/// Issues one request and reads the whole response.
async fn request(
    address: SocketAddr,
    method: &str,
    path: &str,
    body: Option<serde_json::Value>,
) -> Response {
    let mut stream = connect(address, method, path, body).await;
    let mut raw = Vec::new();
    // `Connection: close` makes the server close after the body, which is what
    // lets a `read_to_end` collect a whole SSE response.
    stream
        .read_to_end(&mut raw)
        .await
        .expect("the response is readable");
    parse_response(&raw)
}

/// Issues one request and returns the socket, so a test can drop it early.
async fn connect(
    address: SocketAddr,
    method: &str,
    path: &str,
    body: Option<serde_json::Value>,
) -> TcpStream {
    let mut stream = TcpStream::connect(address).await.expect("connects");
    let payload = body.map(|body| body.to_string()).unwrap_or_default();
    let mut head = format!("{method} {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n");
    if !payload.is_empty() {
        head.push_str("Content-Type: application/json\r\n");
        head.push_str(&format!("Content-Length: {}\r\n", payload.len()));
    }
    head.push_str("\r\n");
    stream
        .write_all(head.as_bytes())
        .await
        .expect("writes head");
    if !payload.is_empty() {
        stream
            .write_all(payload.as_bytes())
            .await
            .expect("writes body");
    }
    stream
}

/// Parses a raw HTTP/1.1 response, de-chunking the body.
fn parse_response(raw: &[u8]) -> Response {
    let separator = find(raw, b"\r\n\r\n").expect("the response has a header block");
    let head = std::str::from_utf8(&raw[..separator]).expect("the head is UTF-8");
    let mut lines = head.split("\r\n");
    let status_line = lines.next().expect("a status line");
    let status = status_line
        .split_whitespace()
        .nth(1)
        .expect("the status code is the second token")
        .parse()
        .expect("the status code is a number");
    let mut headers = HashMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_owned());
        }
    }
    let body = &raw[separator + 4..];
    let body = match headers.get("transfer-encoding").map(String::as_str) {
        Some("chunked") => decode_chunked(body),
        _ => body.to_vec(),
    };
    Response {
        status,
        headers,
        body,
    }
}

/// Decodes a chunked body.
fn decode_chunked(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut cursor = 0;
    while cursor < raw.len() {
        let Some(line_end) = find(&raw[cursor..], b"\r\n") else {
            break;
        };
        let size_line = std::str::from_utf8(&raw[cursor..cursor + line_end]).expect("size line");
        let size_field = size_line.split(';').next().expect("size field").trim();
        let size = usize::from_str_radix(size_field, 16).expect("chunk size is hex");
        cursor += line_end + 2;
        if size == 0 {
            break;
        }
        out.extend_from_slice(&raw[cursor..cursor + size]);
        cursor += size + 2;
    }
    out
}

/// Offset of `needle` in `haystack`.
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// A chat completion body with one user turn.
fn chat_body(max_tokens: u32, stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": "inferno-fixture",
        "messages": [{"role": "user", "content": "count to eleven"}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    })
}

#[tokio::test]
async fn non_stream_chat_completion_returns_the_whole_answer_and_usage() {
    let harness = Harness::quick().await;
    let response = request(
        harness.address,
        "POST",
        "/v1/chat/completions",
        Some(chat_body(11, false)),
    )
    .await;
    assert_eq!(response.status, 200, "{}", response.text());

    let body = response.json();
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "inferno-fixture");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(
        body["choices"][0]["message"]["content"], MOCK_TEXT,
        "the mock emits exactly its scripted text"
    );
    let prompt = body["usage"]["prompt_tokens"].as_u64().expect("prompt");
    let completion = body["usage"]["completion_tokens"]
        .as_u64()
        .expect("completion");
    let total = body["usage"]["total_tokens"].as_u64().expect("total");
    assert_eq!(completion, 11, "one token per byte of {MOCK_TEXT:?}");
    assert!(prompt > 0, "the rendered prompt tokenizes to something");
    assert_eq!(total, prompt + completion, "usage adds up");

    harness.stop().await;
}

#[tokio::test]
async fn stream_chat_completion_matches_the_buffered_answer() {
    let harness = Harness::quick().await;
    let mut request_body = chat_body(11, true);
    request_body["stream_options"] = serde_json::json!({"include_usage": true});
    let streamed = request(
        harness.address,
        "POST",
        "/v1/chat/completions",
        Some(request_body),
    )
    .await;
    assert_eq!(streamed.status, 200, "{}", streamed.text());
    assert_eq!(
        streamed.headers.get("content-type").map(String::as_str),
        Some("text/event-stream"),
        "streaming must use SSE"
    );

    let events = streamed.sse_events();
    assert!(
        events.len() >= 4,
        "expected a role chunk, deltas and a terminator, got {events:?}"
    );
    assert_eq!(events.last().map(String::as_str), Some("[DONE]"));

    let mut content = String::new();
    let mut finish_reason = None;
    let mut usage = None;
    for document in &events[..events.len() - 1] {
        let chunk: serde_json::Value = serde_json::from_str(document).expect("chunk is JSON");
        assert_eq!(chunk["object"], "chat.completion.chunk");
        if let Some(block) = chunk.get("usage") {
            usage = Some(block.clone());
        }
        for choice in chunk["choices"].as_array().expect("choices") {
            if let Some(delta) = choice["delta"]["content"].as_str() {
                content.push_str(delta);
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                finish_reason = Some(reason.to_owned());
            }
        }
    }
    assert_eq!(content, MOCK_TEXT, "deltas must add up to the answer");
    assert_eq!(finish_reason.as_deref(), Some("length"));

    let usage = usage.expect("`stream_options.include_usage` must add a usage chunk");
    assert_eq!(usage["completion_tokens"], 11);
    assert_eq!(
        usage["total_tokens"],
        usage["prompt_tokens"].as_u64().expect("prompt") + 11,
        "the streamed usage must add up too"
    );

    // The buffered path must produce the same text.
    let buffered = request(
        harness.address,
        "POST",
        "/v1/chat/completions",
        Some(chat_body(11, false)),
    )
    .await
    .json();
    assert_eq!(buffered["choices"][0]["message"]["content"], content);

    harness.stop().await;
}

#[tokio::test]
async fn max_tokens_is_honoured_exactly() {
    let harness = Harness::quick().await;
    let body = request(
        harness.address,
        "POST",
        "/v1/chat/completions",
        Some(chat_body(4, false)),
    )
    .await
    .json();
    assert_eq!(
        body["choices"][0]["message"]["content"], "Hell",
        "four byte-tokens of {MOCK_TEXT:?}"
    );
    assert_eq!(body["usage"]["completion_tokens"], 4);
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    harness.stop().await;
}

#[tokio::test]
async fn a_stop_string_truncates_the_answer_and_never_appears() {
    let harness = Harness::quick().await;
    let mut request_body = chat_body(11, false);
    request_body["stop"] = serde_json::json!("lo w");
    let body = request(
        harness.address,
        "POST",
        "/v1/chat/completions",
        Some(request_body),
    )
    .await
    .json();
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .expect("content");
    assert_eq!(content, "Hel", "the stop string truncates the answer");
    assert!(!content.contains("lo w"), "the stop text must not leak");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    harness.stop().await;
}

#[tokio::test]
async fn a_stream_never_leaks_a_stop_string_that_spans_tokens() {
    let harness = Harness::quick().await;
    let mut request_body = chat_body(11, true);
    request_body["stop"] = serde_json::json!(["lo w"]);
    let response = request(
        harness.address,
        "POST",
        "/v1/chat/completions",
        Some(request_body),
    )
    .await;
    let events = response.sse_events();
    let mut content = String::new();
    for document in events.iter().filter(|document| *document != "[DONE]") {
        let chunk: serde_json::Value = serde_json::from_str(document).expect("chunk is JSON");
        for choice in chunk["choices"].as_array().expect("choices") {
            if let Some(delta) = choice["delta"]["content"].as_str() {
                content.push_str(delta);
            }
        }
    }
    assert_eq!(content, "Hel", "the stop string truncates the stream too");
    assert!(
        !response.text().contains("lo w"),
        "the stop text must not leak"
    );
    harness.stop().await;
}

#[tokio::test]
async fn exceeding_max_concurrent_requests_is_a_429() {
    let engine = Arc::new(scripted_engine(
        Duration::from_millis(400),
        Duration::from_millis(100),
    ));
    let harness = Harness::start(engine, 1).await;

    let address = harness.address;
    let first = tokio::spawn(async move {
        request(
            address,
            "POST",
            "/v1/chat/completions",
            Some(chat_body(3, false)),
        )
        .await
    });
    // Give the first request time to reach the engine.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // The second request asks for SSE: admission still runs before the response
    // head, so it must be a 429 and not a 200 with an in-band error.
    let second = request(
        harness.address,
        "POST",
        "/v1/chat/completions",
        Some(chat_body(3, true)),
    )
    .await;
    assert_eq!(
        second.status,
        429,
        "a streamed request must be refused before the stream starts: {}",
        second.text()
    );
    let error = second.json();
    assert_eq!(error["error"]["code"], "too_many_requests");
    assert!(
        error["error"]["message"]
            .as_str()
            .expect("message")
            .contains("max_concurrent_requests"),
        "the message must name the limit: {error}"
    );

    let first = first.await.expect("the first request finishes");
    assert_eq!(first.status, 200, "{}", first.text());

    // The permit must be back: the next request is accepted.
    let third = request(
        harness.address,
        "POST",
        "/v1/chat/completions",
        Some(chat_body(1, false)),
    )
    .await;
    assert_eq!(third.status, 200, "{}", third.text());
    harness.stop().await;
}

#[tokio::test]
async fn dropping_a_stream_cancels_the_request() {
    let engine = Arc::new(scripted_engine(
        Duration::from_millis(5),
        Duration::from_millis(50),
    ));
    let harness = Harness::start(engine.clone(), 4).await;

    let mut stream = connect(
        harness.address,
        "POST",
        "/v1/chat/completions",
        Some(chat_body(11, true)),
    )
    .await;
    // Read the beginning of the stream, then vanish like a closed browser tab.
    let mut buffer = [0_u8; 1024];
    let read = tokio::time::timeout(Duration::from_secs(5), stream.read(&mut buffer))
        .await
        .expect("the first bytes arrive")
        .expect("the stream is readable");
    assert!(read > 0, "the stream started");
    drop(stream);

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while engine.cancels() == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "the engine was never told to cancel"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(engine.cancels(), 1, "exactly one cancel");

    let metrics = request(harness.address, "GET", "/metrics", None)
        .await
        .text();
    assert!(
        metrics.contains("inferno_gpu_mock_cancels 1"),
        "the engine's cancel counter must be re-exported:\n{metrics}"
    );
    harness.stop().await;
}

#[tokio::test]
async fn metrics_expose_the_contracted_names_and_a_histogram() {
    let harness = Harness::quick().await;
    let _ = request(
        harness.address,
        "POST",
        "/v1/chat/completions",
        Some(chat_body(2, false)),
    )
    .await;

    let response = request(harness.address, "GET", "/metrics", None).await;
    assert_eq!(response.status, 200);
    assert_eq!(
        response.headers.get("content-type").map(String::as_str),
        Some("text/plain; version=0.0.4; charset=utf-8")
    );
    let text = response.text();
    for name in [
        "inferno_requests_total",
        "inferno_tokens_generated_total",
        "inferno_ttft_seconds_bucket",
        "inferno_inter_token_seconds_bucket",
        "inferno_queue_depth",
        "inferno_in_flight",
        "inferno_engine_errors_total",
    ] {
        assert!(text.contains(name), "{name} is missing from:\n{text}");
    }
    assert!(
        text.contains("# TYPE inferno_ttft_seconds histogram"),
        "the TTFT histogram must be typed:\n{text}"
    );
    assert!(
        text.contains("inferno_ttft_seconds_count 1"),
        "the one request must be counted:\n{text}"
    );
    assert!(
        text.contains("inferno_requests_total{status=\"ok\"} 1"),
        "the request must be counted as ok:\n{text}"
    );
    assert!(
        text.contains("inferno_tokens_generated_total 2"),
        "two tokens were generated:\n{text}"
    );
    assert!(
        text.contains("inferno_gpu_mock_tokens_emitted"),
        "the engine's own counters must be re-exported:\n{text}"
    );
    harness.stop().await;
}

#[tokio::test]
async fn health_is_alive_and_readiness_tracks_the_engine() {
    // An engine that cannot take work: `/health` is still 200, `/readyz` is 503.
    let cold = Arc::new(MockEngine::new(MockConfig {
        ready: false,
        ..MockConfig::default()
    }));
    let harness = Harness::start(cold, 4).await;

    let health = request(harness.address, "GET", "/health", None).await;
    assert_eq!(health.status, 200, "liveness must not depend on the engine");
    assert_eq!(health.json()["status"], "ok");

    let ready = request(harness.address, "GET", "/readyz", None).await;
    assert_eq!(ready.status, 503, "readiness must reflect the engine");
    assert_eq!(ready.json()["status"], "not_ready");

    // The same endpoint reports ready once the engine can serve.
    let warm = Harness::quick().await;
    let ready = request(warm.address, "GET", "/readyz", None).await;
    assert_eq!(ready.status, 200);
    assert_eq!(ready.json()["status"], "ready");
    assert!(
        ready.json()["engine"]
            .as_str()
            .expect("engine")
            .starts_with("inferno-mock"),
        "readiness reports the engine version: {ready:?}"
    );

    harness.stop().await;
    warm.stop().await;
}

#[tokio::test]
async fn models_lists_the_loaded_model_with_its_context() {
    let harness = Harness::quick().await;
    let body = request(harness.address, "GET", "/v1/models", None)
        .await
        .json();
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "inferno-fixture");
    assert_eq!(body["data"][0]["context_length"], 4096);
    harness.stop().await;
}

#[tokio::test]
async fn legacy_completions_serve_the_same_generator() {
    let harness = Harness::quick().await;
    let response = request(
        harness.address,
        "POST",
        "/v1/completions",
        Some(serde_json::json!({
            "model": "inferno-fixture",
            "prompt": "count to eleven",
            "max_tokens": 11,
        })),
    )
    .await;
    assert_eq!(response.status, 200, "{}", response.text());
    let body = response.json();
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["choices"][0]["text"], MOCK_TEXT);
    assert_eq!(body["usage"]["completion_tokens"], 11);
    assert_eq!(
        body["usage"]["total_tokens"],
        body["usage"]["prompt_tokens"].as_u64().expect("prompt") + 11
    );
    harness.stop().await;
}

#[tokio::test]
async fn bad_parameters_are_a_400_with_the_openai_envelope() {
    let harness = Harness::quick().await;
    let response = request(
        harness.address,
        "POST",
        "/v1/chat/completions",
        Some(serde_json::json!({"messages": [], "model": "inferno-fixture"})),
    )
    .await;
    assert_eq!(response.status, 400, "{}", response.text());
    let body = response.json();
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["message"], "messages must not be empty");
    harness.stop().await;
}

#[tokio::test]
async fn an_unready_engine_rejects_a_request_with_503() {
    let cold = Arc::new(MockEngine::new(MockConfig {
        ready: false,
        ..MockConfig::default()
    }));
    let harness = Harness::start(cold, 4).await;
    let response = request(
        harness.address,
        "POST",
        "/v1/chat/completions",
        Some(chat_body(2, false)),
    )
    .await;
    assert_eq!(response.status, 503, "{}", response.text());
    assert_eq!(response.json()["error"]["code"], "engine_not_ready");
    harness.stop().await;
}

#[tokio::test]
async fn graceful_shutdown_cancels_requests_that_outlive_the_drain() {
    let engine = Arc::new(scripted_engine(
        Duration::from_millis(20),
        Duration::from_millis(200),
    ));
    let state = AppState::with_engine(
        Config {
            // A short drain, so the test does not sit through the default.
            drain_timeout_ms: 100,
            ..fixture_config(4)
        },
        engine.clone(),
    )
    .expect("the fixture assets load");
    let server = Server::bind(Arc::clone(&state))
        .await
        .expect("the server binds port 0");
    let address = server.local_addr().expect("the bound address is known");
    let (shutdown, shutdown_signal) = oneshot::channel();
    let task = tokio::spawn(async move {
        let _ = server
            .run_until(async move {
                let _ = shutdown_signal.await;
            })
            .await;
    });

    // One long stream that will still be running when the shutdown starts.
    let mut stream = connect(
        address,
        "POST",
        "/v1/chat/completions",
        Some(chat_body(11, true)),
    )
    .await;
    let mut buffer = [0_u8; 256];
    let read = tokio::time::timeout(Duration::from_secs(5), stream.read(&mut buffer))
        .await
        .expect("the stream starts")
        .expect("the stream is readable");
    assert!(read > 0, "the stream must have started before the shutdown");

    let _ = shutdown.send(());
    tokio::time::timeout(Duration::from_secs(10), task)
        .await
        .expect("the server shuts down within the drain plus the grace")
        .expect("the server task did not panic");

    assert_eq!(
        engine.cancels(),
        1,
        "the request that outlived the drain must be cancelled, not abandoned"
    );
}

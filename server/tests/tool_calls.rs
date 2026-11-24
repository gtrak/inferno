//! The `<tool_call>` → `tool_calls` mapping, driven over the socket.
//!
//! `template::parse_tool_calls` has unit tests, but until now nothing drove a
//! tool call through HTTP: the report claimed integration coverage of the path
//! without a test that could see the response envelope (`finish_reason`,
//! `message.tool_calls`, the `content: null` rule). `MockEngine` emits the markup
//! the model would, so the assertions are on the exact JSON a client parses.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use inferno_server::config::{Config, EngineKind, LogConfig, LogFormat, MetricsConfig};
use inferno_server::engine::{MockConfig, MockEngine};
use inferno_server::server::{AppState, Server};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::oneshot;

/// A model answer that is nothing but one tool call.
const CALL_ONLY: &str = concat!(
    "<tool_call>\n",
    "<function=get_weather>\n",
    "<parameter=city>\nParis\n</parameter>\n",
    "<parameter=days>\n3\n</parameter>\n",
    "</function>\n",
    "</tool_call>",
);

/// A model answer that leads into a tool call with prose.
const PROSE_THEN_CALL: &str = concat!(
    "Let me check.\n",
    "<tool_call>\n",
    "<function=lookup>\n",
    "</function>\n",
    "</tool_call>",
);

/// The fixture model directory (identity tokenizer + the model chat template).
fn fixture_model() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/model")
}

/// A configuration that serves the fixtures.
fn fixture_config() -> Config {
    Config {
        listen: "127.0.0.1:0".to_owned(),
        model_path: fixture_model(),
        model_id: "inferno-fixture".to_owned(),
        max_concurrent_requests: 4,
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

/// The ids of `text`: with the fixture tokenizer one byte is one token, so the
/// engine's output decodes back to `text` byte for byte.
fn script(text: &str) -> Vec<u32> {
    let tokenizer = inferno_server::tokenizer::Tokenizer::from_model_directory(fixture_model())
        .expect("the fixture tokenizer loads");
    let ids = tokenizer
        .encode(text)
        .unwrap_or_else(|error| panic!("the fixture tokenizer encodes {text:?}: {error}"));
    assert_eq!(ids.len(), text.len(), "one token per byte");
    ids
}

/// A running server whose engine emits `text`.
struct Serving {
    address: SocketAddr,
    shutdown: oneshot::Sender<()>,
    task: tokio::task::JoinHandle<()>,
}

impl Serving {
    async fn start(text: &str) -> Self {
        let engine = MockEngine::new(MockConfig {
            tokens: script(text),
            ttft: Duration::from_millis(2),
            per_token: Duration::from_millis(1),
            ..MockConfig::default()
        });
        let state = AppState::with_engine(fixture_config(), Arc::new(engine))
            .expect("the fixture assets load");
        let server = Server::bind(Arc::clone(&state))
            .await
            .expect("the server binds port 0");
        let address = server.local_addr().expect("the bound address is known");
        let (shutdown, signal) = oneshot::channel();
        let task = tokio::spawn(async move {
            let _ = server
                .run_until(async move {
                    let _ = signal.await;
                })
                .await;
        });
        Self {
            address,
            shutdown,
            task,
        }
    }

    async fn stop(self) {
        let _ = self.shutdown.send(());
        let _ = self.task.await;
    }

    /// `POST /v1/chat/completions` and returns the parsed response body.
    async fn chat(&self, max_tokens: u32) -> serde_json::Value {
        let body = serde_json::json!({
            "model": "inferno-fixture",
            "messages": [{"role": "user", "content": "what is the weather in Paris?"}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        });
        let payload = body.to_string();
        let mut stream = TcpStream::connect(self.address).await.expect("connects");
        let head = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            self.address,
            payload.len()
        );
        stream
            .write_all(head.as_bytes())
            .await
            .expect("writes the head");
        stream
            .write_all(payload.as_bytes())
            .await
            .expect("writes the body");
        let mut raw = Vec::new();
        stream
            .read_to_end(&mut raw)
            .await
            .expect("the response is readable");
        let text = String::from_utf8_lossy(&raw).into_owned();
        let separator = text
            .find("\r\n\r\n")
            .expect("the response has a header block");
        let status = text
            .split_whitespace()
            .nth(1)
            .expect("a status code")
            .to_owned();
        assert_eq!(status, "200", "response: {text}");
        serde_json::from_str(&text[separator + 4..])
            .unwrap_or_else(|error| panic!("body is not JSON ({error}): {text}"))
    }
}

#[tokio::test]
async fn a_tool_call_becomes_tool_calls_and_the_tool_calls_finish_reason() {
    let serving = Serving::start(CALL_ONLY).await;
    let ids = script(CALL_ONLY);
    let response = serving
        .chat(u32::try_from(ids.len()).expect("tokens fit"))
        .await;
    serving.stop().await;

    let choice = &response["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    // The whole answer was the call, so `content` is null rather than an empty
    // string: that is what tells a client there is no prose to render.
    assert!(
        choice["message"]["content"].is_null(),
        "content must be null: {response}"
    );
    let calls = choice["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls is an array");
    assert_eq!(calls.len(), 1, "one call: {response}");
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    let arguments: serde_json::Value = serde_json::from_str(
        calls[0]["function"]["arguments"]
            .as_str()
            .expect("arguments"),
    )
    .expect("arguments are JSON");
    assert_eq!(arguments, serde_json::json!({"city": "Paris", "days": "3"}));
    assert_eq!(response["usage"]["completion_tokens"], ids.len() as u64);
}

#[tokio::test]
async fn prose_before_a_tool_call_stays_content() {
    let serving = Serving::start(PROSE_THEN_CALL).await;
    let ids = script(PROSE_THEN_CALL);
    let response = serving
        .chat(u32::try_from(ids.len()).expect("tokens fit"))
        .await;
    serving.stop().await;

    let choice = &response["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    assert_eq!(choice["message"]["content"], "Let me check.");
    let calls = choice["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls is an array");
    assert_eq!(calls.len(), 1, "one call: {response}");
    assert_eq!(calls[0]["function"]["name"], "lookup");
    let arguments: serde_json::Value = serde_json::from_str(
        calls[0]["function"]["arguments"]
            .as_str()
            .expect("arguments"),
    )
    .expect("arguments are JSON");
    assert_eq!(arguments, serde_json::json!({}));
}

//! `queue_ms` measures the wait for the engine to take the request.
//!
//! The engine's `submit` is where a saturated core blocks, so the queue time is
//! accept → `submit` returning. It used to stop at the call, which measured
//! microseconds no matter how long the core held the request. The double below
//! separates the two waits on purpose: it takes 25 ms to accept and a further
//! 400 ms to emit the first token, so a `queue_ms` that measured the whole request
//! would be caught by the upper bound and one that measured nothing by the lower.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::StreamExt;
use inferno_server::api::generate::{GenerationPlan, SamplingParams, generate};
use inferno_server::config::{Config, EngineKind, LogConfig, LogFormat, MetricsConfig};
use inferno_server::engine::{
    Engine, EngineError, EngineToken, GenerateRequest, TOKEN_FLAG_FINAL, TOKEN_FLAG_FIRST,
};
use inferno_server::server::AppState;
use tracing::field::{Field, Visit};
use tracing::subscriber::with_default;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

/// How long the double holds `submit` before it owns the request.
const ACCEPT_AFTER: Duration = Duration::from_millis(25);

/// How long after accepting it takes the double to emit its first token.
const FIRST_TOKEN_AFTER: Duration = Duration::from_millis(400);

/// The fixture model directory (identity tokenizer + the model chat template).
fn fixture_model() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/model")
}

/// An engine that blocks in `submit`, the way a core with a full inbox does.
struct DelayedEngine {
    accept_after: Duration,
    first_token_after: Duration,
}

impl Engine for DelayedEngine {
    fn version(&self) -> String {
        "delayed-test 0.0.0".to_owned()
    }

    fn submit(&self, _req_id: u64, req: GenerateRequest) -> Result<(), EngineError> {
        std::thread::sleep(self.accept_after);
        let sink = req.sink;
        let wait = self.first_token_after;
        tokio::spawn(async move {
            tokio::time::sleep(wait).await;
            let token = EngineToken {
                id: u32::from(b'A'),
                flags: TOKEN_FLAG_FIRST | TOKEN_FLAG_FINAL,
            };
            let _ = sink.send(token).await;
        });
        Ok(())
    }

    fn cancel(&self, _req_id: u64) -> Result<(), EngineError> {
        Ok(())
    }

    fn metrics_json(&self) -> String {
        String::from("{}")
    }
}

/// Every `queue_ms` value recorded on the spans of this process.
#[derive(Debug, Default)]
struct QueueFields {
    values: Mutex<Vec<f64>>,
}

impl QueueFields {
    fn record(&self, value: f64) {
        if let Ok(mut values) = self.values.lock() {
            values.push(value);
        }
    }

    /// The largest recorded value, which is the only request in this test file.
    fn largest(&self) -> Option<f64> {
        self.values
            .lock()
            .ok()
            .and_then(|values| values.iter().copied().reduce(f64::max))
    }
}

/// Captures the `queue_ms` field of the request spans.
struct QueueFieldLayer {
    fields: Arc<QueueFields>,
}

impl<S: tracing::Subscriber> Layer<S> for QueueFieldLayer {
    fn on_record(
        &self,
        _span: &tracing::span::Id,
        values: &tracing::span::Record<'_>,
        _ctx: Context<'_, S>,
    ) {
        values.record(&mut QueueFieldVisitor {
            fields: &self.fields,
        });
    }
}

/// Picks the `queue_ms` field out of a recorded field set.
struct QueueFieldVisitor<'a> {
    fields: &'a QueueFields,
}

impl Visit for QueueFieldVisitor<'_> {
    fn record_f64(&mut self, field: &Field, value: f64) {
        if field.name() == "queue_ms" {
            self.fields.record(value);
        }
    }

    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
}

/// A configuration that serves the fixtures on the delayed engine.
fn fixture_config() -> Config {
    Config {
        listen: "127.0.0.1:0".to_owned(),
        model_path: fixture_model(),
        model_id: "inferno-fixture".to_owned(),
        max_concurrent_requests: 1,
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

/// The shortest plan that generates: one token, no sampling.
fn fixture_plan() -> GenerationPlan {
    GenerationPlan {
        prompt_tokens: vec![1, 2, 3],
        max_tokens: 1,
        sampling: SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seed: 0,
        },
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        stream: false,
    }
}

#[test]
fn queue_ms_is_the_wait_for_the_engine_to_own_the_request() {
    let fields = Arc::new(QueueFields::default());
    let subscriber = tracing_subscriber::registry().with(QueueFieldLayer {
        fields: Arc::clone(&fields),
    });
    // A current-thread runtime, because the capturing subscriber is thread local
    // and the whole generation has to be polled on the thread that set it.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a current-thread runtime");

    let events = with_default(subscriber, || {
        runtime.block_on(async {
            let engine = DelayedEngine {
                accept_after: ACCEPT_AFTER,
                first_token_after: FIRST_TOKEN_AFTER,
            };
            let state = AppState::with_engine(fixture_config(), Arc::new(engine))
                .expect("the fixture assets load");
            let permit = state.admission.try_admit().expect("the first request fits");
            let started = Instant::now();
            let events = generate(state, fixture_plan(), permit)
                .collect::<Vec<_>>()
                .await;
            (events, started.elapsed())
        })
    });
    let (events, elapsed) = events;

    let failures: Vec<_> = events
        .iter()
        .filter(|event| event.is_err())
        .map(|event| format!("{event:?}"))
        .collect();
    assert!(
        failures.is_empty(),
        "the generation must not fail: {failures:?}"
    );
    assert!(
        events.iter().any(|event| matches!(
            event,
            Ok(inferno_server::api::generate::GenerationEvent::Done {
                completion_tokens: 1,
                ..
            })
        )),
        "the engine's single token must reach the stream: {events:?}"
    );

    let queue_ms = fields
        .largest()
        .expect("the request span recorded queue_ms");
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    // Lower bound: the acceptance wait is inside the measurement (it used to be
    // entirely outside it, which read as ~0.004 ms).
    assert!(
        queue_ms >= ACCEPT_AFTER.as_secs_f64() * 1000.0 * 0.8,
        "queue_ms {queue_ms} missed the {} ms the engine held submit",
        ACCEPT_AFTER.as_secs_f64() * 1000.0
    );
    // Upper bound: it is the queue, not the request. The token alone arrived
    // `FIRST_TOKEN_AFTER` after acceptance, so a field that measured the whole
    // generation would be at least that large.
    assert!(
        queue_ms < FIRST_TOKEN_AFTER.as_secs_f64() * 1000.0 * 0.5,
        "queue_ms {queue_ms} measured more than the queue (whole request {elapsed_ms} ms)"
    );
}

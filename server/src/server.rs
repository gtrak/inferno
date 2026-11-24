//! Server assembly: model assets, engine, metrics, admission limit and the
//! lifecycle (bind, serve, drain, shut the engine down).

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::net::TcpListener;

use crate::admission::Admission;
use crate::api;
use crate::config::{Config, EngineKind};
use crate::engine::{Engine, MockConfig, MockEngine, ZigEngine};
use crate::metrics::Metrics;
use crate::template::ChatTemplate;
use crate::tokenizer::Tokenizer;

/// How long cancelled requests get to unwind before the engine is shut down.
const SHUTDOWN_GRACE: Duration = Duration::from_millis(250);

/// What `--engine mock` says before the tokenizer splits it into tokens.
const MOCK_SENTENCE: &str =
    "inferno mock engine: one two three four five six seven eight nine ten.";

/// Starting the server failed.
#[derive(Debug, thiserror::Error)]
pub enum StartupError {
    /// `tokenizer.json` or `chat_template.jinja` is missing or broken.
    #[error(transparent)]
    Assets(#[from] AssetError),
    /// The engine refused to initialise.
    #[error("engine failed to initialise: {0}")]
    Engine(#[from] crate::engine::EngineError),
    /// The listen address is unusable.
    #[error("cannot listen on {address}: {source}")]
    Listen {
        /// The configured address.
        address: String,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// The HTTP server itself failed.
    #[error("http server failed: {0}")]
    Serve(#[from] std::io::Error),
}

impl From<crate::tokenizer::TokenizerError> for StartupError {
    fn from(error: crate::tokenizer::TokenizerError) -> Self {
        Self::Assets(AssetError::Tokenizer(error))
    }
}

impl From<crate::template::TemplateError> for StartupError {
    fn from(error: crate::template::TemplateError) -> Self {
        Self::Assets(AssetError::Template(error))
    }
}

/// Model asset loading failed.
#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    /// The tokenizer is unusable.
    #[error(transparent)]
    Tokenizer(#[from] crate::tokenizer::TokenizerError),
    /// The chat template is unusable.
    #[error(transparent)]
    Template(#[from] crate::template::TemplateError),
}

/// Everything the handlers share.
pub struct AppState {
    /// Effective configuration.
    pub config: Arc<Config>,
    /// The engine (Zig core or mock).
    pub engine: Arc<dyn Engine>,
    /// The model tokenizer.
    pub tokenizer: Arc<Tokenizer>,
    /// The model chat template.
    pub template: Arc<ChatTemplate>,
    /// Prometheus metrics.
    pub metrics: Arc<Metrics>,
    /// In-flight budget.
    pub admission: Arc<Admission>,
    /// Unix time the process started, reported as `created` in responses.
    pub created: u64,
    next_request_id: AtomicU64,
    /// Request ids the engine still knows; the shutdown cancels what is left.
    active: Mutex<HashSet<u64>>,
}

impl AppState {
    /// Loads the model assets and builds the engine the configuration names.
    pub fn build(config: Config) -> Result<Arc<Self>, StartupError> {
        let tokenizer = Arc::new(Tokenizer::from_model_directory(&config.model_path)?);
        let template = Arc::new(ChatTemplate::from_file(
            config.model_path.join("chat_template.jinja"),
        )?);
        let engine: Arc<dyn Engine> = match config.engine {
            EngineKind::Mock => Arc::new(mock_engine(&tokenizer)),
            EngineKind::Zig => Arc::new(ZigEngine::initialise(&core_config_json(&config))?),
        };
        Ok(Self::assemble(config, tokenizer, template, engine))
    }

    /// Loads the model assets and drives a caller-supplied engine.
    pub fn with_engine(config: Config, engine: Arc<dyn Engine>) -> Result<Arc<Self>, StartupError> {
        let tokenizer = Arc::new(Tokenizer::from_model_directory(&config.model_path)?);
        let template = Arc::new(ChatTemplate::from_file(
            config.model_path.join("chat_template.jinja"),
        )?);
        Ok(Self::assemble(config, tokenizer, template, engine))
    }

    /// Wires the shared state together.
    fn assemble(
        config: Config,
        tokenizer: Arc<Tokenizer>,
        template: Arc<ChatTemplate>,
        engine: Arc<dyn Engine>,
    ) -> Arc<Self> {
        let metrics = Arc::new(Metrics::new());
        let admission = Arc::new(Admission::new(
            config.max_concurrent_requests,
            metrics.in_flight_gauge(),
        ));
        Arc::new(Self {
            config: Arc::new(config),
            engine,
            tokenizer,
            template,
            metrics,
            admission,
            created: unix_now(),
            next_request_id: AtomicU64::new(1),
            active: Mutex::new(HashSet::new()),
        })
    }

    /// A fresh request id, unique for the lifetime of the process.
    pub fn next_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Remembers a request the engine is working on.
    pub(crate) fn register_request(&self, req_id: u64) {
        if let Ok(mut active) = self.active.lock() {
            active.insert(req_id);
        }
    }

    /// Forgets a request that is over; true when it was still registered.
    pub(crate) fn forget_request(&self, req_id: u64) -> bool {
        self.active
            .lock()
            .map(|mut active| active.remove(&req_id))
            .unwrap_or(false)
    }

    /// Cancels every request still in flight and reports how many there were.
    ///
    /// This is the last step of the drain: requests that outlived
    /// `drain_timeout_ms` are cancelled rather than left to a forced teardown.
    pub fn cancel_active(&self) -> usize {
        let ids: Vec<u64> = self
            .active
            .lock()
            .map(|active| active.iter().copied().collect())
            .unwrap_or_default();
        for req_id in &ids {
            if let Err(error) = self.engine.cancel(*req_id) {
                tracing::warn!(req_id, %error, "cancel during shutdown failed");
            }
        }
        // The registrations go too: each request's own guard must not cancel twice.
        if let Ok(mut active) = self.active.lock() {
            for req_id in &ids {
                active.remove(req_id);
            }
        }
        ids.len()
    }

    /// The default `max_tokens` when the client does not ask for one.
    pub fn default_max_tokens(&self, prompt_tokens: usize) -> u32 {
        // Half the remaining context, so a request can never be cut off by the
        // context window alone; the client keeps control by sending `max_tokens`.
        let remaining = (self.config.max_context as usize).saturating_sub(prompt_tokens);
        u32::try_from(remaining.max(1) / 2)
            .unwrap_or(u32::MAX)
            .max(1)
    }
}

/// A mock engine whose output is the tokenization of [`MOCK_SENTENCE`].
fn mock_engine(tokenizer: &Tokenizer) -> MockEngine {
    let tokens = tokenizer
        .encode(MOCK_SENTENCE)
        .unwrap_or_else(|_| vec![1, 2, 3, 4, 5, 6, 7, 8]);
    assert!(!tokens.is_empty(), "the mock sentence must tokenize");
    MockEngine::new(MockConfig {
        tokens,
        ..MockConfig::default()
    })
}

/// The engine configuration document handed to `inferno_core_init`.
fn core_config_json(config: &Config) -> String {
    let mut document: HashMap<&str, String> = HashMap::new();
    document.insert("model_path", config.model_path.display().to_string());
    document.insert("model_id", config.model_id.clone());
    document.insert("max_context", config.max_context.to_string());
    document.insert("kv_bytes", config.kv_bytes.to_string());
    document.insert(
        "max_concurrent_requests",
        config.max_concurrent_requests.to_string(),
    );
    serde_json::to_string(&document).expect("a map of strings serialises")
}

/// Wall-clock seconds since the Unix epoch.
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_secs())
        .unwrap_or(0)
}

/// A bound listener plus its state.
pub struct Server {
    listener: TcpListener,
    state: Arc<AppState>,
}

impl Server {
    /// Binds the configured address.
    pub async fn bind(state: Arc<AppState>) -> Result<Self, StartupError> {
        let address = state
            .config
            .socket_address()
            .map_err(|error| StartupError::Listen {
                address: state.config.listen.clone(),
                source: std::io::Error::new(std::io::ErrorKind::InvalidInput, error.to_string()),
            })?;
        let listener = TcpListener::bind(address)
            .await
            .map_err(|source| StartupError::Listen {
                address: state.config.listen.clone(),
                source,
            })?;
        Ok(Self { listener, state })
    }

    /// The address actually bound (useful when the config asked for port 0).
    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// The shared state, so tests can inspect metrics and counters.
    pub fn state(&self) -> Arc<AppState> {
        Arc::clone(&self.state)
    }

    /// Serves until `SIGINT`/`SIGTERM`.
    pub async fn run(self) -> Result<(), StartupError> {
        self.run_until(shutdown_signal()).await
    }

    /// Serves until `shutdown` resolves, then drains and shuts the engine down.
    ///
    /// The drain runs concurrently with the HTTP server, not after it: hyper only
    /// returns from `serve` once every connection has finished, and a stream that
    /// ignores the shutdown would otherwise hold the process open forever. So the
    /// drain waits `drain_timeout_ms` for in-flight requests, cancels whatever is
    /// left, gives them a moment to unwind, and only then shuts the engine down.
    pub async fn run_until(
        self,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> Result<(), StartupError> {
        let (signal_sent, signal_received) = tokio::sync::oneshot::channel();
        let shutdown = async move {
            shutdown.await;
            let _ = signal_sent.send(());
        };

        let state = Arc::clone(&self.state);
        let drain_timeout = Duration::from_millis(self.state.config.drain_timeout_ms);
        let drain = tokio::spawn(async move {
            if signal_received.await.is_err() {
                return;
            }
            tracing::info!(
                in_flight = state.admission.in_flight(),
                "no longer accepting requests; draining"
            );
            state.admission.wait_idle(drain_timeout).await;
            let cancelled = state.cancel_active();
            if cancelled > 0 {
                tracing::warn!(cancelled, "cancelling requests that outlived the drain");
                tokio::time::sleep(SHUTDOWN_GRACE).await;
            }
            state.engine.shutdown();
            tracing::info!("engine shut down");
        });

        let router = api::router(Arc::clone(&self.state));
        let served = axum::serve(self.listener, router)
            .with_graceful_shutdown(shutdown)
            .await;
        let _ = drain.await;
        served?;
        Ok(())
    }
}

/// Resolves on `SIGINT` or `SIGTERM`.
pub async fn shutdown_signal() {
    let interrupt = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::error!(%error, "cannot listen for SIGINT");
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => {
                tracing::error!(%error, "cannot listen for SIGTERM");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = interrupt => tracing::info!("SIGINT received"),
        () = terminate => tracing::info!("SIGTERM received"),
    }
}

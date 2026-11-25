//! [`MockEngine`]: a deterministic engine so the whole HTTP surface is testable
//! (and benchmarkable) without a GPU.
//!
//! It emits a caller-supplied token sequence after an artificial TTFT delay and
//! an artificial per-token delay, honours `max_tokens` exactly (it cycles the
//! sequence when the client asks for more tokens than it holds) and counts
//! cancellations, which is how the disconnect test proves that dropping a stream
//! cancels the request.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use super::{
    Engine, EngineError, EngineToken, GenerateRequest, TOKEN_FLAG_FINAL, TOKEN_FLAG_FIRST,
};

/// How the mock engine behaves.
#[derive(Debug, Clone)]
pub struct MockConfig {
    /// Reported by [`Engine::version`].
    pub version: String,
    /// Tokens emitted, cycled when `max_tokens` is larger.
    pub tokens: Vec<u32>,
    /// Delay before the first token.
    pub ttft: Duration,
    /// Delay between tokens.
    pub per_token: Duration,
    /// False makes [`Engine::ready`] false, so `/readyz` can be exercised.
    pub ready: bool,
}

impl Default for MockConfig {
    fn default() -> Self {
        Self {
            version: "inferno-mock 0.1.0".to_owned(),
            tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
            ttft: Duration::from_millis(5),
            per_token: Duration::from_millis(1),
            ready: true,
        }
    }
}

/// Counters the mock keeps so tests can assert on engine-side effects.
#[derive(Debug, Default)]
struct MockCounters {
    emitted: AtomicU64,
    cancels: AtomicU64,
    cancelled: Mutex<HashSet<u64>>,
}

impl MockCounters {
    fn is_cancelled(&self, req_id: u64) -> bool {
        self.cancelled
            .lock()
            .map(|cancelled| cancelled.contains(&req_id))
            .unwrap_or(false)
    }

    fn settle(&self, req_id: u64) {
        if let Ok(mut cancelled) = self.cancelled.lock() {
            cancelled.remove(&req_id);
        }
    }
}

/// A deterministic, GPU-free [`Engine`].
pub struct MockEngine {
    config: MockConfig,
    counters: Arc<MockCounters>,
}

impl MockEngine {
    /// Builds a mock engine; `config.tokens` must not be empty.
    pub fn new(config: MockConfig) -> Self {
        assert!(!config.tokens.is_empty(), "a mock engine needs tokens");
        Self {
            config,
            counters: Arc::new(MockCounters::default()),
        }
    }

    /// How many tokens the generator has produced so far.
    pub fn emitted(&self) -> u64 {
        self.counters.emitted.load(Ordering::Relaxed)
    }

    /// How many requests were cancelled (client disconnect or timeout).
    pub fn cancels(&self) -> u64 {
        self.counters.cancels.load(Ordering::Relaxed)
    }
}

impl Engine for MockEngine {
    fn version(&self) -> String {
        self.config.version.clone()
    }

    fn submit(&self, req_id: u64, req: GenerateRequest) -> Result<(), EngineError> {
        if !self.config.ready {
            return Err(EngineError::NotInitialised);
        }
        let count = req.max_tokens;
        if count == 0 {
            return Ok(());
        }
        let config = self.config.clone();
        let counters = Arc::clone(&self.counters);
        let sink = req.sink.clone();
        tokio::spawn(async move {
            tokio::time::sleep(config.ttft).await;
            for index in 0..count {
                if counters.is_cancelled(req_id) {
                    break;
                }
                let mut flags = 0;
                if index == 0 {
                    flags |= TOKEN_FLAG_FIRST;
                }
                if index + 1 == count {
                    flags |= TOKEN_FLAG_FINAL;
                }
                let token = EngineToken {
                    id: config.tokens[index as usize % config.tokens.len()],
                    flags,
                };
                if sink.send(token).await.is_err() {
                    // The consumer is gone: the request is over.
                    break;
                }
                counters.emitted.fetch_add(1, Ordering::Relaxed);
                if index + 1 < count {
                    tokio::time::sleep(config.per_token).await;
                }
            }
            counters.settle(req_id);
        });
        Ok(())
    }

    fn cancel(&self, req_id: u64) -> Result<(), EngineError> {
        if let Ok(mut cancelled) = self.counters.cancelled.lock() {
            cancelled.insert(req_id);
        }
        self.counters.cancels.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn metrics_json(&self) -> String {
        let document = serde_json::json!({
            "inferno_gpu_mock_tokens_emitted": self.emitted(),
            "inferno_gpu_mock_cancels": self.cancels(),
        });
        document.to_string()
    }

    fn ready(&self) -> bool {
        self.config.ready
    }
}

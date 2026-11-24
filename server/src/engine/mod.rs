//! The engine boundary: everything the HTTP layer needs from the Zig core.
//!
//! The four required methods of [`Engine`] match the contract in
//! `.agents/tasks/issue-007-rust-server.md`. [`Engine::ready`] and
//! [`Engine::shutdown`] are defaulted additions that the readiness probe and the
//! graceful shutdown need; they have no effect on the C ABI.

pub mod ffi;
mod mock;
pub mod probe;
mod zig;

pub use mock::{MockConfig, MockEngine};
pub use zig::ZigEngine;

use tokio::sync::mpsc;

/// `flags` bit 1 of `inferno_token_cb`: the token is the last one of the request.
pub const TOKEN_FLAG_FINAL: u32 = 1;
/// `flags` bit 2 of `inferno_token_cb`: the engine failed; `id` carries the code.
pub const TOKEN_FLAG_ERROR: u32 = 2;
/// `flags` bit 4 of `inferno_token_cb`: the first token, the TTFT mark.
pub const TOKEN_FLAG_FIRST: u32 = 4;

/// Depth of the per-request token channel.
///
/// Deep enough that a healthy consumer never fills it, shallow enough that a
/// client which stops reading cannot pin memory for a whole generation.
pub const TOKEN_CHANNEL_DEPTH: usize = 1024;

/// One token as the engine callback delivers it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineToken {
    /// Vocabulary id of the token, or the error code when [`Self::is_error`].
    pub id: u32,
    /// Bit set: see [`TOKEN_FLAG_FINAL`], [`TOKEN_FLAG_ERROR`], [`TOKEN_FLAG_FIRST`].
    pub flags: u32,
}

impl EngineToken {
    /// True for the last token of the request.
    pub fn is_final(&self) -> bool {
        self.flags & TOKEN_FLAG_FINAL != 0
    }

    /// True when the engine reports a failure instead of a token.
    pub fn is_error(&self) -> bool {
        self.flags & TOKEN_FLAG_ERROR != 0
    }

    /// True for the first token of the request (the TTFT mark).
    pub fn is_first(&self) -> bool {
        self.flags & TOKEN_FLAG_FIRST != 0
    }

    /// The engine error code carried in [`Self::id`] when [`Self::is_error`].
    pub fn error_code(&self) -> i32 {
        self.id as i32
    }
}

/// Where an engine pushes the tokens of one request.
pub type TokenSink = mpsc::Sender<EngineToken>;

/// One generation request: the `req_json` document `inferno_core_submit` parses,
/// plus the channel the engine pushes the tokens into.
pub struct GenerateRequest {
    /// Tokenized prompt, already chat-templated.
    pub prompt_tokens: Vec<u32>,
    /// Hard upper bound on emitted tokens.
    pub max_tokens: u32,
    /// Sampling temperature (`0.0` is greedy).
    pub temperature: f32,
    /// Nucleus sampling threshold.
    pub top_p: f32,
    /// Top-k cut-off (`0` disables it).
    pub top_k: u32,
    /// Min-p cut-off.
    pub min_p: f32,
    /// Sampling seed.
    pub seed: u64,
    /// Token ids that end the generation.
    pub stop_token_ids: Vec<u32>,
    /// True when the client asked for `text/event-stream`.
    pub stream: bool,
    /// Scheduler priority; `0` is the default.
    pub priority: i32,
    /// Channel the engine pushes [`EngineToken`]s into.
    pub sink: TokenSink,
}

impl GenerateRequest {
    /// The `req_json` document of the engine contract.
    pub fn wire_json(&self) -> String {
        let document = serde_json::json!({
            "prompt_tokens": self.prompt_tokens,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "seed": self.seed,
            "stop_token_ids": self.stop_token_ids,
            "stream": self.stream,
            "priority": self.priority,
        });
        document.to_string()
    }
}

impl std::fmt::Debug for GenerateRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GenerateRequest")
            .field("prompt_tokens", &self.prompt_tokens.len())
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("top_k", &self.top_k)
            .field("min_p", &self.min_p)
            .field("seed", &self.seed)
            .field("stop_token_ids", &self.stop_token_ids)
            .field("stream", &self.stream)
            .field("priority", &self.priority)
            .finish_non_exhaustive()
    }
}

/// Why an engine call failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EngineError {
    /// `inferno_core_submit` returned a negative code.
    #[error("engine rejected the request with code {0}")]
    Rejected(i32),
    /// The engine has no core handle (not initialised, or already shut down).
    #[error("engine is not initialised")]
    NotInitialised,
    /// `cancel` on a request the engine does not know.
    #[error("unknown request id {0}")]
    UnknownRequest(u64),
    /// Anything else, with the message the core or the wrapper produced.
    #[error("{0}")]
    Failed(String),
}

/// The Zig core, as the control plane sees it.
pub trait Engine: Send + Sync {
    /// Version string of the linked core.
    fn version(&self) -> String;

    /// Hands one request to the engine; tokens arrive on `req.sink`.
    fn submit(&self, req_id: u64, req: GenerateRequest) -> Result<(), EngineError>;

    /// Cancels one in-flight request.
    fn cancel(&self, req_id: u64) -> Result<(), EngineError>;

    /// The engine's own `metrics_json` document (parsed and re-exported at `/metrics`).
    fn metrics_json(&self) -> String;

    /// True when the engine can serve requests (`/readyz`).
    fn ready(&self) -> bool {
        true
    }

    /// Releases the engine. Called once, after the server drained.
    fn shutdown(&self) {}
}

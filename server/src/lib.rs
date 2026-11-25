//! `inferno-server`: the transport layer of the inferno inference engine.
//!
//! HTTP/SSE only: the OpenAI-compatible surface (`api/`), the server assembly
//! (`server.rs`) and the binary (`main.rs`).
//!
//! Everything that is integration or ownership rather than transport - request
//! admission, the engine boundary, the tokenizer and chat template, stop
//! strings, configuration, logging, Prometheus metrics - lives in the
//! `inferno-host` crate and is re-exported here, so transport code and the
//! integration tests keep one spelling for it.

pub mod api;
pub mod server;

pub use inferno_host::{admission, config, engine, metrics, stop, telemetry, template, tokenizer};

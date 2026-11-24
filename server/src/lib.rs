//! `inferno-server`: the Rust control plane of the inferno inference engine.
//!
//! The hot path is Zig (`core/`); everything here is orchestration: the
//! OpenAI-compatible HTTP surface, the tokenizer and chat template, admission
//! control, configuration, logging and Prometheus metrics.

pub mod admission;
pub mod api;
pub mod config;
pub mod engine;
pub mod metrics;
pub mod server;
pub mod stop;
pub mod telemetry;
pub mod template;
pub mod tokenizer;

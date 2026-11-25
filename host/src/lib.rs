//! `inferno-host`: the integration and ownership layer of the inferno engine.
//!
//! Everything the engine needs from Rust that is not transport: request
//! admission, the engine boundary (the C ABI, the handle types, `ZigEngine`,
//! `MockEngine`, the device probe), the tokenizer and chat template, stop
//! strings, configuration, logging and Prometheus metrics.
//!
//! The hot path is Zig (`core/`); this crate is the layer that decides what to
//! run and owns the lifetimes, and it is where the load pipeline, sequence
//! control and lifetime policy grow. It links `libinferno_core.a` (`build.rs`).
//!
//! Transport lives in `inferno-server`, which depends on this crate. Nothing
//! here may depend on `axum` in return.

pub mod admission;
pub mod config;
pub mod engine;
pub mod metrics;
pub mod stop;
pub mod telemetry;
pub mod template;
pub mod tokenizer;

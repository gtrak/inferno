//! Logging setup: one human layer or one JSON layer, chosen by configuration.

use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::config::{LogConfig, LogFormat};

/// Installing the subscriber failed.
#[derive(Debug, thiserror::Error)]
pub enum TelemetryError {
    /// The log filter does not parse.
    #[error("log filter is invalid: {0}")]
    Filter(#[from] tracing_subscriber::filter::ParseError),
    /// A subscriber was already installed (a second `init` call).
    #[error("cannot install the tracing subscriber: {0}")]
    Install(#[from] tracing_subscriber::util::TryInitError),
}

/// Installs the global subscriber.
///
/// `INFERNO_LOG` (or `RUST_LOG`) wins over the configured level when set, which is
/// what makes a field investigation possible without editing the config file.
pub fn init(log: &LogConfig) -> Result<(), TelemetryError> {
    let filter = match std::env::var("INFERNO_LOG").or_else(|_| std::env::var("RUST_LOG")) {
        Ok(value) => EnvFilter::try_new(value),
        Err(_) => EnvFilter::try_new(&log.level),
    }?;

    let registry = tracing_subscriber::registry().with(filter);
    match log.format {
        LogFormat::Json => registry
            .with(
                tracing_subscriber::fmt::layer()
                    .json()
                    .with_current_span(true)
                    .with_span_list(false)
                    .flatten_event(true),
            )
            .try_init()
            .map_err(TelemetryError::Install),
        LogFormat::Pretty => registry
            .with(tracing_subscriber::fmt::layer().with_ansi(false))
            .try_init()
            .map_err(TelemetryError::Install),
    }
}

//! Configuration: a TOML file, environment overrides and the command line, in
//! that order of precedence.
//!
//! `deny_unknown_fields` is deliberate: a misspelled key would otherwise be
//! silently ignored and the server would start with a default the operator did
//! not ask for. Starting refuses instead.

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::ValueEnum;
use figment::Figment;
use figment::providers::{Env, Format, Serialized, Toml};
use serde::{Deserialize, Serialize};

/// Prefix of the environment overrides (`INFERNO_LOG__LEVEL=debug`).
const ENV_PREFIX: &str = "INFERNO_";

/// The bare `INFERNO_LOG` name, which is *not* a `[log]` configuration key.
///
/// `telemetry::init` reads it as the log filter override it documents, so the
/// config layer must not fold it onto the `log` struct: `INFERNO_LOG=debug`
/// used to abort startup with "expected struct LogConfig". `INFERNO_LOG__LEVEL`
/// and `INFERNO_LOG__FORMAT` are unaffected - only the exact bare name is dropped.
const LOG_FILTER_VARIABLE: &str = "log";

/// Where a configuration value came from, for the startup log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueSource {
    /// Built-in default.
    Default,
    /// The TOML file.
    File,
    /// An `INFERNO_*` variable.
    Environment,
    /// The command line.
    CommandLine,
}

/// Which engine the server drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum EngineKind {
    /// The real Zig core through the C ABI.
    Zig,
    /// A deterministic GPU-free engine, for tests and benchmarks.
    Mock,
}

impl std::fmt::Display for EngineKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zig => write!(formatter, "zig"),
            Self::Mock => write!(formatter, "mock"),
        }
    }
}

/// Log output shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    /// One JSON object per event.
    Json,
    /// Human readable lines.
    Pretty,
}

/// `[log]` section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogConfig {
    /// `tracing` filter directive, e.g. `info` or `inferno_server=debug,warn`.
    pub level: String,
    /// Output shape.
    pub format: LogFormat,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: "info".to_owned(),
            format: LogFormat::Pretty,
        }
    }
}

/// `[metrics]` section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricsConfig {
    /// Exposes `/metrics` when true; returns 404 when false.
    pub enabled: bool,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// The effective server configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// `host:port` to listen on.
    pub listen: String,
    /// Directory holding `tokenizer.json` and `chat_template.jinja`.
    pub model_path: PathBuf,
    /// The model id clients ask for and `/v1/models` reports.
    pub model_id: String,
    /// Admission limit; excess requests get 429.
    pub max_concurrent_requests: u32,
    /// Context window the model is served with.
    pub max_context: u32,
    /// Bytes of VRAM the KV cache and recurrent state may use.
    pub kv_bytes: u64,
    /// Which engine to drive.
    pub engine: EngineKind,
    /// Log level and format.
    pub log: LogConfig,
    /// Metric exposure.
    pub metrics: MetricsConfig,
    /// Wall-clock budget of one request.
    pub request_timeout_ms: u64,
    /// How long a shutdown waits for in-flight requests before cancelling them.
    pub drain_timeout_ms: u64,
}

/// The model directory the engine is built for, unless the config says otherwise.
///
/// The weights never live in the repository, so the default follows the user's
/// home directory; every deployment is expected to set `model_path` explicitly.
fn default_model_path() -> PathBuf {
    match std::env::var("HOME") {
        Ok(home) => PathBuf::from(home).join("models/qwen38-flash-next-nvfp4"),
        Err(_) => PathBuf::from("/models/qwen38-flash-next-nvfp4"),
    }
}

/// The `INFERNO_*` environment layer: `INFERNO_A__B` maps onto the `a.b` key.
///
/// `ignore` runs after `split`, so it drops exactly the bare `log` name and
/// every nested `log__*` key still reaches the `[log]` section.
fn env_layer() -> Env {
    Env::prefixed(ENV_PREFIX)
        .split("__")
        .ignore(&[LOG_FILTER_VARIABLE])
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:8080".to_owned(),
            model_path: default_model_path(),
            model_id: "qwen3.8-flash-next-nvfp4".to_owned(),
            max_concurrent_requests: 8,
            max_context: 32768,
            kv_bytes: 4 * 1024 * 1024 * 1024,
            engine: EngineKind::Zig,
            log: LogConfig::default(),
            metrics: MetricsConfig::default(),
            request_timeout_ms: 120_000,
            drain_timeout_ms: 10_000,
        }
    }
}

/// Command-line overrides, all optional so the layers below win when absent.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Overrides {
    /// `--listen`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub listen: Option<String>,
    /// `--model-path`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_path: Option<PathBuf>,
    /// `--model-id`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// `--max-concurrent-requests`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_concurrent_requests: Option<u32>,
    /// `--max-context`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_context: Option<u32>,
    /// `--kv-bytes`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_bytes: Option<u64>,
    /// `--engine`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine: Option<EngineKind>,
    /// `--log-level`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_level: Option<String>,
    /// `--log-format`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_format: Option<LogFormat>,
    /// `--request-timeout-ms`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_timeout_ms: Option<u64>,
    /// `--drain-timeout-ms`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drain_timeout_ms: Option<u64>,
    /// `--no-metrics`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<MetricsConfig>,
}

impl Overrides {
    /// Folds the CLI into the `Config` shape figment merges.
    fn into_layer(self) -> OverridesLayer {
        OverridesLayer {
            listen: self.listen,
            model_path: self.model_path,
            model_id: self.model_id,
            max_concurrent_requests: self.max_concurrent_requests,
            max_context: self.max_context,
            kv_bytes: self.kv_bytes,
            engine: self.engine,
            log: if self.log_level.is_some() || self.log_format.is_some() {
                Some(PartialLogConfig {
                    level: self.log_level,
                    format: self.log_format,
                })
            } else {
                None
            },
            metrics: self.metrics,
            request_timeout_ms: self.request_timeout_ms,
            drain_timeout_ms: self.drain_timeout_ms,
        }
    }
}

/// The CLI layer in the shape of [`Config`], with nested sections as partials.
#[derive(Debug, Clone, Serialize)]
struct OverridesLayer {
    #[serde(skip_serializing_if = "Option::is_none")]
    listen: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_concurrent_requests: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_context: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    engine: Option<EngineKind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    log: Option<PartialLogConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metrics: Option<MetricsConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_timeout_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    drain_timeout_ms: Option<u64>,
}

/// `[log]` with optional members, so a CLI override keeps the file's other half.
#[derive(Debug, Clone, Serialize)]
struct PartialLogConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    level: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<LogFormat>,
}

/// A configuration that cannot be served.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// A layer could not be read or does not fit `Config` (unknown key, wrong type).
    #[error("configuration is invalid: {0}")]
    Layered(#[from] Box<figment::Error>),
    /// A value is well typed but not usable.
    #[error("configuration is invalid: {0}")]
    Invalid(String),
}

impl Config {
    /// Builds the effective configuration: defaults, then the TOML file, then the
    /// environment, then the command line.
    pub fn load(
        path: Option<&std::path::Path>,
        overrides: Overrides,
    ) -> Result<(Self, ConfigSources), ConfigError> {
        let mut figment = Figment::from(Serialized::defaults(Config::default()));
        let mut sources = ConfigSources::default();
        if let Some(path) = path {
            figment = figment.merge(Toml::file(path));
            sources.file = Some(path.display().to_string());
        }
        figment = figment.merge(env_layer());
        if std::env::vars().any(|(name, _)| name.starts_with(ENV_PREFIX)) {
            sources.environment = true;
        }
        let overrides = overrides.into_layer();
        sources.command_line = overrides.has_values();
        figment = figment.merge(Serialized::defaults(overrides));

        let config: Self = figment
            .extract()
            .map_err(|error| ConfigError::Layered(Box::new(error)))?;
        config.validate()?;
        Ok((config, sources))
    }

    /// Rejects values that are well typed but unusable.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.socket_address()
            .map_err(|error| ConfigError::Invalid(format!("listen: {error}")))?;
        if self.max_concurrent_requests < 1 {
            return Err(ConfigError::Invalid(
                "max_concurrent_requests must be at least 1".to_owned(),
            ));
        }
        if self.max_context < 2 {
            return Err(ConfigError::Invalid(
                "max_context must leave room for a prompt and a token".to_owned(),
            ));
        }
        if self.kv_bytes == 0 {
            return Err(ConfigError::Invalid(
                "kv_bytes must be at least 1".to_owned(),
            ));
        }
        if self.request_timeout_ms < 1 {
            return Err(ConfigError::Invalid(
                "request_timeout_ms must be at least 1".to_owned(),
            ));
        }
        if self.model_id.trim().is_empty() {
            return Err(ConfigError::Invalid(
                "model_id must not be empty".to_owned(),
            ));
        }
        tracing_subscriber::EnvFilter::try_new(&self.log.level)
            .map_err(|error| ConfigError::Invalid(format!("log.level: {error}")))?;
        Ok(())
    }

    /// The parsed `listen` address.
    pub fn socket_address(&self) -> Result<SocketAddr, std::net::AddrParseError> {
        self.listen.parse()
    }

    /// The effective configuration with secrets redacted, for the startup log.
    ///
    /// No field is a secret today; the walk is here so that adding one (an API
    /// token, a hub credential) cannot leak it into a log line by accident.
    pub fn redacted(&self) -> serde_json::Value {
        let mut value = serde_json::to_value(self).expect("Config serialises");
        redact(&mut value);
        value
    }
}

/// Masks every value whose key looks like a credential.
fn redact(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(object) => {
            for (key, item) in object.iter_mut() {
                let lowered = key.to_ascii_lowercase();
                if lowered.contains("token")
                    || lowered.contains("secret")
                    || lowered.contains("password")
                    || lowered.contains("credential")
                {
                    *item = serde_json::Value::String("<redacted>".to_owned());
                } else {
                    redact(item);
                }
            }
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(redact),
        _ => {}
    }
}

/// Where the effective configuration came from.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConfigSources {
    /// The TOML file, when one was given.
    pub file: Option<String>,
    /// True when at least one `INFERNO_*` variable is set.
    pub environment: bool,
    /// True when the command line contributed a value.
    pub command_line: bool,
}

impl OverridesLayer {
    /// True when any override is present.
    fn has_values(&self) -> bool {
        self.listen.is_some()
            || self.model_path.is_some()
            || self.model_id.is_some()
            || self.max_concurrent_requests.is_some()
            || self.max_context.is_some()
            || self.kv_bytes.is_some()
            || self.engine.is_some()
            || self.log.is_some()
            || self.metrics.is_some()
            || self.request_timeout_ms.is_some()
            || self.drain_timeout_ms.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_temp(name: &str, contents: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!("inferno-config-{}", std::process::id()));
        std::fs::create_dir_all(&directory).expect("temp dir");
        let path = directory.join(name);
        std::fs::write(&path, contents).expect("temp file");
        path
    }

    #[test]
    fn defaults_are_valid() {
        Config::default()
            .validate()
            .expect("the defaults are servable");
    }

    #[test]
    fn a_file_value_wins_over_the_default() {
        let path = write_temp("listen.toml", "listen = \"0.0.0.0:9001\"\n");
        let (config, sources) = Config::load(Some(&path), Overrides::default()).expect("loads");
        assert_eq!(config.listen, "0.0.0.0:9001");
        assert!(sources.file.is_some());
    }

    #[test]
    fn a_command_line_value_wins_over_the_file() {
        let path = write_temp("both.toml", "listen = \"0.0.0.0:9001\"\n");
        let overrides = Overrides {
            listen: Some("127.0.0.1:9100".to_owned()),
            ..Overrides::default()
        };
        let (config, sources) = Config::load(Some(&path), overrides).expect("loads");
        assert_eq!(config.listen, "127.0.0.1:9100");
        assert!(sources.command_line);
    }

    #[test]
    fn a_command_line_log_level_keeps_the_file_format() {
        let path = write_temp("log.toml", "[log]\nlevel = \"warn\"\nformat = \"json\"\n");
        let overrides = Overrides {
            log_level: Some("debug".to_owned()),
            ..Overrides::default()
        };
        let (config, _) = Config::load(Some(&path), overrides).expect("loads");
        assert_eq!(config.log.level, "debug");
        assert_eq!(config.log.format, LogFormat::Json);
    }

    #[test]
    fn an_unknown_key_is_refused() {
        let path = write_temp("typo.toml", "lstien = \"0.0.0.0:9001\"\n");
        let error = Config::load(Some(&path), Overrides::default())
            .expect_err("a misspelled key must refuse to start");
        let message = error.to_string();
        assert!(message.contains("lstien"), "unexpected error: {message}");
    }

    #[test]
    fn an_unusable_value_is_refused() {
        let path = write_temp("zero.toml", "max_concurrent_requests = 0\n");
        let error = Config::load(Some(&path), Overrides::default())
            .expect_err("zero concurrency is not servable");
        assert!(error.to_string().contains("max_concurrent_requests"));
    }

    #[test]
    fn secrets_do_not_survive_redaction() {
        let mut value = serde_json::json!({
            "model_id": "qwen",
            "api_token": "s3cret",
            "nested": {"hub_secret": "s3cret"},
        });
        redact(&mut value);
        assert_eq!(value["model_id"], "qwen");
        assert_eq!(value["api_token"], "<redacted>");
        assert_eq!(value["nested"]["hub_secret"], "<redacted>");
    }
}

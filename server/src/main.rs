//! `inferno-server`: command line and startup.
//!
//! `--probe` and `--version` are the smoke commands the build gate runs; with no
//! action flag the binary serves the OpenAI-compatible API.

use std::process::ExitCode;

use clap::Parser;
use inferno_server::config::{Config, LogFormat, Overrides};
use inferno_server::server::{AppState, Server};
use inferno_server::telemetry;

/// Conventional exit code for a bad command line.
const EXIT_USAGE: u8 = 2;

/// Conventional exit code for a failed start.
const EXIT_FAILURE: u8 = 1;

/// The command line.
#[derive(Debug, Parser)]
#[command(
    name = "inferno-server",
    about = "OpenAI-compatible control plane for the inferno inference engine"
)]
struct Cli {
    /// TOML configuration file.
    #[arg(long, value_name = "FILE")]
    config: Option<std::path::PathBuf>,
    /// host:port to listen on.
    #[arg(long, value_name = "ADDR")]
    listen: Option<String>,
    /// Directory holding tokenizer.json and chat_template.jinja.
    #[arg(long, value_name = "DIR")]
    model_path: Option<std::path::PathBuf>,
    /// Model id reported to clients.
    #[arg(long, value_name = "ID")]
    model_id: Option<String>,
    /// Requests served concurrently; excess requests get 429.
    #[arg(long, value_name = "N")]
    max_concurrent_requests: Option<u32>,
    /// Context window in tokens.
    #[arg(long, value_name = "N")]
    max_context: Option<u32>,
    /// VRAM budget for KV cache and recurrent state.
    #[arg(long, value_name = "BYTES")]
    kv_bytes: Option<u64>,
    /// Engine to drive: zig or mock.
    #[arg(long, value_enum)]
    engine: Option<inferno_server::config::EngineKind>,
    /// Log filter directive, e.g. info or inferno_server=debug.
    #[arg(long = "log-level", value_name = "FILTER")]
    log_level: Option<String>,
    /// Log shape: pretty or json.
    #[arg(long = "log-format", value_enum)]
    log_format: Option<LogFormat>,
    /// Per-request wall-clock budget.
    #[arg(long, value_name = "MS")]
    request_timeout_ms: Option<u64>,
    /// How long a shutdown waits for in-flight requests.
    #[arg(long, value_name = "MS")]
    drain_timeout_ms: Option<u64>,
    /// Do not expose /metrics.
    #[arg(long)]
    no_metrics: bool,
    /// Print the effective configuration and exit.
    #[arg(long)]
    print_config: bool,
    /// Print the JSON device probe of the Zig core and exit.
    #[arg(long)]
    probe: bool,
    /// Print the version of the Zig core and exit.
    #[arg(long)]
    version: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if cli.probe {
        return probe_devices();
    }
    if cli.version {
        return print_version();
    }
    run(cli)
}

/// Prints the JSON device probe from the core library.
fn probe_devices() -> ExitCode {
    match inferno_server::engine::probe::probe_document() {
        Ok(document) => {
            println!("{document}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("inferno-server: device probe failed: {message}");
            ExitCode::FAILURE
        }
    }
}

/// Prints the version of the linked core.
fn print_version() -> ExitCode {
    println!("{}", inferno_server::engine::probe::version());
    ExitCode::SUCCESS
}

/// Loads the configuration, builds the state and serves.
fn run(cli: Cli) -> ExitCode {
    let overrides = Overrides {
        listen: cli.listen,
        model_path: cli.model_path,
        model_id: cli.model_id,
        max_concurrent_requests: cli.max_concurrent_requests,
        max_context: cli.max_context,
        kv_bytes: cli.kv_bytes,
        engine: cli.engine,
        log_level: cli.log_level,
        log_format: cli.log_format,
        request_timeout_ms: cli.request_timeout_ms,
        drain_timeout_ms: cli.drain_timeout_ms,
        metrics: if cli.no_metrics {
            Some(inferno_server::config::MetricsConfig { enabled: false })
        } else {
            None
        },
    };

    let (config, sources) = match Config::load(cli.config.as_deref(), overrides) {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("inferno-server: {error}");
            return ExitCode::from(EXIT_USAGE);
        }
    };

    if let Err(error) = telemetry::init(&config.log) {
        eprintln!("inferno-server: {error}");
        return ExitCode::from(EXIT_FAILURE);
    }

    tracing::info!(config = %config.redacted(), file = ?sources.file, environment = sources.environment, command_line = sources.command_line, "effective configuration");

    if cli.print_config {
        println!(
            "{}",
            serde_json::to_string_pretty(&config.redacted()).expect("serialises")
        );
        return ExitCode::SUCCESS;
    }

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            tracing::error!(%error, "cannot build the async runtime");
            return ExitCode::from(EXIT_FAILURE);
        }
    };

    runtime.block_on(async move {
        let state = match AppState::build(config) {
            Ok(state) => state,
            Err(error) => {
                tracing::error!(%error, "cannot start: {error}");
                return ExitCode::from(EXIT_FAILURE);
            }
        };
        let server = match Server::bind(state).await {
            Ok(server) => server,
            Err(error) => {
                tracing::error!(%error, "cannot start: {error}");
                return ExitCode::from(EXIT_FAILURE);
            }
        };
        match server.local_addr() {
            Ok(address) => tracing::info!(%address, "listening"),
            Err(error) => tracing::warn!(%error, "cannot report the bound address"),
        }
        match server.run().await {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                tracing::error!(%error, "server failed: {error}");
                ExitCode::from(EXIT_FAILURE)
            }
        }
    })
}

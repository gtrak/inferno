//! The `INFERNO_*` environment layer, driven through the real binary.
//!
//! `INFERNO_LOG` is the log-filter override `telemetry::init` documents (like
//! `RUST_LOG`), not a `[log]` configuration key. It used to be read as one, so
//! the documented escape hatch aborted startup with
//! `invalid type: found string "debug", expected struct LogConfig`.
//!
//! Only a real process start can show this: the variable is read by figment's
//! environment provider, not by an in-process double. The child runs with a
//! cleared environment so the suite's own variables cannot change the answer.

use std::process::Command;

use serde_json::Value;

/// The binary under test, built by cargo for integration tests.
const BINARY: &str = env!("CARGO_BIN_EXE_inferno-server");

/// Runs `--print-config` with exactly `variables` set.
fn print_config(variables: &[(&str, &str)]) -> std::process::Output {
    let mut command = Command::new(BINARY);
    command.env_clear().arg("--print-config");
    for (name, value) in variables {
        command.env(name, value);
    }
    command
        .output()
        .unwrap_or_else(|error| panic!("cannot run {BINARY}: {error}"))
}

/// The configuration document of a successful `--print-config`.
///
/// `--print-config` prints pretty JSON whose opening brace is alone on its line,
/// and every log record is a single line, so the first `{\n` opens the document
/// however much the startup log wrote before it.
fn config_document(output: &std::process::Output) -> Value {
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        output.status.success(),
        "exit {} with stderr: {}stdout: {stdout}",
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stderr),
    );
    let start = stdout
        .find("{\n")
        .unwrap_or_else(|| panic!("--print-config wrote no JSON document: {stdout}"));
    serde_json::from_str(&stdout[start..])
        .unwrap_or_else(|error| panic!("config is not JSON ({error}): {stdout}"))
}

#[test]
fn inferno_log_overrides_the_filter_without_being_a_config_key() {
    let config = config_document(&print_config(&[("INFERNO_LOG", "debug")]));
    // The `[log]` section survived as the struct it is, and keeps the built-in
    // default: `INFERNO_LOG` selects the filter, it never becomes `log.level`.
    assert_eq!(config["log"]["level"], "info");
    assert_eq!(config["log"]["format"], "pretty");
}

#[test]
fn a_nested_log_key_still_comes_from_the_environment() {
    let config = config_document(&print_config(&[
        ("INFERNO_LOG__LEVEL", "debug"),
        ("INFERNO_LOG__FORMAT", "json"),
    ]));
    // Dropping the bare name must not drop the `log` namespace: these two are the
    // documented way to override the section from the environment.
    assert_eq!(config["log"]["level"], "debug");
    assert_eq!(config["log"]["format"], "json");
}

#[test]
fn an_unknown_environment_key_is_still_refused() {
    let output = print_config(&[("INFERNO_LSTIEN", "0.0.0.0:9001")]);
    assert!(!output.status.success(), "a typo must refuse to start");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("lstien"),
        "the refusal must name the key"
    );
}

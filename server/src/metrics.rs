//! The Prometheus metrics the control plane owns.
//!
//! Names follow the issue contract: `inferno_requests_total`,
//! `inferno_tokens_generated_total`, `inferno_ttft_seconds`,
//! `inferno_inter_token_seconds`, `inferno_queue_depth`, `inferno_in_flight`,
//! `inferno_engine_errors_total`. `prometheus-client` adds the `_total` suffix to
//! counters itself, so the registry names are spelled without it.
//!
//! The engine's own `metrics_json` is not registered here: `/metrics` appends the
//! `inferno_gpu_*` keys from that document after the registry text, so the
//! process exposes one consistent snapshot per scrape and the Zig side stays free
//! to name its own counters.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{Histogram, exponential_buckets};
use prometheus_client::registry::Registry;

/// Status label of `inferno_requests_total`.
#[derive(Debug, Clone, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct RequestStatus {
    /// `ok`, `client_gone`, `timeout`, `error` or `rejected`.
    pub status: String,
}

/// Every metric family the server registers.
#[derive(Debug)]
pub struct Metrics {
    /// The registry `/metrics` encodes.
    pub registry: Registry,
    requests: Family<RequestStatus, Counter>,
    tokens_generated: Counter,
    ttft: Histogram,
    inter_token: Histogram,
    queue_depth: Gauge,
    in_flight: Gauge,
    engine_errors: Counter,
}

impl Metrics {
    /// Registers every family.
    pub fn new() -> Self {
        let mut registry = Registry::default();

        let requests: Family<RequestStatus, Counter> = Family::default();
        // Touch every status once, so the family is exported before the first
        // request finishes and `rate()` queries have a series to start from.
        for status in ["ok", "client_gone", "timeout", "error", "rejected"] {
            let _series = requests.get_or_create(&RequestStatus {
                status: status.to_owned(),
            });
        }
        registry.register(
            "inferno_requests",
            "Finished generation requests by status",
            requests.clone(),
        );

        let tokens_generated: Counter = Counter::default();
        registry.register(
            "inferno_tokens_generated",
            "Tokens the engine generated",
            tokens_generated.clone(),
        );

        // 1 ms .. ~4 s, factor 2: TTFT and inter-token latency live there.
        let ttft = Histogram::new(exponential_buckets(0.001, 2.0, 12));
        registry.register(
            "inferno_ttft_seconds",
            "Time from submit to the first token",
            ttft.clone(),
        );

        let inter_token = Histogram::new(exponential_buckets(0.0005, 2.0, 12));
        registry.register(
            "inferno_inter_token_seconds",
            "Time between two generated tokens",
            inter_token.clone(),
        );

        let queue_depth: Gauge = Gauge::default();
        registry.register(
            "inferno_queue_depth",
            "Admitted requests waiting for their first token",
            queue_depth.clone(),
        );

        let in_flight: Gauge = Gauge::default();
        registry.register(
            "inferno_in_flight",
            "Requests currently holding an admission permit",
            in_flight.clone(),
        );

        let engine_errors: Counter = Counter::default();
        registry.register(
            "inferno_engine_errors",
            "Engine failures reported through the token callback or submit",
            engine_errors.clone(),
        );

        Self {
            registry,
            requests,
            tokens_generated,
            ttft,
            inter_token,
            queue_depth,
            in_flight,
            engine_errors,
        }
    }

    /// The gauge `Admission` keeps in step with the permit count.
    pub fn in_flight_gauge(&self) -> Gauge {
        self.in_flight.clone()
    }

    /// Counts one finished request and the tokens it produced.
    ///
    /// The TTFT histogram is written when the first token arrives (see
    /// [`Self::observe_ttft`]), so a stream the client abandons still contributes
    /// its latency to the histogram.
    pub fn finish_request(&self, status: &str, completion_tokens: u64) {
        self.requests
            .get_or_create(&RequestStatus {
                status: status.to_owned(),
            })
            .inc();
        self.tokens_generated.inc_by(completion_tokens);
    }

    /// Counts one rejection (429) or failed request.
    pub fn reject(&self) {
        self.requests
            .get_or_create(&RequestStatus {
                status: "rejected".to_owned(),
            })
            .inc();
    }

    /// Records the TTFT of one request.
    pub fn observe_ttft(&self, seconds: f64) {
        self.ttft.observe(seconds);
    }

    /// Records the gap between two tokens.
    pub fn observe_inter_token(&self, seconds: f64) {
        self.inter_token.observe(seconds);
    }

    /// Counts one engine failure.
    pub fn engine_error(&self) {
        self.engine_errors.inc();
    }

    /// A request moved from "waiting" to "streaming".
    pub fn start_of_stream(&self) {
        self.queue_depth.dec();
    }

    /// A request entered the waiting state (it holds a permit, no token yet).
    pub fn enter_queue(&self) {
        self.queue_depth.inc();
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Parses the engine's `metrics_json` and appends the `inferno_gpu_*` keys as
/// gauges in Prometheus text format.
///
/// Anything that is not a finite number, or is not named `inferno_gpu_*`, is
/// skipped: the engine document is data from another language and the endpoint
/// must stay parseable even while the core is being developed.
pub fn append_engine_metrics(document: &str, output: &mut String) {
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(document) else {
        return;
    };
    let Some(object) = parsed.as_object() else {
        return;
    };
    let mut names: Vec<&String> = object.keys().collect();
    names.sort();
    for name in names {
        if !name.starts_with("inferno_gpu_") {
            continue;
        }
        let Some(value) = object[name].as_f64() else {
            continue;
        };
        output.push_str("# TYPE ");
        output.push_str(name);
        output.push_str(" gauge\n");
        output.push_str(name);
        output.push(' ');
        output.push_str(&format!("{value}\n"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_exposes_the_contracted_names() {
        let metrics = Metrics::new();
        let mut text = String::new();
        prometheus_client::encoding::prometheus_text::encode(&mut text, &metrics.registry)
            .expect("encodes");
        for name in [
            "inferno_requests_total",
            "inferno_tokens_generated_total",
            "inferno_ttft_seconds",
            "inferno_inter_token_seconds",
            "inferno_queue_depth",
            "inferno_in_flight",
            "inferno_engine_errors_total",
        ] {
            assert!(text.contains(name), "{name} missing from:\n{text}");
        }
    }

    #[test]
    fn a_histogram_is_encoded_with_buckets_sum_and_count() {
        let metrics = Metrics::new();
        metrics.observe_ttft(0.02);
        let mut text = String::new();
        prometheus_client::encoding::prometheus_text::encode(&mut text, &metrics.registry)
            .expect("encodes");
        assert!(
            text.contains("# TYPE inferno_ttft_seconds histogram"),
            "{text}"
        );
        assert!(
            text.contains("inferno_ttft_seconds_bucket{le=\"0.032\"} 1"),
            "{text}"
        );
        assert!(text.contains("inferno_ttft_seconds_sum 0.02"), "{text}");
        assert!(text.contains("inferno_ttft_seconds_count 1"), "{text}");
    }

    #[test]
    fn engine_metrics_are_re_exported_as_gauges() {
        let mut text = String::new();
        append_engine_metrics(
            r#"{"inferno_gpu_kv_used_bytes": 1024, "inferno_gpu_ok": true, "other": 3}"#,
            &mut text,
        );
        assert_eq!(
            text,
            "# TYPE inferno_gpu_kv_used_bytes gauge\ninferno_gpu_kv_used_bytes 1024\n"
        );
    }

    #[test]
    fn a_broken_engine_document_is_ignored() {
        let mut text = String::new();
        append_engine_metrics("not json", &mut text);
        assert!(text.is_empty());
    }
}

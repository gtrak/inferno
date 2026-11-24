# `inferno-server` — the Rust control plane

The OpenAI-compatible HTTP surface in front of the Zig inference core. The hot
path (CUDA driver, kernels, scheduling, sampling) is `core/` in Zig; everything
here is control plane: HTTP/SSE, the tokenizer and chat template, admission
control, configuration, logging and Prometheus metrics. Nothing in this crate
touches per-token compute.

```
server/src/
  engine/          the Engine trait, the Zig C ABI (ffi.rs), ZigEngine, MockEngine
  api/             axum handlers: chat, completions, models, health, metrics
  tokenizer.rs     tokenizer.json (HF `tokenizers`)
  template.rs      chat_template.jinja (minijinja) + tool-call parsing
  stop.rs          stop strings, matched on decoded text
  admission.rs     the max_concurrent_requests permits (429 beyond them)
  config.rs        TOML + INFERNO_* env + CLI, deny_unknown_fields
  metrics.rs       the Prometheus registry
  server.rs        AppState, bind, serve, drain, engine shutdown
```

## Quick start (no GPU)

`--engine mock` serves a deterministic engine that emits the tokenization of

> `inferno mock engine: one two three four five six seven eight nine ten.`

so the whole HTTP layer can be exercised on a laptop or in CI.

```bash
cargo run -p inferno-server --release -- --engine mock \
  --model-path ~/models/qwen38-flash-next-nvfp4 \
  --model-id qwen3.8-flash-next-nvfp4 --listen 127.0.0.1:8080
```

The real engine is `--engine zig` (the default). It needs the engine C ABI, which
lands in issue I-15; until then the core library only exports the device probe, so
`--engine zig` refuses to start with a clear message and `--engine mock` is the
supported mode.

### Smoke commands

```bash
inferno-server --probe      # JSON device probe from the Zig core
inferno-server --version    # version of the linked core
inferno-server --print-config   # effective configuration, secrets redacted
```

## curl examples

```bash
# liveness / readiness / model list
curl -s localhost:8080/health
curl -s localhost:8080/readyz
curl -s localhost:8080/v1/models

# buffered chat completion
curl -s localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
        "model": "qwen3.8-flash-next-nvfp4",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 32,
        "temperature": 0.0
      }'

# streaming chat completion (-N keeps curl from buffering the SSE body)
curl -N -s localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
        "model": "qwen3.8-flash-next-nvfp4",
        "messages": [{"role": "user", "content": "Count to five."}],
        "max_tokens": 64,
        "stream": true,
        "stream_options": {"include_usage": true}
      }'

# a stop string and a tool schema
curl -s localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "stop": ["\n\n"],
        "tools": [{
          "type": "function",
          "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]}
          }
        }]
      }'

# legacy completion
curl -s localhost:8080/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model": "qwen3.8-flash-next-nvfp4", "prompt": "Once upon a time", "max_tokens": 16}'

# metrics
curl -s localhost:8080/metrics
```

The streaming body is `text/event-stream`, one `data: {json}` per delta, a final
chunk carrying `finish_reason`, an optional usage chunk, then `data: [DONE]`.

## Configuration

Precedence: defaults → `--config <file.toml>` → `INFERNO_*` environment →
command line. Nested keys use `__` in the environment
(`INFERNO_LOG__LEVEL=debug`). Unknown keys are refused at startup, so a typo can
never silently fall back to a default.

The one `INFERNO_*` name that is not a configuration key is `INFERNO_LOG`: like
`RUST_LOG` it sets the log filter (see `telemetry::init`) and is skipped by the
config layer, so `INFERNO_LOG=debug` turns up the verbosity without touching the
`[log]` section.

```toml
listen = "0.0.0.0:8080"
model_path = "/home/gary/models/qwen38-flash-next-nvfp4"
model_id = "qwen3.8-flash-next-nvfp4"
max_concurrent_requests = 8      # excess requests get 429 immediately
max_context = 32768
kv_bytes = 4294967296
engine = "zig"                   # or "mock"
request_timeout_ms = 120000
drain_timeout_ms = 10000

[log]
level = "info"                   # any tracing filter directive
format = "json"                  # or "pretty"

[metrics]
enabled = true
```

## Behaviour worth knowing

- **Backpressure**: `max_concurrent_requests` permits *are* the queue. A request
  that cannot take one gets `429` at once; nothing queues unboundedly.
- **Stop strings** are matched on the decoded text, never per token, and the
  stop text is never emitted — not even when it spans several tokens.
- **Client disconnect** (the SSE body being dropped) cancels the request in the
  engine and frees the permit; the engine's own cancel counter shows it.
- **Graceful shutdown** on `SIGINT`/`SIGTERM`: stop accepting, wait up to
  `drain_timeout_ms` for in-flight requests, cancel whatever is left (a stream
  that would otherwise hold the process open), then `inferno_core_shutdown`. No
  forced kills.
- **Readiness** (`/readyz`) follows the engine handle, not the tokenizer: the
  process can be alive and serving `/health` while `/readyz` is 503.
- **Tool calls**: the model's `<tool_call><function=…><parameter=…>` output is
  parsed from the buffered answer into OpenAI `tool_calls` (with
  `finish_reason: "tool_calls"`). Streaming carries the raw text as content
  deltas; no tool-call assembly is done mid-stream.

## Metrics

Prometheus text (0.0.4) at `/metrics`, names per the issue contract:

| metric | type | meaning |
|---|---|---|
| `inferno_requests_total{status}` | counter | `ok`, `client_gone`, `timeout`, `error`, `rejected` |
| `inferno_tokens_generated_total` | counter | tokens emitted by the engine |
| `inferno_ttft_seconds` | histogram | submit → first token |
| `inferno_inter_token_seconds` | histogram | token-to-token latency |
| `inferno_queue_depth` | gauge | admitted requests waiting for their first token |
| `inferno_in_flight` | gauge | requests holding an admission permit |
| `inferno_engine_errors_total` | counter | engine failures |

Plus every `inferno_gpu_*` key the engine reports in its own `metrics_json`,
re-exported as a gauge.

## Tests

```bash
cargo test -p inferno-server          # unit + integration (mock engine, port 0)
bash tools/gate.sh                    # the project gate
```

- `tests/integration.rs` (14 tests) starts the server in-process on a random port and drives
  it over a raw socket: buffered and streamed answers must contain exactly the
  mock's scripted text, `max_tokens` must be honoured exactly, a stop string must
  truncate without leaking, the 429 must appear when the permit is taken, a
  dropped stream must cancel in the engine, `/metrics` must carry the contracted
  names and a real histogram, and `/health`/`/readyz` must follow the engine.
- `tests/chat_goldens.rs` pins the chat template against Python `jinja2` (byte
  identical, all cases) and the prompt token ids against the real model
  tokenizer (needs `INFERNO_MODEL_DIR`, default `~/models/qwen38-flash-next-nvfp4`;
  it fails rather than skips when the tokenizer is absent).
- `tests/fixtures/make_fixtures.py` regenerates both fixture sets and records the
  provenance (renderer, tokenizer path, SHA-256 of tokenizer and template).

## Notes on the engine boundary

`server/src/engine/ffi.rs` declares the C ABI of issue I-15 verbatim, including
the `flags` bits (`1` final, `2` error, `4` first token). `server/build.rs` looks
for those symbols in `libinferno_core.a`; while they are missing it compiles the
`not implemented` fallbacks in `server/src/engine/ffi_absent.rs`, so the crate
links and `--engine zig` fails with a message instead of a link error. The day
the core exports the ABI the fallbacks disappear without an edit.

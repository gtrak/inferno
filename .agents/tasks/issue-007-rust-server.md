# I-07 — Rust control plane: OpenAI-compatible server, tokenizer, metrics, config, logging

## Working agreement

- Sole writer in your worktree. You own **`server/` only** (plus `Cargo.toml` at
  the root if it needs a workspace-level change). Never touch `core/`.
- Read `plan.md` §4/§5 first. The Zig core is the hot path; Rust is the control
  plane. Nothing in Rust may run per-token work that Zig could do.
- Builds on I-01: `Cargo.toml` workspace with member `server`, `server/build.rs`
  that links the Zig static lib.
- Crates allowed (keep the dependency set small and mainstream):
  `axum 0.8`, `tokio`, `serde`/`serde_json`, `tokenizers`, `tracing`,
  `tracing-subscriber`, `prometheus-client` (or `metrics` + `metrics-exporter-prometheus`),
  `thiserror`, `clap`, `figment` (or `toml` + `serde`), `futures`, `async-stream`.
  Pin exact versions in `Cargo.toml`. `cargo clippy -D warnings` must pass.

## The engine boundary (fixed contract — implement both sides of the Rust half)

Zig exports a C ABI (implemented later in I-15). Declare it in
`server/src/engine/ffi.rs` exactly as follows and keep it stable:

```c
typedef struct inferno_engine inferno_engine;
typedef void (*inferno_token_cb)(void* ctx, uint64_t req_id, uint32_t token_id, uint32_t flags);
extern inferno_engine* inferno_core_init(const char* config_json, uint32_t len);
extern void            inferno_core_shutdown(inferno_engine*);
extern const char*     inferno_core_version(void);
extern int32_t         inferno_core_submit(inferno_engine*, uint64_t req_id,
                                           const char* req_json, uint32_t len);
extern int32_t         inferno_core_cancel(inferno_engine*, uint64_t req_id);
extern void            inferno_core_set_token_cb(inferno_engine*, inferno_token_cb, void* ctx);
extern int32_t         inferno_core_metrics(inferno_engine*, char* buf, uint32_t cap);
```

`flags` bits: `1 = final token`, `2 = error`, `4 = first token (TTFT marks)`.
The callback is invoked from a Zig worker thread; the Rust side must only push
into a channel (no locks held across FFI, no panics across FFI).

`req_json` (Rust → Zig):
```json
{"prompt_tokens":[1,2,3],"max_tokens":128,"temperature":0.7,"top_p":0.95,
 "top_k":20,"min_p":0.0,"seed":42,"stop_token_ids":[248046,248044],
 "stream":true,"priority":0}
```
`inferno_core_submit` returns `0` on accept, negative error codes; a rejected
request must be visible to the client as HTTP 503/500 with a message.

### Rust traits (so everything is testable without the GPU)

```rust
pub trait Engine: Send + Sync {
    fn version(&self) -> String;
    fn submit(&self, req_id: u64, req: GenerateRequest) -> Result<(), EngineError>;
    fn cancel(&self, req_id: u64) -> Result<(), EngineError>;
    fn metrics_json(&self) -> String;
}
pub struct MockEngine { /* deterministic sequence, configurable delay */ }
pub struct ZigEngine  { /* wraps the C ABI, owns the callback channel */ }
```

Select with `--engine mock|zig` (default `zig`; `mock` when the core is stubbed).
`MockEngine` must be able to emit N tokens with an artificial per-token delay and
a TTFT delay so the whole HTTP layer is testable and benchmarkable without a GPU.

## What to build

### 1. HTTP surface (`axum`)

- `POST /v1/chat/completions` — OpenAI-compatible: `messages` (system/user/
  assistant/tool roles, content as string or content-part array), `stream`
  (`text/event-stream`, `data: {...}\n\n`, terminated by `data: [DONE]`),
  `max_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stop`, `n=1`,
  `stream_options.include_usage`. Response fields: `id`, `object`, `created`,
  `model`, `choices[].{index,message|delta,finish_reason}`, `usage`
  (`prompt_tokens`, `completion_tokens`, `total_tokens`).
- `POST /v1/completions` (legacy text completion) — thin wrapper.
- `GET /v1/models` — the loaded model id + context length.
- `GET /health` (liveness) and `GET /readyz` (engine initialized, VRAM ok).
- `GET /metrics` — Prometheus text format.
- Errors: OpenAI error envelope `{"error":{"message","type","code"}}`, correct
  status codes (400 for bad params, 429 for admission rejection, 503 for
  not-ready, 500 internal).
- **Backpressure**: a bounded in-flight counter (config `max_concurrent_requests`);
  excess requests get 429 immediately, never queue unboundedly.
- Client disconnect (stream dropped) must call `cancel`.

### 2. Tokenizer + chat template (`server/src/tokio… no: server/src/tokenizer.rs`, `template.rs`)

- Load `tokenizer.json` with the `tokenizers` crate (BPE, no special-token
  processing surprises); expose `encode(text) -> Vec<u32>`, `decode(&[u32]) -> String`
  (with `skip_special_tokens`).
- Render the chat template from `chat_template.jinja` (`minijinja` is allowed as
  an extra dependency) with `messages`, `tools` (JSON passthrough), `add_generation_prompt`,
  and store the rendered string + the tool-call parsing of the output
  (`<tool_call>…</tool_call>`-style — read the template and implement what it
  actually emits; if it uses a different delimiter, follow the template).
- Stop-string handling: encode each stop string, and truncate the stream at the
  first character boundary that completes a stop string (must not leak the stop
  text to the client). Do this on the *decoded text*, not per token.
- Golden test: a set of `(messages, rendered_prefix, expected_ids)` fixtures in
  `server/tests/fixtures/` produced by the real tokenizer; assert exact ids.

### 3. Config (`server/src/config.rs`)

TOML file + env overrides + CLI (`clap`): `listen`, `model_path`, `model_id`,
`max_concurrent_requests`, `max_context`, `kv_bytes`, `engine` choice, `log`
level/format, `metrics` enabled, `request_timeout_ms`. Validate with assertions;
refuse to start on an unknown key (typo protection) and print an effective config
at startup with secrets redacted.

### 4. Observability

- `tracing` with a JSON layer and a human layer selected by env; one span per
  request with `req_id`, `prompt_tokens`, `completion_tokens`, `ttft_ms`,
  `decode_ms`, `queue_ms`, `finish_reason`.
- Prometheus metrics: `inferno_requests_total{status}`,
  `inferno_tokens_generated_total`, `inferno_ttft_seconds` (histogram),
  `inferno_inter_token_seconds` (histogram), `inferno_queue_depth`,
  `inferno_in_flight`, `inferno_engine_errors_total`, plus a passthrough of the
  engine's `metrics_json` (parse and re-export `inferno_gpu_*` keys it provides).
- Graceful shutdown on SIGINT/SIGTERM: stop accepting, cancel/await in-flight
  (bounded), then `inferno_core_shutdown`. No forced kills.

## Gates

```
bash tools/gate.sh ; echo EXIT=$?
cargo test -p inferno-server   # unit + integration
```

Integration test (must be part of `cargo test`, using `--engine mock` and a
random port): start the server in-process, then
1. non-stream chat completion returns 200 with a non-empty `choices[0].message.content`
   and a `usage` block that adds up;
2. stream chat completion yields ≥3 SSE chunks then `data: [DONE]`, and the
   concatenated deltas equal the non-stream content;
3. `max_tokens` is honoured exactly;
4. a stop string is honoured and never appears in the output;
5. 429 is returned when `max_concurrent_requests` is exceeded (simulate by
   setting it to 1 and firing 2 slow mock requests);
6. client disconnect cancels (assert the engine's cancel counter metric moved);
7. `/metrics` contains `inferno_requests_total` and a valid histogram;
8. `/health` 200 before readiness, `/readyz` reflects engine state.
Also `curl` examples in `server/README.md` that a human can paste.

Non-vacuity: the test must assert on *content* (e.g. the mock emits a known
sequence and the test asserts the exact concatenation), not just status codes.

## Report format

Implemented / Changed files / Validation (paste gate + test output) / Open risks
(the chat-template behaviour you implemented, and anything the Zig core will have
to provide that is not in the ABI above) / Recommended next step.

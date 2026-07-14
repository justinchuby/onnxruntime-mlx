# TODO / Follow-ups

## Tracer dependency (`onnx-runtime-tracer`) after crates.io publish

**Context.** The MLX EP compiles the pure-Rust tracer in unconditionally
(`rust/Cargo.toml` — `onnx-runtime-tracer` is a *non-optional* dependency), and
whether any trace is *recorded* is decided at **runtime** by env vars
(`ONNX_GENAI_MLX_TRACE`, `ONNX_GENAI_MLX_SIGNPOST`, `ONNX_GENAI_MLX_GPU_CAPTURE`,
…). With no env var set, `MlxTracer::is_enabled()` is `false` and tracing is a
near-zero-cost no-op. So "compiled in by default + runtime-controlled recording"
is already the behaviour — the PyPI wheel ships with the tracer built in.

To make the EP publishable to crates.io, the dependency was changed from a
path-only dep to `{ path = ..., version = "0.1.0-dev.0" }`, so `cargo publish`
resolves it from the registry (the path is stripped at publish time).

**Follow-ups once `onnx-runtime-tracer` is on crates.io:**

- [ ] Publish `onnx-runtime-tracer` to crates.io (owner: onnx-genai team), then
      confirm the version requirement in `rust/Cargo.toml` matches the published
      version (currently pinned to `0.1.0-dev.0`). Re-run the `publish-crate`
      job in `publish.yml`.
- [ ] Keep the tracer **compiled in by default**. Consider putting it behind a
      *default-on* `tracing` cargo feature so downstream consumers can opt out
      (`--no-default-features`) for a leaner build, while the default build (and
      the PyPI wheel) keeps full observability. Recording stays runtime-gated by
      the `ONNX_GENAI_MLX_TRACE*` env vars regardless.
- [ ] Once the tracer is a real registry dep, drop the `git clone ../onnx-genai`
      step from the crates.io `publish-crate` job (it is only needed while the
      dep is resolved via the local path).

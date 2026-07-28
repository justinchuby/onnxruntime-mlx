//! Minimal in-crate logger for the plugin dylib.
//!
//! A cdylib has its own statically-linked copy of the `log` crate, so
//! `log::set_logger` will always succeed (no conflict with the host).
//! The logger writes to stderr with the `[rust-mlx-ep]` prefix, matching
//! the format the crate historically used, so existing monitoring/grep
//! patterns still work.
//!
//! # Default behaviour (silent)
//!
//! With no env vars set the max level is **Warn** — only `error!` and
//! `warn!` messages (caught panics, user-visible failures) are emitted.
//! Everything else is silent.
//!
//! # Opt-in levels
//!
//! | Env var                          | Max level |
//! |----------------------------------|-----------|
//! | (none)                           | Warn      |
//! | `ONNXRUNTIME_EP_MLX_VERBOSE=1`   | Info      |
//! | `ONNXRUNTIME_EP_MLX_TRACE=<path>`| Debug     |
//! | `RUST_LOG=onnxruntime_ep_mlx=<l>`| explicit  |
//!
//! `RUST_LOG` overrides the other env vars when set, accepting standard
//! level names (`error`, `warn`, `info`, `debug`, `trace`).

use log::{Level, LevelFilter, Log, Metadata, Record};
use std::sync::Once;

static INIT: Once = Once::new();

struct EpLogger {
    level: LevelFilter,
}

impl Log for EpLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.level
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            let level_tag = match record.level() {
                Level::Error => "ERROR",
                Level::Warn => "WARN",
                Level::Info => "INFO",
                Level::Debug => "DEBUG",
                Level::Trace => "TRACE",
            };
            eprintln!("[rust-mlx-ep] {level_tag}: {}", record.args());
        }
    }

    fn flush(&self) {}
}

/// Resolve the desired log level from environment variables.
fn resolve_level() -> LevelFilter {
    // Highest-priority: explicit RUST_LOG for our crate.
    if let Ok(val) = std::env::var("RUST_LOG") {
        // Accept either bare level ("debug") or crate-qualified ("onnxruntime_ep_mlx=debug").
        let level_str = val
            .split(',')
            .find(|s| s.contains("onnxruntime_ep_mlx"))
            .and_then(|s| s.split('=').nth(1))
            .unwrap_or(&val);
        if let Ok(f) = level_str.parse::<LevelFilter>() {
            return f;
        }
    }
    // Next: tracing enabled → debug (internal diagnostics visible).
    if std::env::var("ONNXRUNTIME_EP_MLX_TRACE")
        .ok()
        .filter(|s| !s.is_empty())
        .is_some()
    {
        return LevelFilter::Debug;
    }
    // Next: verbose summary → info.
    if std::env::var("ONNXRUNTIME_EP_MLX_VERBOSE")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        return LevelFilter::Info;
    }
    // Default: only errors and warnings (caught panics, capture failures).
    LevelFilter::Warn
}

/// Install the logger. Idempotent — safe to call from any entry point.
pub fn init() {
    INIT.call_once(|| {
        let level = resolve_level();
        let logger = Box::new(EpLogger { level });
        // `set_boxed_logger` can only succeed once per process (per the dylib's
        // copy of `log`). In the vanishingly unlikely event another copy of `log`
        // inside this dylib already set one, silently ignore.
        let _ = log::set_boxed_logger(logger);
        log::set_max_level(level);
    });
}

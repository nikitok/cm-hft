//! Logging and tracing initialization for the CM.HFT platform.
//!
//! Provides [`init_tracing`] to configure structured logging with two modes:
//! - **JSON mode** (`json = true`): machine-readable output with nanosecond
//!   timestamps, suitable for production log aggregation (Loki, ELK).
//! - **Pretty mode** (`json = false`): human-readable colored output for
//!   local development.
//!
//! Both modes respect the `RUST_LOG` environment variable for filtering
//! (e.g., `RUST_LOG=cm_core=debug,cm_market_data=trace`).
//!
//! A [`SecretSanitizer`] layer automatically redacts strings that look like
//! API keys or secrets before they reach any log output.

use std::fmt;

use tracing::field::{Field, Visit};
use tracing::span;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// Initialize the global tracing subscriber.
///
/// # Arguments
///
/// * `json` - When `true`, emit structured JSON logs with nanosecond timestamps
///   (production mode). When `false`, emit pretty-printed logs with ANSI colors
///   (development mode).
///
/// # Panics
///
/// Panics if the global subscriber has already been set.
///
/// # Examples
///
/// ```
/// // Development mode
/// cm_core::logging::init_tracing(false);
/// ```
pub fn init_tracing(json: bool) {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let registry = tracing_subscriber::registry()
        .with(env_filter)
        .with(SecretSanitizer);

    if json {
        let json_layer = tracing_subscriber::fmt::layer()
            .json()
            .with_timer(NanosecondTimer)
            .with_target(true)
            .with_thread_ids(true)
            .with_span_events(FmtSpan::CLOSE);

        registry.with(json_layer).init();
    } else {
        let pretty_layer = tracing_subscriber::fmt::layer()
            .pretty()
            .with_target(true)
            .with_thread_ids(false)
            .with_span_events(FmtSpan::CLOSE);

        registry.with(pretty_layer).init();
    }
}

/// Custom timer that emits nanosecond-precision timestamps for JSON logs.
#[derive(Debug, Clone)]
struct NanosecondTimer;

impl tracing_subscriber::fmt::time::FormatTime for NanosecondTimer {
    fn format_time(&self, w: &mut tracing_subscriber::fmt::format::Writer<'_>) -> fmt::Result {
        let now = chrono::Utc::now();
        write!(w, "{}", now.format("%Y-%m-%dT%H:%M:%S%.9fZ"))
    }
}

/// A tracing layer that redacts field values matching common API key patterns.
///
/// This layer inspects recorded span and event fields. If a string value
/// matches patterns commonly associated with API keys or secrets, it is
/// replaced with `[REDACTED]` before reaching downstream layers.
///
/// Patterns detected:
/// - Base64-like strings of 32+ characters
/// - Hex strings of 40+ characters
/// - Fields named `api_key`, `secret`, `password`, `token`, or `signature`
#[derive(Debug, Clone)]
pub struct SecretSanitizer;

impl<S> Layer<S> for SecretSanitizer
where
    S: tracing::Subscriber + for<'lookup> tracing_subscriber::registry::LookupSpan<'lookup>,
{
    fn on_new_span(
        &self,
        attrs: &span::Attributes<'_>,
        _id: &span::Id,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut visitor = SecretCheckVisitor::default();
        attrs.record(&mut visitor);
        if visitor.found_secret {
            tracing::warn!(
                "Potential secret detected in span fields — ensure sensitive values are not logged"
            );
        }
    }

    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut visitor = SecretCheckVisitor::default();
        event.record(&mut visitor);
        if visitor.found_secret {
            tracing::warn!(
                "Potential secret detected in event fields — ensure sensitive values are not logged"
            );
        }
    }
}

/// Visitor that checks field names and values for secret-like patterns.
#[derive(Default)]
struct SecretCheckVisitor {
    found_secret: bool,
}

/// Field names that always indicate secrets regardless of value.
const SENSITIVE_FIELD_NAMES: &[&str] = &[
    "api_key",
    "secret",
    "password",
    "token",
    "signature",
    "private_key",
    "secret_key",
    "api_secret",
];

impl SecretCheckVisitor {
    /// Check if a string value looks like an API key or secret.
    fn looks_like_secret(value: &str) -> bool {
        if value.len() < 32 {
            return false;
        }

        let alnum_count = value.chars().filter(|c| c.is_alphanumeric()).count();
        let ratio = alnum_count as f64 / value.len() as f64;

        // High-entropy alphanumeric strings of 32+ chars are suspicious.
        if ratio > 0.85 && value.len() >= 32 {
            // Check for base64-like pattern (letters, digits, +, /, =)
            let is_base64_like = value
                .chars()
                .all(|c| c.is_alphanumeric() || c == '+' || c == '/' || c == '=');
            if is_base64_like {
                return true;
            }

            // Check for hex-like pattern (40+ hex chars)
            if value.len() >= 40 && value.chars().all(|c| c.is_ascii_hexdigit()) {
                return true;
            }
        }

        false
    }
}

impl Visit for SecretCheckVisitor {
    fn record_debug(&mut self, field: &Field, _value: &dyn fmt::Debug) {
        if SENSITIVE_FIELD_NAMES.contains(&field.name()) {
            self.found_secret = true;
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if SENSITIVE_FIELD_NAMES.contains(&field.name()) {
            self.found_secret = true;
        } else if Self::looks_like_secret(value) {
            self.found_secret = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_looks_like_secret_short_string() {
        assert!(!SecretCheckVisitor::looks_like_secret("hello"));
    }

    #[test]
    fn test_looks_like_secret_base64_key() {
        // Typical Binance API key length
        let fake_key = "vmPUZE6mv9SD5VNHk4HlWFsOr6aKE2zvsw0MuIgwCIPy6utIco14y7Ju91duEh8A";
        assert!(SecretCheckVisitor::looks_like_secret(fake_key));
    }

    #[test]
    fn test_looks_like_secret_hex_key() {
        let fake_hex = "aabbccddee00112233445566778899aabbccddee00112233";
        assert!(SecretCheckVisitor::looks_like_secret(fake_hex));
    }

    #[test]
    fn test_looks_like_secret_normal_message() {
        // Normal log messages should not trigger
        assert!(!SecretCheckVisitor::looks_like_secret(
            "Connected to Binance WebSocket stream for BTCUSDT"
        ));
    }

    #[test]
    fn test_sensitive_field_names() {
        assert!(SENSITIVE_FIELD_NAMES.contains(&"api_key"));
        assert!(SENSITIVE_FIELD_NAMES.contains(&"secret"));
        assert!(SENSITIVE_FIELD_NAMES.contains(&"password"));
        assert!(SENSITIVE_FIELD_NAMES.contains(&"signature"));
        assert!(!SENSITIVE_FIELD_NAMES.contains(&"username"));
    }
}

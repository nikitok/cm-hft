//! Circuit breaker for emergency trading halt.
//!
//! The [`CircuitBreaker`] uses an [`AtomicBool`] for the trading-enabled flag
//! so that the hot-path check (`is_trading_enabled`) is lock-free. The trigger
//! reason and timestamp are behind a `parking_lot::Mutex` since they are only
//! written during exceptional events.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::pipeline::RiskReject;

/// Current state of the circuit breaker, returned by [`CircuitBreaker::status`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct CircuitBreakerStatus {
    /// Whether trading is currently enabled.
    pub trading_enabled: bool,
    /// Whether the circuit breaker has been triggered.
    pub triggered: bool,
    /// The reason the breaker was triggered, if any.
    pub trigger_reason: Option<String>,
}

/// Lock-free circuit breaker for emergency trading halts.
///
/// Checked on every order submission. The `is_trading_enabled` method is a
/// single atomic load with `Relaxed` ordering — suitable for the hot path.
pub struct CircuitBreaker {
    trading_enabled: AtomicBool,
    trigger_reason: parking_lot::Mutex<Option<String>>,
    triggered_at: parking_lot::Mutex<Option<std::time::Instant>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with trading enabled.
    pub fn new() -> Self {
        Self {
            trading_enabled: AtomicBool::new(true),
            trigger_reason: parking_lot::Mutex::new(None),
            triggered_at: parking_lot::Mutex::new(None),
        }
    }

    /// Returns `true` if trading is currently enabled.
    ///
    /// This is a single atomic load and is safe to call on the hot path.
    #[inline]
    pub fn is_trading_enabled(&self) -> bool {
        self.trading_enabled.load(Ordering::Relaxed)
    }

    /// Trigger the circuit breaker, halting all trading.
    ///
    /// Stores the reason and timestamp for diagnostics and logs an error.
    pub fn trigger(&self, reason: String) {
        self.trading_enabled.store(false, Ordering::SeqCst);
        tracing::error!(reason = %reason, "CIRCUIT BREAKER TRIGGERED — trading halted");
        *self.trigger_reason.lock() = Some(reason);
        *self.triggered_at.lock() = Some(std::time::Instant::now());
    }

    /// Reset the circuit breaker, re-enabling trading.
    pub fn reset(&self) {
        self.trading_enabled.store(true, Ordering::SeqCst);
        tracing::warn!("circuit breaker reset — trading re-enabled");
        *self.trigger_reason.lock() = None;
        *self.triggered_at.lock() = None;
    }

    /// Return the current circuit breaker status.
    pub fn status(&self) -> CircuitBreakerStatus {
        let enabled = self.is_trading_enabled();
        let reason = self.trigger_reason.lock().clone();
        CircuitBreakerStatus {
            trading_enabled: enabled,
            triggered: !enabled,
            trigger_reason: reason,
        }
    }

    /// Check the circuit breaker, returning an error if it has been triggered.
    ///
    /// Intended for use inside a [`RiskCheck`](crate::pipeline::RiskCheck)
    /// implementation.
    pub fn check(&self) -> Result<(), RiskReject> {
        if self.is_trading_enabled() {
            Ok(())
        } else {
            let reason = self
                .trigger_reason
                .lock()
                .clone()
                .unwrap_or_else(|| "unknown".to_string());
            Err(RiskReject::CircuitBreakerActive { reason })
        }
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initially_enabled() {
        let cb = CircuitBreaker::new();
        assert!(cb.is_trading_enabled());
    }

    #[test]
    fn test_trigger_disables_trading() {
        let cb = CircuitBreaker::new();
        cb.trigger("test trigger".to_string());
        assert!(!cb.is_trading_enabled());
    }

    #[test]
    fn test_reset_re_enables_trading() {
        let cb = CircuitBreaker::new();
        cb.trigger("test".to_string());
        assert!(!cb.is_trading_enabled());
        cb.reset();
        assert!(cb.is_trading_enabled());
    }

    #[test]
    fn test_check_when_enabled() {
        let cb = CircuitBreaker::new();
        assert!(cb.check().is_ok());
    }

    #[test]
    fn test_check_when_triggered() {
        let cb = CircuitBreaker::new();
        cb.trigger("drawdown exceeded".to_string());
        let result = cb.check();
        assert!(result.is_err());
        match result.unwrap_err() {
            RiskReject::CircuitBreakerActive { reason } => {
                assert_eq!(reason, "drawdown exceeded");
            }
            other => panic!("expected CircuitBreakerActive, got {:?}", other),
        }
    }

    #[test]
    fn test_status_when_enabled() {
        let cb = CircuitBreaker::new();
        let status = cb.status();
        assert!(status.trading_enabled);
        assert!(!status.triggered);
        assert!(status.trigger_reason.is_none());
    }

    #[test]
    fn test_status_when_triggered() {
        let cb = CircuitBreaker::new();
        cb.trigger("test reason".to_string());
        let status = cb.status();
        assert!(!status.trading_enabled);
        assert!(status.triggered);
        assert_eq!(status.trigger_reason, Some("test reason".to_string()));
    }

    #[test]
    fn test_status_after_reset() {
        let cb = CircuitBreaker::new();
        cb.trigger("reason".to_string());
        cb.reset();
        let status = cb.status();
        assert!(status.trading_enabled);
        assert!(!status.triggered);
        assert!(status.trigger_reason.is_none());
    }

    #[test]
    fn test_multiple_triggers() {
        let cb = CircuitBreaker::new();
        cb.trigger("first".to_string());
        cb.trigger("second".to_string());
        let status = cb.status();
        assert!(!status.trading_enabled);
        assert_eq!(status.trigger_reason, Some("second".to_string()));
    }

    #[test]
    fn test_default_is_enabled() {
        let cb = CircuitBreaker::default();
        assert!(cb.is_trading_enabled());
    }
}

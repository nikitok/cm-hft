//! # cm-risk
//!
//! Pre-trade risk management. Every order passes through a chain of risk
//! checks before reaching the exchange gateway. Includes position limits,
//! order size limits, rate limiting, drawdown checks, fat-finger protection,
//! and a circuit breaker with kill switch.

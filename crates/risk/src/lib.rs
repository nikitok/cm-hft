//! # cm-risk
//!
//! Pre-trade risk management. Every order passes through a chain of risk
//! checks before reaching the exchange gateway. Includes position limits,
//! order size limits, rate limiting, drawdown checks, fat-finger protection,
//! and a circuit breaker with kill switch.

pub mod checks;
pub mod circuit_breaker;
pub mod kill_switch;
pub mod pipeline;

pub use checks::{
    DrawdownCheck, FatFingerCheck, MaxOrderSizeCheck, MaxPositionCheck, OrderRateLimitCheck,
};
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerStatus};
pub use kill_switch::{kill_switch_router, KillSwitchState, StatusResponse};
pub use pipeline::{RiskCheck, RiskContext, RiskPipeline, RiskReject};

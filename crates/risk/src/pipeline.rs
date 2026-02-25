//! Risk check pipeline.
//!
//! Every order passes through a sequence of [`RiskCheck`] implementations
//! before it is allowed to reach the exchange gateway. The first check that
//! fails short-circuits the pipeline and returns the rejection reason.

use cm_core::types::*;
use cm_oms::order::Order;
use cm_oms::position::PositionTracker;

/// Context available to risk checks on each order evaluation.
pub struct RiskContext<'a> {
    /// Current position tracker for worst-case position lookups.
    pub position_tracker: &'a PositionTracker,
    /// Current mid price (best bid + best ask) / 2, if available.
    pub current_mid_price: Option<Price>,
    /// Realized + unrealized PnL for the current trading day.
    pub daily_pnl: f64,
    /// Number of currently open (non-terminal) orders.
    pub open_order_count: usize,
}

/// Reason for rejecting an order during pre-trade risk evaluation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum RiskReject {
    /// Resulting position would exceed the configured maximum.
    #[error("max position exceeded: current {current}, order would result in {resulting}, limit {limit}")]
    MaxPosition {
        current: f64,
        resulting: f64,
        limit: f64,
    },
    /// Single order size exceeds the configured limit.
    #[error("max order size exceeded: {size} > {limit}")]
    MaxOrderSize { size: f64, limit: f64 },
    /// Order submission rate exceeds the configured limit.
    #[error("rate limit exceeded: {current} orders/sec, limit {limit}")]
    RateLimit { current: u32, limit: u32 },
    /// Daily PnL drawdown exceeds the configured limit.
    #[error("daily loss limit exceeded: PnL {pnl}, limit {limit}")]
    DailyLossLimit { pnl: f64, limit: f64 },
    /// Order price deviates too far from the current mid price.
    #[error("fat finger: price {price} deviates {deviation_bps} bps from mid {mid}, limit {limit_bps} bps")]
    FatFinger {
        price: f64,
        mid: f64,
        deviation_bps: f64,
        limit_bps: f64,
    },
    /// The circuit breaker has been triggered; all trading is halted.
    #[error("circuit breaker active: {reason}")]
    CircuitBreakerActive { reason: String },
    /// Trading has been disabled (e.g., via kill switch).
    #[error("trading disabled")]
    TradingDisabled,
}

/// Trait for individual pre-trade risk checks.
///
/// Each implementation inspects the order and the current risk context,
/// returning `Ok(())` if the order passes or `Err(RiskReject)` if it
/// should be blocked.
pub trait RiskCheck: Send + Sync {
    /// Human-readable name for logging and diagnostics.
    fn name(&self) -> &str;

    /// Evaluate the order against this risk check.
    fn check(&self, order: &Order, ctx: &RiskContext) -> Result<(), RiskReject>;
}

/// Pipeline that runs all registered risk checks in sequence.
///
/// Checks are evaluated in insertion order. The first failure short-circuits
/// evaluation and returns the rejection.
pub struct RiskPipeline {
    checks: Vec<Box<dyn RiskCheck>>,
}

impl RiskPipeline {
    /// Create a new, empty risk pipeline.
    pub fn new() -> Self {
        Self { checks: Vec::new() }
    }

    /// Append a risk check to the end of the pipeline.
    pub fn add_check(&mut self, check: impl RiskCheck + 'static) {
        self.checks.push(Box::new(check));
    }

    /// Run all checks against the given order and context.
    ///
    /// Returns `Ok(())` if every check passes, or the first `RiskReject`
    /// encountered.
    pub fn check_order(&self, order: &Order, ctx: &RiskContext) -> Result<(), RiskReject> {
        for check in &self.checks {
            check.check(order, ctx)?;
        }
        Ok(())
    }

    /// Returns the number of checks registered in the pipeline.
    pub fn check_count(&self) -> usize {
        self.checks.len()
    }
}

impl Default for RiskPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cm_oms::order::{Order, OrderStatus};
    use cm_oms::position::PositionTracker;

    fn make_test_order() -> Order {
        Order {
            id: OrderId(1),
            client_order_id: "test_001".to_string(),
            exchange_order_id: None,
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            side: Side::Buy,
            order_type: OrderType::Limit,
            price: Price::from(50000.0),
            quantity: Quantity::from(0.1),
            filled_quantity: Quantity::zero(8),
            status: OrderStatus::New,
            created_at: Timestamp::now(),
            updated_at: Timestamp::now(),
        }
    }

    fn make_risk_context(tracker: &PositionTracker) -> RiskContext {
        RiskContext {
            position_tracker: tracker,
            current_mid_price: Some(Price::from(50000.0)),
            daily_pnl: 0.0,
            open_order_count: 0,
        }
    }

    /// A check that always passes.
    struct AlwaysPass;
    impl RiskCheck for AlwaysPass {
        fn name(&self) -> &str {
            "always_pass"
        }
        fn check(&self, _order: &Order, _ctx: &RiskContext) -> Result<(), RiskReject> {
            Ok(())
        }
    }

    /// A check that always rejects.
    struct AlwaysFail;
    impl RiskCheck for AlwaysFail {
        fn name(&self) -> &str {
            "always_fail"
        }
        fn check(&self, _order: &Order, _ctx: &RiskContext) -> Result<(), RiskReject> {
            Err(RiskReject::TradingDisabled)
        }
    }

    #[test]
    fn test_empty_pipeline_passes() {
        let pipeline = RiskPipeline::new();
        let tracker = PositionTracker::new();
        let ctx = make_risk_context(&tracker);
        let order = make_test_order();
        assert!(pipeline.check_order(&order, &ctx).is_ok());
    }

    #[test]
    fn test_all_checks_pass() {
        let mut pipeline = RiskPipeline::new();
        pipeline.add_check(AlwaysPass);
        pipeline.add_check(AlwaysPass);
        pipeline.add_check(AlwaysPass);

        let tracker = PositionTracker::new();
        let ctx = make_risk_context(&tracker);
        let order = make_test_order();
        assert!(pipeline.check_order(&order, &ctx).is_ok());
    }

    #[test]
    fn test_first_check_fails() {
        let mut pipeline = RiskPipeline::new();
        pipeline.add_check(AlwaysFail);
        pipeline.add_check(AlwaysPass);

        let tracker = PositionTracker::new();
        let ctx = make_risk_context(&tracker);
        let order = make_test_order();
        let result = pipeline.check_order(&order, &ctx);
        assert!(matches!(result, Err(RiskReject::TradingDisabled)));
    }

    #[test]
    fn test_second_check_fails() {
        let mut pipeline = RiskPipeline::new();
        pipeline.add_check(AlwaysPass);
        pipeline.add_check(AlwaysFail);

        let tracker = PositionTracker::new();
        let ctx = make_risk_context(&tracker);
        let order = make_test_order();
        let result = pipeline.check_order(&order, &ctx);
        assert!(matches!(result, Err(RiskReject::TradingDisabled)));
    }

    #[test]
    fn test_check_count() {
        let mut pipeline = RiskPipeline::new();
        assert_eq!(pipeline.check_count(), 0);
        pipeline.add_check(AlwaysPass);
        assert_eq!(pipeline.check_count(), 1);
        pipeline.add_check(AlwaysFail);
        assert_eq!(pipeline.check_count(), 2);
    }

    #[test]
    fn test_default_creates_empty() {
        let pipeline = RiskPipeline::default();
        assert_eq!(pipeline.check_count(), 0);
    }
}

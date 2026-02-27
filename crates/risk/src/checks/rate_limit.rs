//! Order rate limit risk check.
//!
//! Tracks recent order timestamps in a sliding window and rejects orders
//! that would exceed the configured per-second or per-minute rate.

use std::collections::VecDeque;
use std::time::Instant;

use cm_oms::order::Order;

use crate::pipeline::{RiskCheck, RiskContext, RiskReject};

/// Rejects orders that exceed the configured submission rate.
///
/// Uses a sliding window of recent order timestamps protected by a
/// `parking_lot::Mutex`. The lock is only held briefly to push/prune
/// timestamps.
pub struct OrderRateLimitCheck {
    max_per_second: u32,
    max_per_minute: u32,
    recent_orders: parking_lot::Mutex<VecDeque<Instant>>,
}

impl OrderRateLimitCheck {
    /// Create a new rate limit check.
    pub fn new(max_per_second: u32, max_per_minute: u32) -> Self {
        Self {
            max_per_second,
            max_per_minute,
            recent_orders: parking_lot::Mutex::new(VecDeque::new()),
        }
    }

    /// Record an order submission timestamp. Call this after an order passes
    /// all risk checks.
    pub fn record_order(&self) {
        self.record_order_at(Instant::now());
    }

    /// Record an order at a specific instant (for testing).
    pub fn record_order_at(&self, now: Instant) {
        let mut orders = self.recent_orders.lock();
        orders.push_back(now);
    }
}

impl RiskCheck for OrderRateLimitCheck {
    fn name(&self) -> &str {
        "rate_limit"
    }

    fn check(&self, _order: &Order, _ctx: &RiskContext) -> Result<(), RiskReject> {
        let now = Instant::now();
        let mut orders = self.recent_orders.lock();

        // Prune entries older than 60 seconds
        let one_minute_ago = now - std::time::Duration::from_secs(60);
        while let Some(&front) = orders.front() {
            if front < one_minute_ago {
                orders.pop_front();
            } else {
                break;
            }
        }

        // Count orders in the last second
        let one_second_ago = now - std::time::Duration::from_secs(1);
        let count_last_second = orders.iter().filter(|&&t| t >= one_second_ago).count() as u32;

        if count_last_second >= self.max_per_second {
            return Err(RiskReject::RateLimit {
                current: count_last_second,
                limit: self.max_per_second,
            });
        }

        // Count orders in the last minute
        let count_last_minute = orders.len() as u32;
        if count_last_minute >= self.max_per_minute {
            return Err(RiskReject::RateLimit {
                current: count_last_minute,
                limit: self.max_per_minute,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cm_core::types::*;
    use cm_oms::order::OrderStatus;
    use cm_oms::position::PositionTracker;

    fn make_order() -> Order {
        Order {
            id: OrderId(1),
            client_order_id: "test".to_string(),
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

    fn make_ctx(tracker: &PositionTracker) -> RiskContext<'_> {
        RiskContext {
            position_tracker: tracker,
            current_mid_price: Some(Price::from(50000.0)),
            daily_pnl: 0.0,
            open_order_count: 0,
        }
    }

    #[test]
    fn test_within_rate_limit() {
        let check = OrderRateLimitCheck::new(5, 100);
        let tracker = PositionTracker::new();
        let ctx = make_ctx(&tracker);
        let order = make_order();

        // No previous orders, should pass
        assert!(check.check(&order, &ctx).is_ok());
    }

    #[test]
    fn test_exceeds_per_second_limit() {
        let check = OrderRateLimitCheck::new(3, 100);
        let tracker = PositionTracker::new();
        let ctx = make_ctx(&tracker);
        let order = make_order();

        let now = Instant::now();
        // Record 3 orders in the last second
        for i in 0..3 {
            check.record_order_at(now - std::time::Duration::from_millis(100 * i));
        }

        let result = check.check(&order, &ctx);
        assert!(matches!(result, Err(RiskReject::RateLimit { .. })));
    }

    #[test]
    fn test_exceeds_per_minute_limit() {
        let check = OrderRateLimitCheck::new(100, 5);
        let tracker = PositionTracker::new();
        let ctx = make_ctx(&tracker);
        let order = make_order();

        let now = Instant::now();
        // Record 5 orders spread over the last minute
        for i in 0..5 {
            check.record_order_at(now - std::time::Duration::from_secs(10 * i));
        }

        let result = check.check(&order, &ctx);
        assert!(matches!(result, Err(RiskReject::RateLimit { .. })));
    }

    #[test]
    fn test_old_orders_pruned() {
        let check = OrderRateLimitCheck::new(3, 100);
        let tracker = PositionTracker::new();
        let ctx = make_ctx(&tracker);
        let order = make_order();

        let now = Instant::now();
        // Record orders more than 60 seconds ago — should be pruned
        for i in 0..10 {
            check.record_order_at(now - std::time::Duration::from_secs(61 + i));
        }

        assert!(check.check(&order, &ctx).is_ok());
    }

    #[test]
    fn test_at_exact_per_second_boundary() {
        let check = OrderRateLimitCheck::new(3, 100);
        let tracker = PositionTracker::new();
        let ctx = make_ctx(&tracker);
        let order = make_order();

        let now = Instant::now();
        // Record exactly (limit - 1) orders, next should pass
        for i in 0..2 {
            check.record_order_at(now - std::time::Duration::from_millis(100 * i));
        }

        assert!(check.check(&order, &ctx).is_ok());
    }
}

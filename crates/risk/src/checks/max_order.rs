//! Maximum order size risk check.
//!
//! Rejects individual orders that exceed the configured size limits.
//! Market orders have a tighter limit than limit/post-only orders.

use cm_core::types::*;
use cm_oms::order::Order;

use crate::pipeline::{RiskCheck, RiskContext, RiskReject};

/// Rejects orders whose size exceeds configured limits.
pub struct MaxOrderSizeCheck {
    /// Maximum size for limit and post-only orders (in base currency).
    pub max_order_size: f64,
    /// Maximum size for market orders (typically tighter).
    pub max_market_order_size: f64,
}

impl RiskCheck for MaxOrderSizeCheck {
    fn name(&self) -> &str {
        "max_order_size"
    }

    fn check(&self, order: &Order, _ctx: &RiskContext) -> Result<(), RiskReject> {
        let size = order.quantity.to_f64();
        let limit = match order.order_type {
            OrderType::Market => self.max_market_order_size,
            OrderType::Limit | OrderType::PostOnly => self.max_order_size,
        };

        if size > limit {
            return Err(RiskReject::MaxOrderSize { size, limit });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cm_oms::order::OrderStatus;
    use cm_oms::position::PositionTracker;

    fn make_order(order_type: OrderType, qty: f64) -> Order {
        Order {
            id: OrderId(1),
            client_order_id: "test".to_string(),
            exchange_order_id: None,
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            side: Side::Buy,
            order_type,
            price: Price::from(50000.0),
            quantity: Quantity::from(qty),
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
    fn test_limit_order_within_limit() {
        let check = MaxOrderSizeCheck {
            max_order_size: 1.0,
            max_market_order_size: 0.5,
        };
        let tracker = PositionTracker::new();
        let ctx = make_ctx(&tracker);
        let order = make_order(OrderType::Limit, 0.5);
        assert!(check.check(&order, &ctx).is_ok());
    }

    #[test]
    fn test_limit_order_exceeds_limit() {
        let check = MaxOrderSizeCheck {
            max_order_size: 1.0,
            max_market_order_size: 0.5,
        };
        let tracker = PositionTracker::new();
        let ctx = make_ctx(&tracker);
        let order = make_order(OrderType::Limit, 1.5);
        let result = check.check(&order, &ctx);
        assert!(matches!(result, Err(RiskReject::MaxOrderSize { .. })));
    }

    #[test]
    fn test_market_order_uses_tighter_limit() {
        let check = MaxOrderSizeCheck {
            max_order_size: 1.0,
            max_market_order_size: 0.5,
        };
        let tracker = PositionTracker::new();
        let ctx = make_ctx(&tracker);

        // 0.6 is within the limit order limit but exceeds market order limit
        let order = make_order(OrderType::Market, 0.6);
        let result = check.check(&order, &ctx);
        assert!(matches!(result, Err(RiskReject::MaxOrderSize { .. })));
    }

    #[test]
    fn test_market_order_within_limit() {
        let check = MaxOrderSizeCheck {
            max_order_size: 1.0,
            max_market_order_size: 0.5,
        };
        let tracker = PositionTracker::new();
        let ctx = make_ctx(&tracker);
        let order = make_order(OrderType::Market, 0.3);
        assert!(check.check(&order, &ctx).is_ok());
    }

    #[test]
    fn test_post_only_uses_limit_threshold() {
        let check = MaxOrderSizeCheck {
            max_order_size: 1.0,
            max_market_order_size: 0.5,
        };
        let tracker = PositionTracker::new();
        let ctx = make_ctx(&tracker);
        let order = make_order(OrderType::PostOnly, 0.8);
        assert!(check.check(&order, &ctx).is_ok());
    }

    #[test]
    fn test_at_exact_boundary() {
        let check = MaxOrderSizeCheck {
            max_order_size: 1.0,
            max_market_order_size: 0.5,
        };
        let tracker = PositionTracker::new();
        let ctx = make_ctx(&tracker);
        let order = make_order(OrderType::Limit, 1.0);
        assert!(check.check(&order, &ctx).is_ok());
    }
}

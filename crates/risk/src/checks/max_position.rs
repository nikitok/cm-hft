//! Maximum position size risk check.
//!
//! Rejects orders that would cause the worst-case position (net position plus
//! all pending order quantities) to exceed the configured limit.

use cm_core::types::*;
use cm_oms::order::Order;

use crate::pipeline::{RiskCheck, RiskContext, RiskReject};

/// Rejects orders that would breach the maximum position size.
pub struct MaxPositionCheck {
    /// Maximum allowed position in base currency (e.g., BTC).
    pub max_position_size: f64,
}

impl RiskCheck for MaxPositionCheck {
    fn name(&self) -> &str {
        "max_position"
    }

    fn check(&self, order: &Order, ctx: &RiskContext) -> Result<(), RiskReject> {
        let worst_case = ctx
            .position_tracker
            .worst_case_position(&order.exchange, &order.symbol);

        let current = worst_case.to_f64();
        let order_qty = order.quantity.to_f64();

        // Calculate the resulting position after this order
        let resulting = match order.side {
            Side::Buy => current + order_qty,
            Side::Sell => current - order_qty,
        };

        if resulting.abs() > self.max_position_size {
            return Err(RiskReject::MaxPosition {
                current: current.abs(),
                resulting: resulting.abs(),
                limit: self.max_position_size,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cm_oms::order::OrderStatus;
    use cm_oms::position::PositionTracker;

    fn make_order(side: Side, qty: f64) -> Order {
        Order {
            id: OrderId(1),
            client_order_id: "test".to_string(),
            exchange_order_id: None,
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            side,
            order_type: OrderType::Limit,
            price: Price::from(50000.0),
            quantity: Quantity::from(qty),
            filled_quantity: Quantity::zero(8),
            status: OrderStatus::New,
            created_at: Timestamp::now(),
            updated_at: Timestamp::now(),
        }
    }

    #[test]
    fn test_within_limit_passes() {
        let check = MaxPositionCheck {
            max_position_size: 1.0,
        };
        let tracker = PositionTracker::new();
        let ctx = RiskContext {
            position_tracker: &tracker,
            current_mid_price: Some(Price::from(50000.0)),
            daily_pnl: 0.0,
            open_order_count: 0,
        };

        let order = make_order(Side::Buy, 0.5);
        assert!(check.check(&order, &ctx).is_ok());
    }

    #[test]
    fn test_exceeds_limit_rejects() {
        let check = MaxPositionCheck {
            max_position_size: 1.0,
        };
        let tracker = PositionTracker::new();
        // Existing position of 0.8 BTC
        tracker.on_fill(
            Exchange::Binance,
            Symbol::new("BTCUSDT"),
            Side::Buy,
            Price::from(50000.0),
            Quantity::from(0.8),
        );

        let ctx = RiskContext {
            position_tracker: &tracker,
            current_mid_price: Some(Price::from(50000.0)),
            daily_pnl: 0.0,
            open_order_count: 0,
        };

        // This buy of 0.5 would bring position to 1.3, exceeding 1.0
        let order = make_order(Side::Buy, 0.5);
        let result = check.check(&order, &ctx);
        assert!(matches!(result, Err(RiskReject::MaxPosition { .. })));
    }

    #[test]
    fn test_at_exact_boundary_passes() {
        let check = MaxPositionCheck {
            max_position_size: 1.0,
        };
        let tracker = PositionTracker::new();
        let ctx = RiskContext {
            position_tracker: &tracker,
            current_mid_price: Some(Price::from(50000.0)),
            daily_pnl: 0.0,
            open_order_count: 0,
        };

        let order = make_order(Side::Buy, 1.0);
        assert!(check.check(&order, &ctx).is_ok());
    }

    #[test]
    fn test_sell_reduces_position() {
        let check = MaxPositionCheck {
            max_position_size: 1.0,
        };
        let tracker = PositionTracker::new();
        tracker.on_fill(
            Exchange::Binance,
            Symbol::new("BTCUSDT"),
            Side::Buy,
            Price::from(50000.0),
            Quantity::from(0.8),
        );

        let ctx = RiskContext {
            position_tracker: &tracker,
            current_mid_price: Some(Price::from(50000.0)),
            daily_pnl: 0.0,
            open_order_count: 0,
        };

        // Selling 0.3 from a 0.8 position -> 0.5, well within limits
        let order = make_order(Side::Sell, 0.3);
        assert!(check.check(&order, &ctx).is_ok());
    }

    #[test]
    fn test_short_position_exceeds_limit() {
        let check = MaxPositionCheck {
            max_position_size: 1.0,
        };
        let tracker = PositionTracker::new();
        let ctx = RiskContext {
            position_tracker: &tracker,
            current_mid_price: Some(Price::from(50000.0)),
            daily_pnl: 0.0,
            open_order_count: 0,
        };

        // Selling 1.5 with no position -> -1.5, abs > 1.0
        let order = make_order(Side::Sell, 1.5);
        let result = check.check(&order, &ctx);
        assert!(matches!(result, Err(RiskReject::MaxPosition { .. })));
    }
}

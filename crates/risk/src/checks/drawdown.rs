//! PnL drawdown risk check.
//!
//! Rejects orders when the daily PnL falls below the configured drawdown
//! limit. This prevents further trading when losses accumulate beyond
//! acceptable thresholds.

use cm_oms::order::Order;

use crate::pipeline::{RiskCheck, RiskContext, RiskReject};

/// Rejects orders when PnL drawdown exceeds configured limits.
pub struct DrawdownCheck {
    /// Maximum hourly drawdown in USD (reserved for future use).
    pub max_hourly_drawdown: f64,
    /// Maximum daily drawdown in USD.
    pub max_daily_drawdown: f64,
}

impl RiskCheck for DrawdownCheck {
    fn name(&self) -> &str {
        "drawdown"
    }

    fn check(&self, _order: &Order, ctx: &RiskContext) -> Result<(), RiskReject> {
        // Daily PnL is negative when losing money. The drawdown limit is a
        // positive number representing the maximum tolerable loss.
        if ctx.daily_pnl < -self.max_daily_drawdown {
            return Err(RiskReject::DailyLossLimit {
                pnl: ctx.daily_pnl,
                limit: self.max_daily_drawdown,
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

    #[test]
    fn test_positive_pnl_passes() {
        let check = DrawdownCheck {
            max_hourly_drawdown: 500.0,
            max_daily_drawdown: 1000.0,
        };
        let tracker = PositionTracker::new();
        let ctx = RiskContext {
            position_tracker: &tracker,
            current_mid_price: Some(Price::from(50000.0)),
            daily_pnl: 500.0,
            open_order_count: 0,
        };
        let order = make_order();
        assert!(check.check(&order, &ctx).is_ok());
    }

    #[test]
    fn test_small_loss_passes() {
        let check = DrawdownCheck {
            max_hourly_drawdown: 500.0,
            max_daily_drawdown: 1000.0,
        };
        let tracker = PositionTracker::new();
        let ctx = RiskContext {
            position_tracker: &tracker,
            current_mid_price: Some(Price::from(50000.0)),
            daily_pnl: -500.0,
            open_order_count: 0,
        };
        let order = make_order();
        assert!(check.check(&order, &ctx).is_ok());
    }

    #[test]
    fn test_exceeds_daily_limit_rejects() {
        let check = DrawdownCheck {
            max_hourly_drawdown: 500.0,
            max_daily_drawdown: 1000.0,
        };
        let tracker = PositionTracker::new();
        let ctx = RiskContext {
            position_tracker: &tracker,
            current_mid_price: Some(Price::from(50000.0)),
            daily_pnl: -1500.0,
            open_order_count: 0,
        };
        let order = make_order();
        let result = check.check(&order, &ctx);
        assert!(matches!(result, Err(RiskReject::DailyLossLimit { .. })));
    }

    #[test]
    fn test_at_exact_boundary_passes() {
        let check = DrawdownCheck {
            max_hourly_drawdown: 500.0,
            max_daily_drawdown: 1000.0,
        };
        let tracker = PositionTracker::new();
        let ctx = RiskContext {
            position_tracker: &tracker,
            current_mid_price: Some(Price::from(50000.0)),
            daily_pnl: -1000.0,
            open_order_count: 0,
        };
        let order = make_order();
        // At exactly the limit (not below), should pass
        assert!(check.check(&order, &ctx).is_ok());
    }

    #[test]
    fn test_zero_pnl_passes() {
        let check = DrawdownCheck {
            max_hourly_drawdown: 500.0,
            max_daily_drawdown: 1000.0,
        };
        let tracker = PositionTracker::new();
        let ctx = RiskContext {
            position_tracker: &tracker,
            current_mid_price: Some(Price::from(50000.0)),
            daily_pnl: 0.0,
            open_order_count: 0,
        };
        let order = make_order();
        assert!(check.check(&order, &ctx).is_ok());
    }
}

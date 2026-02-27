//! Fat-finger protection risk check.
//!
//! Rejects orders whose price deviates too far from the current mid price.
//! Post-only orders can have a separate (typically wider) threshold since
//! they are less likely to cause immediate adverse fills.

use cm_core::types::*;
use cm_oms::order::Order;

use crate::pipeline::{RiskCheck, RiskContext, RiskReject};

/// Rejects orders whose price deviates excessively from the mid price.
pub struct FatFingerCheck {
    /// Maximum price deviation in basis points for limit/market orders.
    pub max_deviation_bps: f64,
    /// Maximum price deviation in basis points for post-only orders.
    pub max_post_only_deviation_bps: f64,
}

impl RiskCheck for FatFingerCheck {
    fn name(&self) -> &str {
        "fat_finger"
    }

    fn check(&self, order: &Order, ctx: &RiskContext) -> Result<(), RiskReject> {
        let mid = match ctx.current_mid_price {
            Some(p) => p.to_f64(),
            // If we don't have a mid price, we can't check — allow the order
            // through (other checks like circuit breaker will catch systemic issues).
            None => return Ok(()),
        };

        if mid == 0.0 {
            return Ok(());
        }

        let order_price = order.price.to_f64();
        let deviation_bps = ((order_price - mid) / mid).abs() * 10_000.0;

        let limit_bps = match order.order_type {
            OrderType::PostOnly => self.max_post_only_deviation_bps,
            OrderType::Limit | OrderType::Market => self.max_deviation_bps,
        };

        if deviation_bps > limit_bps {
            return Err(RiskReject::FatFinger {
                price: order_price,
                mid,
                deviation_bps,
                limit_bps,
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

    fn make_order_at_price(order_type: OrderType, price: f64) -> Order {
        Order {
            id: OrderId(1),
            client_order_id: "test".to_string(),
            exchange_order_id: None,
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            side: Side::Buy,
            order_type,
            price: Price::from(price),
            quantity: Quantity::from(0.1),
            filled_quantity: Quantity::zero(8),
            status: OrderStatus::New,
            created_at: Timestamp::now(),
            updated_at: Timestamp::now(),
        }
    }

    fn make_ctx_with_mid(tracker: &PositionTracker, mid: f64) -> RiskContext<'_> {
        RiskContext {
            position_tracker: tracker,
            current_mid_price: Some(Price::from(mid)),
            daily_pnl: 0.0,
            open_order_count: 0,
        }
    }

    #[test]
    fn test_within_deviation_passes() {
        let check = FatFingerCheck {
            max_deviation_bps: 50.0,
            max_post_only_deviation_bps: 100.0,
        };
        let tracker = PositionTracker::new();
        let ctx = make_ctx_with_mid(&tracker, 50000.0);

        // 0.1% deviation = 10 bps, within 50 bps limit
        let order = make_order_at_price(OrderType::Limit, 50050.0);
        assert!(check.check(&order, &ctx).is_ok());
    }

    #[test]
    fn test_exceeds_deviation_rejects() {
        let check = FatFingerCheck {
            max_deviation_bps: 50.0,
            max_post_only_deviation_bps: 100.0,
        };
        let tracker = PositionTracker::new();
        let ctx = make_ctx_with_mid(&tracker, 50000.0);

        // 1% deviation = 100 bps, exceeds 50 bps limit
        let order = make_order_at_price(OrderType::Limit, 50500.0);
        let result = check.check(&order, &ctx);
        assert!(matches!(result, Err(RiskReject::FatFinger { .. })));
    }

    #[test]
    fn test_post_only_uses_wider_threshold() {
        let check = FatFingerCheck {
            max_deviation_bps: 50.0,
            max_post_only_deviation_bps: 200.0,
        };
        let tracker = PositionTracker::new();
        let ctx = make_ctx_with_mid(&tracker, 50000.0);

        // 100 bps deviation — exceeds limit for regular orders but within PostOnly
        let order = make_order_at_price(OrderType::PostOnly, 50500.0);
        assert!(check.check(&order, &ctx).is_ok());
    }

    #[test]
    fn test_no_mid_price_passes() {
        let check = FatFingerCheck {
            max_deviation_bps: 50.0,
            max_post_only_deviation_bps: 100.0,
        };
        let tracker = PositionTracker::new();
        let ctx = RiskContext {
            position_tracker: &tracker,
            current_mid_price: None,
            daily_pnl: 0.0,
            open_order_count: 0,
        };

        let order = make_order_at_price(OrderType::Limit, 99999.0);
        assert!(check.check(&order, &ctx).is_ok());
    }

    #[test]
    fn test_downward_deviation_also_caught() {
        let check = FatFingerCheck {
            max_deviation_bps: 50.0,
            max_post_only_deviation_bps: 100.0,
        };
        let tracker = PositionTracker::new();
        let ctx = make_ctx_with_mid(&tracker, 50000.0);

        // Price far below mid
        let order = make_order_at_price(OrderType::Limit, 49000.0);
        let result = check.check(&order, &ctx);
        assert!(matches!(result, Err(RiskReject::FatFinger { .. })));
    }

    #[test]
    fn test_at_exact_boundary() {
        let check = FatFingerCheck {
            max_deviation_bps: 100.0,
            max_post_only_deviation_bps: 200.0,
        };
        let tracker = PositionTracker::new();
        let ctx = make_ctx_with_mid(&tracker, 50000.0);

        // Exactly 100 bps = 1% = 500 USD from 50000
        let order = make_order_at_price(OrderType::Limit, 50500.0);
        assert!(check.check(&order, &ctx).is_ok());
    }

    #[test]
    fn test_market_order_uses_standard_threshold() {
        let check = FatFingerCheck {
            max_deviation_bps: 50.0,
            max_post_only_deviation_bps: 200.0,
        };
        let tracker = PositionTracker::new();
        let ctx = make_ctx_with_mid(&tracker, 50000.0);

        // 100 bps - exceeds 50 bps limit for market orders
        let order = make_order_at_price(OrderType::Market, 50500.0);
        let result = check.check(&order, &ctx);
        assert!(matches!(result, Err(RiskReject::FatFinger { .. })));
    }
}

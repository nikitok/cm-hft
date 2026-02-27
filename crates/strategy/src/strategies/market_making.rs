//! Simple market-making strategy that quotes around the mid price.
//!
//! This is a reference implementation for testing the framework, not a
//! production strategy.

use cm_core::types::*;
use cm_market_data::orderbook::OrderBook;

use crate::context::TradingContext;
use crate::traits::{Fill, Strategy, StrategyParams};

/// Simple market-making strategy that quotes around mid price.
///
/// This is a reference implementation for testing the framework,
/// not a production strategy.
pub struct MarketMakingStrategy {
    /// Spread in basis points.
    spread_bps: f64,
    /// Order size in base currency.
    order_size: f64,
    /// Number of levels to quote on each side.
    num_levels: usize,
    /// Reprice threshold in basis points (only move quotes if price moved more than this).
    reprice_threshold_bps: f64,
    /// Inventory skew factor (0.0 = no skew, 1.0 = max skew).
    skew_factor: f64,
    /// Last quoted mid price.
    last_mid: Option<f64>,
}

impl MarketMakingStrategy {
    /// Default spread in basis points.
    const DEFAULT_SPREAD_BPS: f64 = 5.0;
    /// Default order size in base currency.
    const DEFAULT_ORDER_SIZE: f64 = 0.001;
    /// Default number of levels to quote on each side.
    const DEFAULT_NUM_LEVELS: usize = 1;
    /// Default reprice threshold in basis points.
    const DEFAULT_REPRICE_THRESHOLD_BPS: f64 = 2.0;
    /// Default inventory skew factor.
    const DEFAULT_SKEW_FACTOR: f64 = 0.5;

    /// Create a strategy instance from dynamic parameters, using defaults for
    /// any missing keys.
    pub fn from_params(params: &StrategyParams) -> Self {
        Self {
            spread_bps: params
                .get_f64("spread_bps")
                .unwrap_or(Self::DEFAULT_SPREAD_BPS),
            order_size: params
                .get_f64("order_size")
                .unwrap_or(Self::DEFAULT_ORDER_SIZE),
            num_levels: params
                .get_i64("num_levels")
                .map(|v| v.max(1) as usize)
                .unwrap_or(Self::DEFAULT_NUM_LEVELS),
            reprice_threshold_bps: params
                .get_f64("reprice_threshold_bps")
                .unwrap_or(Self::DEFAULT_REPRICE_THRESHOLD_BPS),
            skew_factor: params
                .get_f64("skew_factor")
                .unwrap_or(Self::DEFAULT_SKEW_FACTOR),
            last_mid: None,
        }
    }

    /// Calculate bid and ask prices for a given level, applying inventory skew.
    ///
    /// `net_position` is the current net position as f64 (positive = long).
    /// Returns `(bid_price, ask_price)` as f64.
    fn calculate_quotes(&self, mid: f64, level: usize, net_position: f64) -> (f64, f64) {
        let half_spread = mid * self.spread_bps / 10_000.0 / 2.0;
        let level_offset = half_spread * level as f64;

        // Inventory skew: if long, widen ask spread (less) and tighten bid spread (more)
        // to encourage sells. If short, the opposite.
        let skew = net_position * self.skew_factor * half_spread;

        let bid = mid - half_spread - level_offset - skew;
        let ask = mid + half_spread + level_offset - skew;

        (bid, ask)
    }
}

impl Strategy for MarketMakingStrategy {
    fn on_book_update(&mut self, ctx: &mut TradingContext, book: &OrderBook) {
        let mid = match book.mid_price() {
            Some(p) => p.to_f64(),
            None => return,
        };

        // Check if mid has moved enough to warrant repricing
        if let Some(last) = self.last_mid {
            if last > 0.0 {
                let move_bps = ((mid - last) / last).abs() * 10_000.0;
                if move_bps < self.reprice_threshold_bps {
                    return;
                }
            }
        }

        // Determine the exchange and symbol from the order book
        if book.best_bid().is_none() || book.best_ask().is_none() {
            return;
        }
        let exchange = book.exchange();
        let symbol = book.symbol().clone();

        // Cancel all existing quotes before requoting.
        ctx.cancel_all(Some(exchange), Some(symbol.clone()));

        let net_pos = ctx.net_position(&exchange, &symbol);

        // Submit new quotes at each level
        for level in 0..self.num_levels {
            let (bid_price, ask_price) = self.calculate_quotes(mid, level, net_pos);

            ctx.submit_order(
                exchange,
                symbol.clone(),
                Side::Buy,
                OrderType::PostOnly,
                Price::from(bid_price),
                Quantity::from(self.order_size),
            );

            ctx.submit_order(
                exchange,
                symbol.clone(),
                Side::Sell,
                OrderType::PostOnly,
                Price::from(ask_price),
                Quantity::from(self.order_size),
            );
        }

        self.last_mid = Some(mid);
    }

    fn on_trade(&mut self, _ctx: &mut TradingContext, _trade: &Trade) {
        // No-op for basic market making
    }

    fn on_fill(&mut self, _ctx: &mut TradingContext, fill: &Fill) {
        tracing::debug!(
            order_id = %fill.order_id,
            side = ?fill.side,
            price = %fill.price,
            quantity = %fill.quantity,
            is_maker = fill.is_maker,
            "market making fill"
        );
    }

    fn on_timer(&mut self, _ctx: &mut TradingContext, _timestamp: Timestamp) {
        // Could be used for periodic requoting; no-op for now
    }

    fn on_params_update(&mut self, params: &StrategyParams) {
        if let Some(v) = params.get_f64("spread_bps") {
            self.spread_bps = v;
        }
        if let Some(v) = params.get_f64("order_size") {
            self.order_size = v;
        }
        if let Some(v) = params.get_i64("num_levels") {
            self.num_levels = v.max(1) as usize;
        }
        if let Some(v) = params.get_f64("reprice_threshold_bps") {
            self.reprice_threshold_bps = v;
        }
        if let Some(v) = params.get_f64("skew_factor") {
            self.skew_factor = v;
        }
        // Reset last_mid so next book update triggers a requote
        self.last_mid = None;
    }

    fn name(&self) -> &str {
        "market_making"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::TradingContext;
    use crate::traits::StrategyParams;
    use cm_market_data::orderbook::OrderBook;
    use cm_oms::Position;

    fn default_params() -> StrategyParams {
        StrategyParams {
            params: serde_json::json!({}),
        }
    }

    fn custom_params() -> StrategyParams {
        StrategyParams {
            params: serde_json::json!({
                "spread_bps": 10.0,
                "order_size": 0.01,
                "num_levels": 3,
                "reprice_threshold_bps": 5.0,
                "skew_factor": 0.8
            }),
        }
    }

    fn make_book(bid: f64, ask: f64) -> OrderBook {
        let mut book = OrderBook::new(Exchange::Binance, Symbol::new("BTCUSDT"));
        book.apply_snapshot(
            &[(Price::from(bid), Quantity::from(1.0))],
            &[(Price::from(ask), Quantity::from(1.0))],
            1,
        );
        book
    }

    fn make_context() -> TradingContext {
        TradingContext::new(vec![], vec![], Timestamp::from_millis(1000))
    }

    fn make_context_with_position(net_qty: f64) -> TradingContext {
        let pos = Position {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            net_quantity: Quantity::from(net_qty),
            avg_entry_price: Price::from(50000.0),
            realized_pnl: Price::zero(8),
            fill_count: 1,
        };
        TradingContext::new(vec![pos], vec![], Timestamp::from_millis(1000))
    }

    // ── from_params tests ──

    #[test]
    fn test_from_params_defaults() {
        let strat = MarketMakingStrategy::from_params(&default_params());
        assert!((strat.spread_bps - 5.0).abs() < 1e-10);
        assert!((strat.order_size - 0.001).abs() < 1e-10);
        assert_eq!(strat.num_levels, 1);
        assert!((strat.reprice_threshold_bps - 2.0).abs() < 1e-10);
        assert!((strat.skew_factor - 0.5).abs() < 1e-10);
        assert!(strat.last_mid.is_none());
    }

    #[test]
    fn test_from_params_custom() {
        let strat = MarketMakingStrategy::from_params(&custom_params());
        assert!((strat.spread_bps - 10.0).abs() < 1e-10);
        assert!((strat.order_size - 0.01).abs() < 1e-10);
        assert_eq!(strat.num_levels, 3);
        assert!((strat.reprice_threshold_bps - 5.0).abs() < 1e-10);
        assert!((strat.skew_factor - 0.8).abs() < 1e-10);
    }

    // ── on_book_update tests ──

    #[test]
    fn test_on_book_update_generates_orders() {
        let mut strat = MarketMakingStrategy::from_params(&default_params());
        let book = make_book(50000.0, 50001.0);
        let mut ctx = make_context();

        strat.on_book_update(&mut ctx, &book);

        // cancel_all + 1 bid + 1 ask = 3 actions
        assert_eq!(ctx.action_count(), 3);
    }

    #[test]
    fn test_on_book_update_multi_level() {
        let params = StrategyParams {
            params: serde_json::json!({ "num_levels": 3 }),
        };
        let mut strat = MarketMakingStrategy::from_params(&params);
        let book = make_book(50000.0, 50001.0);
        let mut ctx = make_context();

        strat.on_book_update(&mut ctx, &book);

        // cancel_all + 3 levels * 2 sides = 7 actions
        assert_eq!(ctx.action_count(), 7);
    }

    #[test]
    fn test_spread_calculation() {
        let params = StrategyParams {
            params: serde_json::json!({
                "spread_bps": 10.0,
                "num_levels": 1
            }),
        };
        let mut strat = MarketMakingStrategy::from_params(&params);
        let book = make_book(50000.0, 50001.0);
        let mut ctx = make_context();

        strat.on_book_update(&mut ctx, &book);

        let actions = ctx.drain_actions();
        let submits: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, crate::context::OrderAction::Submit { .. }))
            .collect();
        assert_eq!(submits.len(), 2);

        // Mid is ~50000.5
        let mid = 50000.5;
        let half_spread = mid * 10.0 / 10_000.0 / 2.0; // ~25.00025

        // First submit should be a Buy (bid)
        if let crate::context::OrderAction::Submit { price, side, .. } = submits[0] {
            assert_eq!(*side, Side::Buy);
            let expected_bid = mid - half_spread;
            let actual = price.to_f64();
            assert!(
                (actual - expected_bid).abs() < 0.1,
                "bid: expected ~{expected_bid}, got {actual}"
            );
        } else {
            panic!("expected Submit action");
        }

        // Second submit should be a Sell (ask)
        if let crate::context::OrderAction::Submit { price, side, .. } = submits[1] {
            assert_eq!(*side, Side::Sell);
            let expected_ask = mid + half_spread;
            let actual = price.to_f64();
            assert!(
                (actual - expected_ask).abs() < 0.1,
                "ask: expected ~{expected_ask}, got {actual}"
            );
        } else {
            panic!("expected Submit action");
        }
    }

    #[test]
    fn test_inventory_skew_long() {
        let params = StrategyParams {
            params: serde_json::json!({
                "spread_bps": 10.0,
                "skew_factor": 1.0
            }),
        };
        let mut strat = MarketMakingStrategy::from_params(&params);
        let book = make_book(50000.0, 50001.0);

        // With a long position, bid should be lower and ask should be lower
        // (skew pushes quotes down to encourage selling)
        let mut ctx_flat = make_context();
        strat.on_book_update(&mut ctx_flat, &book);
        let flat_actions = ctx_flat.drain_actions();

        // Reset last_mid so it requotes
        strat.last_mid = None;

        let mut ctx_long = make_context_with_position(1.0);
        strat.on_book_update(&mut ctx_long, &book);
        let long_actions = ctx_long.drain_actions();

        // Filter to Submit actions only (skip CancelAll)
        let flat_submits: Vec<_> = flat_actions
            .iter()
            .filter(|a| matches!(a, crate::context::OrderAction::Submit { .. }))
            .collect();
        let long_submits: Vec<_> = long_actions
            .iter()
            .filter(|a| matches!(a, crate::context::OrderAction::Submit { .. }))
            .collect();

        // Extract bid prices
        let flat_bid = if let crate::context::OrderAction::Submit { price, .. } = flat_submits[0] {
            price.to_f64()
        } else {
            panic!("expected Submit");
        };

        let long_bid = if let crate::context::OrderAction::Submit { price, .. } = long_submits[0] {
            price.to_f64()
        } else {
            panic!("expected Submit");
        };

        // When long, bid should be lower (less aggressive buying)
        assert!(
            long_bid < flat_bid,
            "long bid {long_bid} should be < flat bid {flat_bid}"
        );

        // Extract ask prices
        let flat_ask = if let crate::context::OrderAction::Submit { price, .. } = flat_submits[1] {
            price.to_f64()
        } else {
            panic!("expected Submit");
        };

        let long_ask = if let crate::context::OrderAction::Submit { price, .. } = long_submits[1] {
            price.to_f64()
        } else {
            panic!("expected Submit");
        };

        // When long, ask should also be lower (more aggressive selling)
        assert!(
            long_ask < flat_ask,
            "long ask {long_ask} should be < flat ask {flat_ask}"
        );
    }

    #[test]
    fn test_reprice_threshold_prevents_requote() {
        let params = StrategyParams {
            params: serde_json::json!({
                "reprice_threshold_bps": 10.0
            }),
        };
        let mut strat = MarketMakingStrategy::from_params(&params);

        // First update -- always quotes (cancel_all + bid + ask)
        let book1 = make_book(50000.0, 50001.0);
        let mut ctx1 = make_context();
        strat.on_book_update(&mut ctx1, &book1);
        assert_eq!(ctx1.action_count(), 3);

        // Second update with tiny price change -- should NOT requote
        // 10 bps of 50000 = 50, so a move of ~1 is well under threshold
        let book2 = make_book(50000.5, 50001.5);
        let mut ctx2 = make_context();
        strat.on_book_update(&mut ctx2, &book2);
        assert_eq!(ctx2.action_count(), 0, "should not requote on small move");
    }

    #[test]
    fn test_reprice_threshold_allows_requote_on_big_move() {
        let params = StrategyParams {
            params: serde_json::json!({
                "reprice_threshold_bps": 2.0
            }),
        };
        let mut strat = MarketMakingStrategy::from_params(&params);

        let book1 = make_book(50000.0, 50001.0);
        let mut ctx1 = make_context();
        strat.on_book_update(&mut ctx1, &book1);
        assert_eq!(ctx1.action_count(), 3); // cancel_all + bid + ask

        // Big move: ~100 bps
        let book2 = make_book(50500.0, 50501.0);
        let mut ctx2 = make_context();
        strat.on_book_update(&mut ctx2, &book2);
        assert_eq!(ctx2.action_count(), 3, "should requote on big move");
    }

    #[test]
    fn test_on_params_update_changes_behavior() {
        let mut strat = MarketMakingStrategy::from_params(&default_params());

        // Verify defaults
        assert!((strat.spread_bps - 5.0).abs() < 1e-10);

        // Update params
        let new_params = StrategyParams {
            params: serde_json::json!({
                "spread_bps": 20.0,
                "order_size": 0.1
            }),
        };
        strat.on_params_update(&new_params);

        assert!((strat.spread_bps - 20.0).abs() < 1e-10);
        assert!((strat.order_size - 0.1).abs() < 1e-10);
        // last_mid should be reset so next update requotes
        assert!(strat.last_mid.is_none());
    }

    #[test]
    fn test_on_fill_is_handled() {
        let mut strat = MarketMakingStrategy::from_params(&default_params());
        let fill = Fill {
            order_id: OrderId(1),
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            side: Side::Buy,
            price: Price::from(50000.0),
            quantity: Quantity::from(0.001),
            timestamp: Timestamp::from_millis(1000),
            is_maker: true,
        };

        let mut ctx = make_context();
        strat.on_fill(&mut ctx, &fill);
        // on_fill is a no-op for simple_mm (cancel_all handles cleanup)
    }

    #[test]
    fn test_name() {
        let strat = MarketMakingStrategy::from_params(&default_params());
        assert_eq!(strat.name(), "market_making");
    }

    #[test]
    fn test_empty_book_no_action() {
        let mut strat = MarketMakingStrategy::from_params(&default_params());
        let book = OrderBook::new(Exchange::Binance, Symbol::new("BTCUSDT"));
        let mut ctx = make_context();

        strat.on_book_update(&mut ctx, &book);
        assert_eq!(ctx.action_count(), 0, "empty book should produce no orders");
    }

    #[test]
    fn test_on_trade_no_op() {
        let mut strat = MarketMakingStrategy::from_params(&default_params());
        let mut ctx = make_context();
        let trade = Trade {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            timestamp: Timestamp::from_millis(1000),
            price: Price::from(50000.0),
            quantity: Quantity::from(1.0),
            side: Side::Buy,
            trade_id: "1".to_string(),
        };
        strat.on_trade(&mut ctx, &trade);
        assert_eq!(ctx.action_count(), 0);
    }

    #[test]
    fn test_on_timer_no_op() {
        let mut strat = MarketMakingStrategy::from_params(&default_params());
        let mut ctx = make_context();
        strat.on_timer(&mut ctx, Timestamp::from_millis(1000));
        assert_eq!(ctx.action_count(), 0);
    }

    #[test]
    fn test_inventory_skew_short() {
        let params = StrategyParams {
            params: serde_json::json!({
                "spread_bps": 10.0,
                "skew_factor": 1.0
            }),
        };
        let mut strat = MarketMakingStrategy::from_params(&params);
        let book = make_book(50000.0, 50001.0);

        // Flat position
        let mut ctx_flat = make_context();
        strat.on_book_update(&mut ctx_flat, &book);
        let flat_actions = ctx_flat.drain_actions();

        strat.last_mid = None;

        // Short position
        let mut ctx_short = make_context_with_position(-1.0);
        strat.on_book_update(&mut ctx_short, &book);
        let short_actions = ctx_short.drain_actions();

        let flat_submits: Vec<_> = flat_actions
            .iter()
            .filter(|a| matches!(a, crate::context::OrderAction::Submit { .. }))
            .collect();
        let short_submits: Vec<_> = short_actions
            .iter()
            .filter(|a| matches!(a, crate::context::OrderAction::Submit { .. }))
            .collect();

        let flat_bid = if let crate::context::OrderAction::Submit { price, .. } = flat_submits[0] {
            price.to_f64()
        } else {
            panic!("expected Submit");
        };

        let short_bid = if let crate::context::OrderAction::Submit { price, .. } = short_submits[0]
        {
            price.to_f64()
        } else {
            panic!("expected Submit");
        };

        // When short, bid should be higher (more aggressive buying to reduce short)
        assert!(
            short_bid > flat_bid,
            "short bid {short_bid} should be > flat bid {flat_bid}"
        );
    }
}

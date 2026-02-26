//! Adaptive market-making strategy with volatility scaling, trade-flow signals,
//! book imbalance, quadratic inventory skew, and hard position limits.

use cm_core::types::*;
use cm_market_data::orderbook::OrderBook;

use crate::context::TradingContext;
use crate::traits::{Fill, Strategy, StrategyParams};

use super::signals::{Ema, TradeFlowSignal, VolatilityTracker};

/// Adaptive market maker with 7 improvements over the basic MM strategy.
pub struct AdaptiveMarketMaker {
    // ── Parameters ──
    base_spread_bps: f64,
    order_size: f64,
    num_levels: usize,
    max_position: f64,
    skew_intensity: f64,
    min_spread_bps: f64,
    trade_flow_factor: f64,
    book_imbalance_factor: f64,
    imbalance_depth: usize,
    vol_baseline_bps: f64,
    reprice_threshold_bps: f64,

    // ── Signal state ──
    vol_tracker: VolatilityTracker,
    trade_flow: TradeFlowSignal,
    fair_value_ema: Ema,

    // ── Quoting state ──
    last_quoted_mid: Option<f64>,
    needs_requote: bool,
    active_bids: Vec<OrderId>,
    active_asks: Vec<OrderId>,
}

impl AdaptiveMarketMaker {
    const DEFAULT_BASE_SPREAD_BPS: f64 = 8.0;
    const DEFAULT_ORDER_SIZE: f64 = 0.001;
    const DEFAULT_NUM_LEVELS: usize = 1;
    const DEFAULT_MAX_POSITION: f64 = 0.01;
    const DEFAULT_SKEW_INTENSITY: f64 = 2.0;
    const DEFAULT_MIN_SPREAD_BPS: f64 = 2.0;
    const DEFAULT_TRADE_FLOW_FACTOR: f64 = 0.3;
    const DEFAULT_BOOK_IMBALANCE_FACTOR: f64 = 0.2;
    const DEFAULT_IMBALANCE_DEPTH: usize = 3;
    const DEFAULT_VOL_BASELINE_BPS: f64 = 2.0;
    const DEFAULT_VOL_EMA_SPAN: usize = 100;
    const DEFAULT_TRADE_FLOW_SPAN: usize = 50;
    const DEFAULT_REPRICE_THRESHOLD_BPS: f64 = 1.5;

    pub fn from_params(params: &StrategyParams) -> Self {
        let vol_ema_span = params
            .get_i64("vol_ema_span")
            .map(|v| v.max(2) as usize)
            .unwrap_or(Self::DEFAULT_VOL_EMA_SPAN);
        let trade_flow_span = params
            .get_i64("trade_flow_span")
            .map(|v| v.max(2) as usize)
            .unwrap_or(Self::DEFAULT_TRADE_FLOW_SPAN);

        Self {
            base_spread_bps: params
                .get_f64("base_spread_bps")
                .unwrap_or(Self::DEFAULT_BASE_SPREAD_BPS),
            order_size: params
                .get_f64("order_size")
                .unwrap_or(Self::DEFAULT_ORDER_SIZE),
            num_levels: params
                .get_i64("num_levels")
                .map(|v| v.max(1) as usize)
                .unwrap_or(Self::DEFAULT_NUM_LEVELS),
            max_position: params
                .get_f64("max_position")
                .unwrap_or(Self::DEFAULT_MAX_POSITION),
            skew_intensity: params
                .get_f64("skew_intensity")
                .unwrap_or(Self::DEFAULT_SKEW_INTENSITY),
            min_spread_bps: params
                .get_f64("min_spread_bps")
                .unwrap_or(Self::DEFAULT_MIN_SPREAD_BPS),
            trade_flow_factor: params
                .get_f64("trade_flow_factor")
                .unwrap_or(Self::DEFAULT_TRADE_FLOW_FACTOR),
            book_imbalance_factor: params
                .get_f64("book_imbalance_factor")
                .unwrap_or(Self::DEFAULT_BOOK_IMBALANCE_FACTOR),
            imbalance_depth: params
                .get_i64("imbalance_depth")
                .map(|v| v.max(1) as usize)
                .unwrap_or(Self::DEFAULT_IMBALANCE_DEPTH),
            vol_baseline_bps: params
                .get_f64("vol_baseline_bps")
                .unwrap_or(Self::DEFAULT_VOL_BASELINE_BPS),
            reprice_threshold_bps: params
                .get_f64("reprice_threshold_bps")
                .unwrap_or(Self::DEFAULT_REPRICE_THRESHOLD_BPS),
            vol_tracker: VolatilityTracker::new(vol_ema_span),
            trade_flow: TradeFlowSignal::new(trade_flow_span),
            fair_value_ema: Ema::new(20),
            last_quoted_mid: None,
            needs_requote: false,
            active_bids: Vec::new(),
            active_asks: Vec::new(),
        }
    }

    /// Compute book imbalance from top `depth` levels.
    /// Returns a value in `[-1, +1]`: positive means bid-heavy (bullish).
    fn book_imbalance(&self, book: &OrderBook) -> f64 {
        let bids = book.bid_depth(self.imbalance_depth);
        let asks = book.ask_depth(self.imbalance_depth);

        let bid_vol: f64 = bids.iter().map(|l| l.quantity.to_f64()).sum();
        let ask_vol: f64 = asks.iter().map(|l| l.quantity.to_f64()).sum();

        let total = bid_vol + ask_vol;
        if total < 1e-12 {
            return 0.0;
        }
        (bid_vol - ask_vol) / total
    }

    /// Quadratic inventory skew: `sgn(r) * r^2 * intensity * half_spread`.
    fn inventory_skew(&self, net_position: f64, half_spread: f64) -> f64 {
        if self.max_position < 1e-12 {
            return 0.0;
        }
        let r = (net_position / self.max_position).clamp(-1.0, 1.0);
        r.signum() * r * r * self.skew_intensity * half_spread
    }
}

impl Strategy for AdaptiveMarketMaker {
    fn on_book_update(&mut self, ctx: &mut TradingContext, book: &OrderBook) {
        let mid = match book.mid_price() {
            Some(p) => p.to_f64(),
            None => return,
        };

        // 1. Update signals.
        self.vol_tracker.update(mid);
        let fair_value = self.fair_value_ema.update(mid);

        // 2. Check reprice threshold (bypass if fill triggered needs_requote).
        if !self.needs_requote {
            if let Some(last) = self.last_quoted_mid {
                if last > 0.0 {
                    let move_bps = ((mid - last) / last).abs() * 10_000.0;
                    if move_bps < self.reprice_threshold_bps {
                        return;
                    }
                }
            }
        }
        self.needs_requote = false;

        // 3. Cancel all existing orders.
        for id in self.active_bids.drain(..) {
            ctx.cancel_order(id);
        }
        for id in self.active_asks.drain(..) {
            ctx.cancel_order(id);
        }

        if book.best_bid().is_none() || book.best_ask().is_none() {
            return;
        }
        let exchange = book.exchange();
        let symbol = book.symbol().clone();
        let net_pos = ctx.net_position(&exchange, &symbol);

        // 4. Volatility-adaptive spread.
        let vol_bps = self.vol_tracker.volatility_bps();
        let vol_ratio = if self.vol_baseline_bps > 0.0 && vol_bps > 0.0 {
            (vol_bps / self.vol_baseline_bps).max(0.5).min(3.0)
        } else {
            1.0
        };
        let base_half = mid * self.base_spread_bps / 10_000.0 / 2.0;
        let half_spread = (base_half * vol_ratio).max(mid * self.min_spread_bps / 10_000.0 / 2.0);

        // 5. Fair value adjustments.
        let trade_flow_adj =
            self.trade_flow.imbalance() * self.trade_flow_factor * half_spread;
        let book_imb_adj =
            self.book_imbalance(book) * self.book_imbalance_factor * half_spread;
        let adjusted_mid = fair_value + trade_flow_adj + book_imb_adj;

        // 6. Inventory skew.
        let skew = self.inventory_skew(net_pos, half_spread);

        // 7. Position limit check.
        let at_max_long = net_pos >= self.max_position - 1e-12;
        let at_max_short = net_pos <= -self.max_position + 1e-12;

        // 8. Submit quotes.
        for level in 0..self.num_levels {
            let level_offset = half_spread * level as f64;

            if !at_max_long {
                let bid = adjusted_mid - half_spread - level_offset - skew;
                ctx.submit_order(
                    exchange,
                    symbol.clone(),
                    Side::Buy,
                    OrderType::PostOnly,
                    Price::from(bid),
                    Quantity::from(self.order_size),
                );
            }

            if !at_max_short {
                let ask = adjusted_mid + half_spread + level_offset - skew;
                ctx.submit_order(
                    exchange,
                    symbol.clone(),
                    Side::Sell,
                    OrderType::PostOnly,
                    Price::from(ask),
                    Quantity::from(self.order_size),
                );
            }
        }

        self.last_quoted_mid = Some(mid);
    }

    fn on_trade(&mut self, _ctx: &mut TradingContext, trade: &Trade) {
        let price = trade.price.to_f64();
        let qty = trade.quantity.to_f64();
        let notional = price * qty;
        let is_buy = trade.side == Side::Buy;
        self.trade_flow.update(is_buy, notional);
    }

    fn on_fill(&mut self, _ctx: &mut TradingContext, fill: &Fill) {
        self.active_bids.retain(|id| *id != fill.order_id);
        self.active_asks.retain(|id| *id != fill.order_id);
        // Trigger immediate requote on next book update.
        self.needs_requote = true;

        tracing::debug!(
            order_id = %fill.order_id,
            side = ?fill.side,
            price = %fill.price,
            quantity = %fill.quantity,
            "adaptive_mm fill"
        );
    }

    fn on_timer(&mut self, _ctx: &mut TradingContext, _timestamp: Timestamp) {}

    fn on_params_update(&mut self, params: &StrategyParams) {
        if let Some(v) = params.get_f64("base_spread_bps") {
            self.base_spread_bps = v;
        }
        if let Some(v) = params.get_f64("order_size") {
            self.order_size = v;
        }
        if let Some(v) = params.get_i64("num_levels") {
            self.num_levels = v.max(1) as usize;
        }
        if let Some(v) = params.get_f64("max_position") {
            self.max_position = v;
        }
        if let Some(v) = params.get_f64("skew_intensity") {
            self.skew_intensity = v;
        }
        if let Some(v) = params.get_f64("min_spread_bps") {
            self.min_spread_bps = v;
        }
        if let Some(v) = params.get_f64("trade_flow_factor") {
            self.trade_flow_factor = v;
        }
        if let Some(v) = params.get_f64("book_imbalance_factor") {
            self.book_imbalance_factor = v;
        }
        if let Some(v) = params.get_i64("imbalance_depth") {
            self.imbalance_depth = v.max(1) as usize;
        }
        if let Some(v) = params.get_f64("vol_baseline_bps") {
            self.vol_baseline_bps = v;
        }
        if let Some(v) = params.get_f64("reprice_threshold_bps") {
            self.reprice_threshold_bps = v;
        }
        self.last_quoted_mid = None;
    }

    fn name(&self) -> &str {
        "adaptive_mm"
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

    fn make_book(bid: f64, ask: f64) -> OrderBook {
        let mut book = OrderBook::new(Exchange::Binance, Symbol::new("BTCUSDT"));
        book.apply_snapshot(
            &[(Price::from(bid), Quantity::from(1.0))],
            &[(Price::from(ask), Quantity::from(1.0))],
            1,
        );
        book
    }

    fn make_book_with_depth(bid: f64, ask: f64) -> OrderBook {
        let mut book = OrderBook::new(Exchange::Binance, Symbol::new("BTCUSDT"));
        let bids = vec![
            (Price::from(bid), Quantity::from(2.0)),
            (Price::from(bid - 1.0), Quantity::from(1.0)),
            (Price::from(bid - 2.0), Quantity::from(0.5)),
        ];
        let asks = vec![
            (Price::from(ask), Quantity::from(1.0)),
            (Price::from(ask + 1.0), Quantity::from(1.0)),
            (Price::from(ask + 2.0), Quantity::from(0.5)),
        ];
        book.apply_snapshot(&bids, &asks, 1);
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

    #[test]
    fn test_from_params_defaults() {
        let strat = AdaptiveMarketMaker::from_params(&default_params());
        assert!((strat.base_spread_bps - 8.0).abs() < 1e-10);
        assert!((strat.order_size - 0.001).abs() < 1e-10);
        assert_eq!(strat.num_levels, 1);
        assert!((strat.max_position - 0.01).abs() < 1e-10);
        assert!((strat.skew_intensity - 2.0).abs() < 1e-10);
        assert!((strat.min_spread_bps - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_generates_orders() {
        let mut strat = AdaptiveMarketMaker::from_params(&default_params());
        let book = make_book(50000.0, 50001.0);
        let mut ctx = make_context();
        strat.on_book_update(&mut ctx, &book);
        // 1 level: 1 bid + 1 ask = 2
        assert_eq!(ctx.action_count(), 2);
    }

    #[test]
    fn test_position_limit_blocks_buys() {
        let params = StrategyParams {
            params: serde_json::json!({"max_position": 0.01}),
        };
        let mut strat = AdaptiveMarketMaker::from_params(&params);
        let book = make_book(50000.0, 50001.0);

        // At max long position — should only quote asks
        let mut ctx = make_context_with_position(0.01);
        strat.on_book_update(&mut ctx, &book);
        let actions = ctx.drain_actions();
        assert_eq!(actions.len(), 1);
        if let crate::context::OrderAction::Submit { side, .. } = &actions[0] {
            assert_eq!(*side, Side::Sell);
        } else {
            panic!("expected Submit");
        }
    }

    #[test]
    fn test_position_limit_blocks_sells() {
        let params = StrategyParams {
            params: serde_json::json!({"max_position": 0.01}),
        };
        let mut strat = AdaptiveMarketMaker::from_params(&params);
        let book = make_book(50000.0, 50001.0);

        // At max short position — should only quote bids
        let mut ctx = make_context_with_position(-0.01);
        strat.on_book_update(&mut ctx, &book);
        let actions = ctx.drain_actions();
        assert_eq!(actions.len(), 1);
        if let crate::context::OrderAction::Submit { side, .. } = &actions[0] {
            assert_eq!(*side, Side::Buy);
        } else {
            panic!("expected Submit");
        }
    }

    #[test]
    fn test_quadratic_skew_long() {
        let params = StrategyParams {
            params: serde_json::json!({
                "max_position": 1.0,
                "skew_intensity": 2.0,
                "base_spread_bps": 10.0,
            }),
        };
        let mut strat = AdaptiveMarketMaker::from_params(&params);
        let book = make_book(50000.0, 50001.0);

        // Flat — get baseline
        let mut ctx_flat = make_context();
        strat.on_book_update(&mut ctx_flat, &book);
        let flat_actions = ctx_flat.drain_actions();

        strat.last_quoted_mid = None;

        // Long 0.5 — skew should push quotes down (encourage selling)
        let mut ctx_long = make_context_with_position(0.5);
        strat.on_book_update(&mut ctx_long, &book);
        let long_actions = ctx_long.drain_actions();

        let flat_ask = match &flat_actions[1] {
            crate::context::OrderAction::Submit { price, .. } => price.to_f64(),
            _ => panic!("expected Submit"),
        };
        let long_ask = match &long_actions[1] {
            crate::context::OrderAction::Submit { price, .. } => price.to_f64(),
            _ => panic!("expected Submit"),
        };
        // When long, skew is positive, so ask = mid + half - skew → lower
        assert!(long_ask < flat_ask, "long ask {long_ask} should be < flat ask {flat_ask}");
    }

    #[test]
    fn test_fill_triggers_requote() {
        let mut strat = AdaptiveMarketMaker::from_params(&default_params());
        let book = make_book(50000.0, 50001.0);

        // First update
        let mut ctx1 = make_context();
        strat.on_book_update(&mut ctx1, &book);
        assert_eq!(ctx1.action_count(), 2);

        // Tiny move — normally would NOT requote
        let book2 = make_book(50000.01, 50001.01);
        let mut ctx2 = make_context();
        strat.on_book_update(&mut ctx2, &book2);
        assert_eq!(ctx2.action_count(), 0, "should skip tiny move");

        // Simulate fill → needs_requote
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
        let mut fill_ctx = make_context();
        strat.on_fill(&mut fill_ctx, &fill);

        // Now even tiny move should requote
        let book3 = make_book(50000.02, 50001.02);
        let mut ctx3 = make_context();
        strat.on_book_update(&mut ctx3, &book3);
        assert!(ctx3.action_count() >= 2, "should requote after fill");
    }

    #[test]
    fn test_trade_flow_signal_updates() {
        let mut strat = AdaptiveMarketMaker::from_params(&default_params());
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
        assert!(strat.trade_flow.imbalance() > 0.0);
    }

    #[test]
    fn test_book_imbalance() {
        let strat = AdaptiveMarketMaker::from_params(&default_params());
        let book = make_book_with_depth(50000.0, 50001.0);
        // bids: 2.0 + 1.0 + 0.5 = 3.5, asks: 1.0 + 1.0 + 0.5 = 2.5
        let imb = strat.book_imbalance(&book);
        assert!(imb > 0.0, "bid-heavy book should give positive imbalance");
    }

    #[test]
    fn test_name() {
        let strat = AdaptiveMarketMaker::from_params(&default_params());
        assert_eq!(strat.name(), "adaptive_mm");
    }

    #[test]
    fn test_on_params_update() {
        let mut strat = AdaptiveMarketMaker::from_params(&default_params());
        let new_params = StrategyParams {
            params: serde_json::json!({
                "base_spread_bps": 15.0,
                "max_position": 0.05,
            }),
        };
        strat.on_params_update(&new_params);
        assert!((strat.base_spread_bps - 15.0).abs() < 1e-10);
        assert!((strat.max_position - 0.05).abs() < 1e-10);
        assert!(strat.last_quoted_mid.is_none());
    }
}

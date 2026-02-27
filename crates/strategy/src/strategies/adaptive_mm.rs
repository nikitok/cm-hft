//! Adaptive market-making strategy with Avellaneda-Stoikov pricing, VPIN-based
//! spread adjustment, asymmetric sizing, and volatility-adaptive quoting.

use cm_core::types::*;
use cm_market_data::orderbook::OrderBook;

use crate::context::TradingContext;
use crate::traits::{Fill, Strategy, StrategyParams};

use super::signals::{Ema, TradeFlowSignal, VolatilityTracker, VpinTracker};

/// Adaptive market maker with A-S pricing, VPIN, and asymmetric sizing.
pub struct AdaptiveMarketMaker {
    // ── Parameters ──
    base_spread_bps: f64,
    order_size: f64,
    num_levels: usize,
    max_position: f64,
    min_spread_bps: f64,
    trade_flow_factor: f64,
    book_imbalance_factor: f64,
    imbalance_depth: usize,
    vol_baseline_bps: f64,
    reprice_threshold_bps: f64,

    // ── A-S parameters ──
    risk_aversion: f64,  // γ — inventory penalty intensity
    fill_intensity: f64, // κ — expected order arrival rate
    time_horizon: f64,   // τ — effective time horizon in ticks

    // ── VPIN parameters ──
    vpin_factor: f64, // spread multiplier: 1 + vpin_factor * vpin

    // ── Asymmetric sizing parameters ──
    size_decay_power: f64, // how fast accumulating side shrinks
    reduce_boost: f64,     // max increase for reducing side

    // ── Signal state ──
    vol_tracker: VolatilityTracker,
    trade_flow: TradeFlowSignal,
    fair_value_ema: Ema,
    vpin_tracker: VpinTracker,

    // ── Quoting state ──
    last_quoted_mid: Option<f64>,
    needs_requote: bool,
}

impl AdaptiveMarketMaker {
    const DEFAULT_BASE_SPREAD_BPS: f64 = 8.0;
    const DEFAULT_ORDER_SIZE: f64 = 0.001;
    const DEFAULT_NUM_LEVELS: usize = 1;
    const DEFAULT_MAX_POSITION: f64 = 0.1;
    const DEFAULT_MIN_SPREAD_BPS: f64 = 2.0;
    const DEFAULT_TRADE_FLOW_FACTOR: f64 = 0.3;
    const DEFAULT_BOOK_IMBALANCE_FACTOR: f64 = 0.2;
    const DEFAULT_IMBALANCE_DEPTH: usize = 3;
    const DEFAULT_VOL_BASELINE_BPS: f64 = 2.0;
    const DEFAULT_VOL_EMA_SPAN: usize = 100;
    const DEFAULT_TRADE_FLOW_SPAN: usize = 50;
    const DEFAULT_REPRICE_THRESHOLD_BPS: f64 = 1.5;

    // A-S defaults
    const DEFAULT_RISK_AVERSION: f64 = 0.3;
    const DEFAULT_FILL_INTENSITY: f64 = 1.5;
    const DEFAULT_TIME_HORIZON: f64 = 1.0;

    // VPIN defaults
    const DEFAULT_VPIN_FACTOR: f64 = 2.0;
    const DEFAULT_VPIN_BUCKET_SIZE: f64 = 50_000.0;
    const DEFAULT_VPIN_N_BUCKETS: usize = 20;

    // Asymmetric sizing defaults
    const DEFAULT_SIZE_DECAY_POWER: f64 = 2.0;
    const DEFAULT_REDUCE_BOOST: f64 = 0.5;

    pub fn from_params(params: &StrategyParams) -> Self {
        let vol_ema_span = params
            .get_i64("vol_ema_span")
            .map(|v| v.max(2) as usize)
            .unwrap_or(Self::DEFAULT_VOL_EMA_SPAN);
        let trade_flow_span = params
            .get_i64("trade_flow_span")
            .map(|v| v.max(2) as usize)
            .unwrap_or(Self::DEFAULT_TRADE_FLOW_SPAN);
        let vpin_bucket_size = params
            .get_f64("vpin_bucket_size")
            .unwrap_or(Self::DEFAULT_VPIN_BUCKET_SIZE);
        let vpin_n_buckets = params
            .get_i64("vpin_n_buckets")
            .map(|v| v.max(1) as usize)
            .unwrap_or(Self::DEFAULT_VPIN_N_BUCKETS);

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
            risk_aversion: params
                .get_f64("risk_aversion")
                .unwrap_or(Self::DEFAULT_RISK_AVERSION),
            fill_intensity: params
                .get_f64("fill_intensity")
                .unwrap_or(Self::DEFAULT_FILL_INTENSITY),
            time_horizon: params
                .get_f64("time_horizon")
                .unwrap_or(Self::DEFAULT_TIME_HORIZON),
            vpin_factor: params
                .get_f64("vpin_factor")
                .unwrap_or(Self::DEFAULT_VPIN_FACTOR),
            size_decay_power: params
                .get_f64("size_decay_power")
                .unwrap_or(Self::DEFAULT_SIZE_DECAY_POWER),
            reduce_boost: params
                .get_f64("reduce_boost")
                .unwrap_or(Self::DEFAULT_REDUCE_BOOST),
            vol_tracker: VolatilityTracker::new(vol_ema_span),
            trade_flow: TradeFlowSignal::new(trade_flow_span),
            fair_value_ema: Ema::new(20),
            vpin_tracker: VpinTracker::new(vpin_bucket_size, vpin_n_buckets),
            last_quoted_mid: None,
            needs_requote: false,
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

        let exchange = book.exchange();
        let symbol = book.symbol().clone();

        // 3. Cancel all existing orders.
        // Note: we use cancel_all because submit_order() doesn't return OrderIds,
        // so we can't track individual orders for targeted cancellation.
        ctx.cancel_all(Some(exchange), Some(symbol.clone()));

        if book.best_bid().is_none() || book.best_ask().is_none() {
            return;
        }
        let net_pos = ctx.net_position(&exchange, &symbol);

        // 4. Fair value adjustments (trade flow + book imbalance).
        let vol_bps = self.vol_tracker.volatility_bps();
        let min_spread_floor = mid * self.min_spread_bps / 10_000.0 / 2.0;

        // Use a preliminary half-spread for signal scaling.
        let preliminary_half = mid * self.base_spread_bps / 10_000.0 / 2.0;
        let trade_flow_adj =
            self.trade_flow.imbalance() * self.trade_flow_factor * preliminary_half;
        let book_imb_adj =
            self.book_imbalance(book) * self.book_imbalance_factor * preliminary_half;
        let adjusted_mid = fair_value + trade_flow_adj + book_imb_adj;

        // 5. Avellaneda-Stoikov pricing.
        let half_spread;
        let reservation_price;

        if vol_bps < 0.01 {
            // Fallback: vol not yet estimated, use heuristic spread.
            half_spread = (mid * self.base_spread_bps / 10_000.0 / 2.0).max(min_spread_floor);
            reservation_price = adjusted_mid;
        } else {
            let gamma = self.risk_aversion;
            let kappa = self.fill_intensity;
            let tau = self.time_horizon;

            // Work in bps domain — makes γ and τ price-level independent.
            // σ_bps is tick-to-tick volatility in bps; squaring stays in bps².
            let sigma_bps = vol_bps;
            let sigma_bps_sq = sigma_bps * sigma_bps;

            // A-S optimal spread in bps:
            //   spread_bps = γ·σ²·τ + (2/γ)·ln(1 + γ/κ)
            // With σ_bps=2, γ=0.3, τ=1: spread = 0.3*4*1 + 6.67*0.18 ≈ 2.4 bps
            let spread_bps =
                gamma * sigma_bps_sq * tau + (2.0 / gamma) * (1.0 + gamma / kappa).ln();

            // Convert bps → dollars.
            let as_spread = mid * spread_bps / 10_000.0;

            // VPIN multiplier: widen spread during toxic flow.
            let vpin_multiplier = 1.0 + self.vpin_factor * self.vpin_tracker.vpin();

            // Floor at min_spread, then apply VPIN multiplier.
            half_spread = (as_spread / 2.0).max(min_spread_floor) * vpin_multiplier;

            // A-S reservation price with NORMALIZED inventory.
            // q_norm ∈ [-1, 1]: position as fraction of max_position.
            let q_norm = if self.max_position > 1e-12 {
                (net_pos / self.max_position).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            // Shift also computed in bps domain, then converted to dollars.
            let shift_bps = q_norm * gamma * sigma_bps_sq * tau;
            let raw_shift = mid * shift_bps / 10_000.0;

            // Clamp shift so the reducing-side quote never crosses mid.
            // At maximum shift, reducing-side sits at mid + 5% of half_spread.
            let max_shift = half_spread * 0.95;
            let shift = raw_shift.clamp(-max_shift, max_shift);
            reservation_price = adjusted_mid - shift;
        }

        // 6. Asymmetric sizing + hard position cap.
        let r = if self.max_position > 1e-12 {
            (net_pos.abs() / self.max_position).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let accum_size = self.order_size * (1.0 - r).powf(self.size_decay_power);
        let reduce_size = self.order_size * (1.0 + r * self.reduce_boost);

        let (mut bid_size, mut ask_size) = if net_pos > 1e-12 {
            // Long: buying accumulates, selling reduces
            (accum_size, reduce_size)
        } else if net_pos < -1e-12 {
            // Short: selling accumulates, buying reduces
            (reduce_size, accum_size)
        } else {
            (self.order_size, self.order_size)
        };

        // Hard position limit: never quote accumulating side beyond max_position.
        // This is unconditional — works even when size_decay_power=0.
        if net_pos >= self.max_position {
            bid_size = 0.0; // at/beyond max long, don't buy more
        }
        if net_pos <= -self.max_position {
            ask_size = 0.0; // at/beyond max short, don't sell more
        }

        // 7. Submit quotes around reservation_price.
        for level in 0..self.num_levels {
            let level_offset = half_spread * level as f64;

            if bid_size > 1e-8 {
                let bid = reservation_price - half_spread - level_offset;
                ctx.submit_order(
                    exchange,
                    symbol.clone(),
                    Side::Buy,
                    OrderType::PostOnly,
                    Price::from(bid),
                    Quantity::from(bid_size),
                );
            }

            if ask_size > 1e-8 {
                let ask = reservation_price + half_spread + level_offset;
                ctx.submit_order(
                    exchange,
                    symbol.clone(),
                    Side::Sell,
                    OrderType::PostOnly,
                    Price::from(ask),
                    Quantity::from(ask_size),
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
        self.vpin_tracker.update(is_buy, notional);
    }

    fn on_fill(&mut self, _ctx: &mut TradingContext, fill: &Fill) {
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
        if let Some(v) = params.get_f64("risk_aversion") {
            self.risk_aversion = v;
        }
        if let Some(v) = params.get_f64("fill_intensity") {
            self.fill_intensity = v;
        }
        if let Some(v) = params.get_f64("time_horizon") {
            self.time_horizon = v;
        }
        if let Some(v) = params.get_f64("vpin_factor") {
            self.vpin_factor = v;
        }
        if let Some(v) = params.get_f64("size_decay_power") {
            self.size_decay_power = v;
        }
        if let Some(v) = params.get_f64("reduce_boost") {
            self.reduce_boost = v;
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
        assert!((strat.max_position - 0.1).abs() < 1e-10);
        assert!((strat.min_spread_bps - 2.0).abs() < 1e-10);
        assert!((strat.risk_aversion - 0.3).abs() < 1e-10);
        assert!((strat.fill_intensity - 1.5).abs() < 1e-10);
        assert!((strat.time_horizon - 1.0).abs() < 1e-10);
        assert!((strat.vpin_factor - 2.0).abs() < 1e-10);
        assert!((strat.size_decay_power - 2.0).abs() < 1e-10);
        assert!((strat.reduce_boost - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_generates_orders() {
        let mut strat = AdaptiveMarketMaker::from_params(&default_params());
        let book = make_book(50000.0, 50001.0);
        let mut ctx = make_context();
        strat.on_book_update(&mut ctx, &book);
        // 1 cancel_all + 1 bid + 1 ask = 3
        assert_eq!(ctx.action_count(), 3);
    }

    #[test]
    fn test_asymmetric_sizing_long() {
        let params = StrategyParams {
            params: serde_json::json!({
                "max_position": 0.01,
                "size_decay_power": 2.0,
                "reduce_boost": 0.5,
            }),
        };
        let mut strat = AdaptiveMarketMaker::from_params(&params);
        let book = make_book(50000.0, 50001.0);

        // At max long position — bid should be tiny, ask should be boosted
        let mut ctx = make_context_with_position(0.01);
        strat.on_book_update(&mut ctx, &book);
        let actions = ctx.drain_actions();
        let submits: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, crate::context::OrderAction::Submit { .. }))
            .collect();

        // With r=1.0: accum_size = 0.001 * (1-1)^2 = 0 → bid skipped (hard cap also blocks)
        // reduce_size = 0.001 * (1 + 1*0.5) = 0.0015 → ask submitted
        assert_eq!(submits.len(), 1, "at max long, only ask should be quoted");
        if let crate::context::OrderAction::Submit { side, quantity, .. } = submits[0] {
            assert_eq!(*side, Side::Sell);
            assert!(quantity.to_f64() > 0.001, "reducing side should be boosted");
        } else {
            panic!("expected Submit");
        }
    }

    #[test]
    fn test_asymmetric_sizing_short() {
        let params = StrategyParams {
            params: serde_json::json!({
                "max_position": 0.01,
                "size_decay_power": 2.0,
                "reduce_boost": 0.5,
            }),
        };
        let mut strat = AdaptiveMarketMaker::from_params(&params);
        let book = make_book(50000.0, 50001.0);

        // At max short position — ask should be tiny, bid should be boosted
        let mut ctx = make_context_with_position(-0.01);
        strat.on_book_update(&mut ctx, &book);
        let actions = ctx.drain_actions();
        let submits: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, crate::context::OrderAction::Submit { .. }))
            .collect();

        // With r=1.0: accum_size = 0 → ask skipped (hard cap also blocks), reduce_size = 0.0015 → bid submitted
        assert_eq!(submits.len(), 1, "at max short, only bid should be quoted");
        if let crate::context::OrderAction::Submit { side, quantity, .. } = submits[0] {
            assert_eq!(*side, Side::Buy);
            assert!(quantity.to_f64() > 0.001, "reducing side should be boosted");
        } else {
            panic!("expected Submit");
        }
    }

    #[test]
    fn test_asymmetric_sizing_partial() {
        let params = StrategyParams {
            params: serde_json::json!({
                "max_position": 0.02,
                "size_decay_power": 2.0,
                "reduce_boost": 0.5,
            }),
        };
        let mut strat = AdaptiveMarketMaker::from_params(&params);
        let book = make_book(50000.0, 50001.0);

        // Half long: r=0.5, accum = 0.001*(0.5)^2 = 0.00025, reduce = 0.001*1.25 = 0.00125
        let mut ctx = make_context_with_position(0.01);
        strat.on_book_update(&mut ctx, &book);
        let actions = ctx.drain_actions();
        let submits: Vec<_> = actions
            .into_iter()
            .filter(|a| matches!(a, crate::context::OrderAction::Submit { .. }))
            .collect();

        // Both sides should be quoted (accum_size > 1e-8)
        assert_eq!(
            submits.len(),
            2,
            "partial inventory should quote both sides"
        );

        let bid_qty = match &submits[0] {
            crate::context::OrderAction::Submit { quantity, .. } => quantity.to_f64(),
            _ => panic!("expected Submit"),
        };
        let ask_qty = match &submits[1] {
            crate::context::OrderAction::Submit { quantity, .. } => quantity.to_f64(),
            _ => panic!("expected Submit"),
        };

        // When long: bid = accum (smaller), ask = reduce (larger)
        assert!(
            bid_qty < ask_qty,
            "when long, bid size ({bid_qty}) should be < ask size ({ask_qty})"
        );
    }

    #[test]
    fn test_as_reservation_price_skew() {
        let params = StrategyParams {
            params: serde_json::json!({
                "max_position": 1.0,
                "base_spread_bps": 10.0,
                "risk_aversion": 0.3,
                "fill_intensity": 1.5,
                "time_horizon": 100.0,
            }),
        };
        let mut strat = AdaptiveMarketMaker::from_params(&params);
        let book = make_book(50000.0, 50001.0);

        // Warm up vol tracker with a few ticks so we're not in the fallback path.
        for price in [50000.0, 50000.5, 50001.0, 50000.3, 50000.8] {
            let b = make_book(price, price + 1.0);
            let mut c = make_context();
            strat.on_book_update(&mut c, &b);
        }

        // Flat — get baseline
        strat.last_quoted_mid = None;
        let mut ctx_flat = make_context();
        strat.on_book_update(&mut ctx_flat, &book);
        let flat_actions = ctx_flat.drain_actions();

        strat.last_quoted_mid = None;

        // Long 0.5 — A-S reservation price should be below mid, pushing quotes down
        let mut ctx_long = make_context_with_position(0.5);
        strat.on_book_update(&mut ctx_long, &book);
        let long_actions = ctx_long.drain_actions();

        // Index 0 = CancelAll, 1 = bid, 2 = ask
        let flat_ask = match &flat_actions[2] {
            crate::context::OrderAction::Submit { price, .. } => price.to_f64(),
            _ => panic!("expected Submit"),
        };
        let long_ask = match &long_actions[2] {
            crate::context::OrderAction::Submit { price, .. } => price.to_f64(),
            _ => panic!("expected Submit"),
        };
        // When long, reservation_price < mid, so ask = reservation_price + half → lower
        assert!(
            long_ask < flat_ask,
            "long ask {long_ask} should be < flat ask {flat_ask}"
        );
    }

    #[test]
    fn test_fill_triggers_requote() {
        let mut strat = AdaptiveMarketMaker::from_params(&default_params());
        let book = make_book(50000.0, 50001.0);

        // First update: cancel_all + bid + ask = 3
        let mut ctx1 = make_context();
        strat.on_book_update(&mut ctx1, &book);
        assert_eq!(ctx1.action_count(), 3);

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

    #[test]
    fn test_vpin_widens_spread() {
        // Create two strategies: one with VPIN=0, one with VPIN=100
        let params_no_vpin = StrategyParams {
            params: serde_json::json!({
                "vpin_factor": 0.0,
                "vpin_bucket_size": 100.0,
                "vpin_n_buckets": 2,
            }),
        };
        let params_with_vpin = StrategyParams {
            params: serde_json::json!({
                "vpin_factor": 100.0,
                "vpin_bucket_size": 100.0,
                "vpin_n_buckets": 2,
            }),
        };

        let mut strat_no = AdaptiveMarketMaker::from_params(&params_no_vpin);
        let mut strat_vp = AdaptiveMarketMaker::from_params(&params_with_vpin);

        // Warm up vol tracker
        for price in [50000.0, 50000.5, 50001.0, 50000.3, 50000.8] {
            let b = make_book(price, price + 1.0);
            let mut c = make_context();
            strat_no.on_book_update(&mut c, &b);
            strat_no.last_quoted_mid = None;
            let mut c2 = make_context();
            strat_vp.on_book_update(&mut c2, &b);
            strat_vp.last_quoted_mid = None;
        }

        // Feed trades to build VPIN (all buys = toxic)
        for _ in 0..30 {
            let trade = Trade {
                exchange: Exchange::Binance,
                symbol: Symbol::new("BTCUSDT"),
                timestamp: Timestamp::from_millis(1000),
                price: Price::from(50000.0),
                quantity: Quantity::from(0.01),
                side: Side::Buy,
                trade_id: "1".to_string(),
            };
            let mut c = make_context();
            strat_no.on_trade(&mut c, &trade);
            let mut c2 = make_context();
            strat_vp.on_trade(&mut c2, &trade);
        }

        // Check VPIN state
        let vpin_no = strat_no.vpin_tracker.vpin();
        let vpin_vp = strat_vp.vpin_tracker.vpin();
        eprintln!("VPIN no_vpin strat: {}", vpin_no);
        eprintln!("VPIN with_vpin strat: {}", vpin_vp);

        // Now get quotes from both
        let book = make_book(50000.0, 50001.0);
        strat_no.last_quoted_mid = None;
        strat_vp.last_quoted_mid = None;
        let mut ctx_no = make_context();
        strat_no.on_book_update(&mut ctx_no, &book);
        let mut ctx_vp = make_context();
        strat_vp.on_book_update(&mut ctx_vp, &book);

        let actions_no = ctx_no.drain_actions();
        let actions_vp = ctx_vp.drain_actions();

        // Extract ask prices
        let ask_no = actions_no
            .iter()
            .find_map(|a| match a {
                crate::context::OrderAction::Submit { side, price, .. } if *side == Side::Sell => {
                    Some(price.to_f64())
                }
                _ => None,
            })
            .expect("no ask from no-vpin strat");

        let ask_vp = actions_vp
            .iter()
            .find_map(|a| match a {
                crate::context::OrderAction::Submit { side, price, .. } if *side == Side::Sell => {
                    Some(price.to_f64())
                }
                _ => None,
            })
            .expect("no ask from with-vpin strat");

        eprintln!("Ask no_vpin: {:.4}, ask with_vpin: {:.4}", ask_no, ask_vp);
        assert!(
            ask_vp > ask_no,
            "VPIN should widen spread: vpin ask {} should be > no-vpin ask {}",
            ask_vp,
            ask_no
        );
    }
}

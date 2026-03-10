//! Adaptive market-making strategy with Avellaneda-Stoikov pricing, VPIN-based
//! spread adjustment, asymmetric sizing, and volatility-adaptive quoting.

use std::collections::{HashMap, VecDeque};

use cm_core::types::*;
use cm_market_data::orderbook::OrderBook;

use crate::context::TradingContext;
use crate::traits::{Fill, Strategy, StrategyParams};

use super::signals::{
    Ema, RegimeDetector, TradeFlowSignal, TradeImbalanceTracker, VolatilityTracker, VpinTracker,
};

/// ML integration mode.
///
/// - `Shift`: shift reservation price (directional bet).
/// - `Width`: widen spread symmetrically based on confidence (protective).
/// - `Skew`: widen only the adverse side of the spread (protective, directional).
#[derive(Clone, Copy, PartialEq, Debug)]
enum MlMode {
    /// Shift reservation price by signal × factor × half_spread.
    Shift,
    /// Widen spread symmetrically by |signal| × factor × half_spread.
    Width,
    /// Widen only the adverse side: if signal > 0, widen ask; if < 0, widen bid.
    /// Never narrows either side — only adds protection.
    Skew,
}

impl MlMode {
    fn from_str(s: &str) -> Self {
        match s {
            "width" => Self::Width,
            "skew" => Self::Skew,
            _ => Self::Shift,
        }
    }
}

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

    // ── ML signal ──
    #[cfg(feature = "ml")]
    ml_signal: Option<cm_ml::predictor::MlSignal>,
    ml_factor: f64,
    ml_threshold: f64,
    ml_mode: MlMode,
    /// Per-symbol ML overrides: symbol → (mode, factor, threshold).
    ml_overrides: HashMap<String, (MlMode, f64, f64)>,

    // ── Trade imbalance filter ──
    imbalance_tracker: TradeImbalanceTracker,
    imbalance_threshold: f64, // |imbalance| above which spread adjustment fires
    imbalance_factor: f64,    // spread multiplier on the vulnerable side; 0 = disabled

    // ── Regime detector ──
    regime_detector: RegimeDetector,
    regime_window_secs: u64,
    regime_drift_enter: f64,
    regime_drift_exit: f64,
    regime_max_mult: f64,
    regime_max_combined_mult: f64, // cap on vpin_multiplier * regime_mult

    // ── Recent mid history for return calculation ──
    recent_mids: VecDeque<f64>,

    // ── Position flush ──
    flush_interval_ticks: u64, // 0 = disabled
    flush_threshold: f64,      // min |position| to trigger flush
    flush_tick_counter: u64,   // internal counter, incremented each on_timer call

    // ── Timer state (set from on_book_update) ──
    // exchange/symbol set from on_book_update — assumes single-symbol usage
    last_mid: f64,
    last_exchange: Option<Exchange>,
    last_symbol: Option<Symbol>,

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

    // Flush defaults
    const DEFAULT_FLUSH_INTERVAL_TICKS: u64 = 0;
    const DEFAULT_FLUSH_THRESHOLD: f64 = 0.0;

    // Trade imbalance defaults
    const DEFAULT_IMBALANCE_WINDOW: usize = 0; // 0 = disabled
    const DEFAULT_IMBALANCE_THRESHOLD: f64 = 0.5;
    const DEFAULT_IMBALANCE_FACTOR: f64 = 0.0; // 0 = disabled

    // Regime detector defaults
    const DEFAULT_REGIME_WINDOW_SECS: u64 = 0; // 0 = disabled
    const DEFAULT_REGIME_DRIFT_ENTER_BPS_HR: f64 = 100.0;
    const DEFAULT_REGIME_DRIFT_EXIT_BPS_HR: f64 = 50.0;
    const DEFAULT_REGIME_MAX_MULT: f64 = 3.0;
    const DEFAULT_REGIME_MAX_COMBINED_MULT: f64 = 5.0;

    // ML defaults
    const DEFAULT_ML_FACTOR: f64 = 1.0;
    const DEFAULT_ML_THRESHOLD: f64 = 0.05; // dead zone: ignore |signal| < threshold
    const DEFAULT_ML_MODE: MlMode = MlMode::Width;
    #[cfg(feature = "ml")]
    const DEFAULT_ML_WEIGHTS_PATH: &str = "models/mid_predictor.safetensors";

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
        let imbalance_window = params
            .get_i64("imbalance_window")
            .map(|v| v.max(0) as usize)
            .unwrap_or(Self::DEFAULT_IMBALANCE_WINDOW);
        let regime_window_secs = params
            .get_i64("regime_window_secs")
            .map(|v| v.max(0) as u64)
            .unwrap_or(Self::DEFAULT_REGIME_WINDOW_SECS);
        let regime_drift_enter = params
            .get_f64("regime_drift_enter_bps_hr")
            .unwrap_or(Self::DEFAULT_REGIME_DRIFT_ENTER_BPS_HR);
        let regime_drift_exit = params
            .get_f64("regime_drift_exit_bps_hr")
            .unwrap_or(Self::DEFAULT_REGIME_DRIFT_EXIT_BPS_HR);
        let regime_max_mult = params
            .get_f64("regime_max_mult")
            .unwrap_or(Self::DEFAULT_REGIME_MAX_MULT);

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
            #[cfg(feature = "ml")]
            ml_signal: {
                let weights_path = params
                    .get_str("ml_weights_path")
                    .unwrap_or(Self::DEFAULT_ML_WEIGHTS_PATH);
                match cm_ml::predictor::MlSignal::try_load(std::path::Path::new(weights_path)) {
                    Ok(sig) => sig,
                    Err(e) => {
                        tracing::warn!("failed to load ML model: {e}");
                        None
                    }
                }
            },
            ml_factor: params
                .get_f64("ml_factor")
                .unwrap_or(Self::DEFAULT_ML_FACTOR),
            ml_threshold: params
                .get_f64("ml_threshold")
                .unwrap_or(Self::DEFAULT_ML_THRESHOLD),
            ml_mode: params
                .get_str("ml_mode")
                .map(MlMode::from_str)
                .unwrap_or(Self::DEFAULT_ML_MODE),
            ml_overrides: Self::parse_ml_overrides(params),
            recent_mids: VecDeque::with_capacity(32),
            flush_interval_ticks: params
                .get_i64("flush_interval_ticks")
                .map(|v| v.max(0) as u64)
                .unwrap_or(Self::DEFAULT_FLUSH_INTERVAL_TICKS),
            flush_threshold: params
                .get_f64("flush_threshold")
                .unwrap_or(Self::DEFAULT_FLUSH_THRESHOLD),
            flush_tick_counter: 0,
            imbalance_tracker: TradeImbalanceTracker::new(imbalance_window),
            imbalance_threshold: params
                .get_f64("imbalance_threshold")
                .unwrap_or(Self::DEFAULT_IMBALANCE_THRESHOLD),
            imbalance_factor: params
                .get_f64("imbalance_factor")
                .unwrap_or(Self::DEFAULT_IMBALANCE_FACTOR),
            regime_detector: RegimeDetector::new(
                regime_window_secs,
                regime_drift_enter,
                regime_drift_exit,
                regime_max_mult,
            ),
            regime_window_secs,
            regime_drift_enter,
            regime_drift_exit,
            regime_max_mult,
            regime_max_combined_mult: params
                .get_f64("regime_max_combined_mult")
                .unwrap_or(Self::DEFAULT_REGIME_MAX_COMBINED_MULT),
            last_mid: 0.0,
            last_exchange: None,
            last_symbol: None,
            last_quoted_mid: None,
            needs_requote: false,
        }
    }

    /// Parse `ml_overrides` from params JSON.
    /// Expected format: `{"ETHUSDT": {"ml_mode": "shift", "ml_factor": 0.3, "ml_threshold": 0.05}}`.
    fn parse_ml_overrides(params: &StrategyParams) -> HashMap<String, (MlMode, f64, f64)> {
        let mut overrides = HashMap::new();
        if let Some(obj) = params.get_object("ml_overrides") {
            for (symbol, val) in obj {
                let mode = val
                    .get("ml_mode")
                    .and_then(|v| v.as_str())
                    .map(MlMode::from_str)
                    .unwrap_or(Self::DEFAULT_ML_MODE);
                let factor = val
                    .get("ml_factor")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(Self::DEFAULT_ML_FACTOR);
                let threshold = val
                    .get("ml_threshold")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(Self::DEFAULT_ML_THRESHOLD);
                overrides.insert(symbol.clone(), (mode, factor, threshold));
            }
        }
        overrides
    }

    /// Look up ML parameters for a symbol.
    /// Returns `(mode, factor, threshold)` — per-symbol override if present, else global defaults.
    #[cfg(feature = "ml")]
    fn ml_params_for_symbol(&self, symbol: &str) -> (MlMode, f64, f64) {
        if let Some(&(mode, factor, threshold)) = self.ml_overrides.get(symbol) {
            (mode, factor, threshold)
        } else {
            (self.ml_mode, self.ml_factor, self.ml_threshold)
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
        self.regime_detector.update(mid, ctx.timestamp.as_nanos());
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

        // Store for on_timer use.
        self.last_mid = mid;
        self.last_exchange = Some(exchange);
        self.last_symbol = Some(symbol.clone());

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

        // ML signal adjustment.
        // Returns (ml_adj, ml_bid_extra, ml_ask_extra):
        //   - ml_adj: shift to reservation price (Shift mode)
        //   - ml_bid_extra: additional spread on bid side (Skew/Width modes)
        //   - ml_ask_extra: additional spread on ask side (Skew/Width modes)
        // Compute all feature inputs BEFORE borrowing ml_signal mutably.
        let (ml_adj, ml_bid_extra, ml_ask_extra) = {
            #[cfg(feature = "ml")]
            {
                let book_imb = self.book_imbalance(book);
                let trade_flow_imb = self.trade_flow.imbalance();
                let vpin_val = self.vpin_tracker.vpin();
                let spread_bps = book.spread_bps().unwrap_or(0.0);
                let recent_return_bps = if let Some(&prev) = self.recent_mids.back() {
                    if prev > 0.0 {
                        (mid - prev) / prev * 10_000.0
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                let norm_pos = if self.max_position > 1e-12 {
                    (net_pos / self.max_position).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                let (ml_mode, ml_factor, ml_threshold) = self.ml_params_for_symbol(&symbol.0);

                if let Some(ref mut ml) = self.ml_signal {
                    let features = cm_ml::features::MlFeatures {
                        book_imbalance: book_imb,
                        trade_flow_imbalance: trade_flow_imb,
                        vpin: vpin_val,
                        volatility_bps: vol_bps,
                        spread_bps,
                        recent_return_bps,
                        normalized_position: norm_pos,
                    };
                    let raw_signal = ml.predict(&features);
                    // Dead zone: ignore weak predictions
                    let signal = if raw_signal.abs() < ml_threshold {
                        0.0
                    } else {
                        // Remove the dead zone and rescale
                        let sign = raw_signal.signum();
                        let magnitude = (raw_signal.abs() - ml_threshold) / (0.5 - ml_threshold);
                        sign * magnitude
                    };

                    match ml_mode {
                        MlMode::Shift => {
                            // Current behavior: shift reservation price.
                            (signal * ml_factor * preliminary_half, 0.0, 0.0)
                        }
                        MlMode::Width => {
                            // Use confidence (|signal|) to widen spread symmetrically.
                            // More confident = wider spread = more protection.
                            let extra = signal.abs() * 2.0 * ml_factor * preliminary_half;
                            (0.0, extra, extra)
                        }
                        MlMode::Skew => {
                            // Widen only the adverse side of the spread.
                            // If signal > 0 (predicts up): widen ask (protect selling too cheap).
                            // If signal < 0 (predicts down): widen bid (protect buying too high).
                            // Never narrows either side.
                            let magnitude = signal.abs() * ml_factor * preliminary_half;
                            if signal > 0.0 {
                                (0.0, 0.0, magnitude)
                            } else if signal < 0.0 {
                                (0.0, magnitude, 0.0)
                            } else {
                                (0.0, 0.0, 0.0)
                            }
                        }
                    }
                } else {
                    (0.0, 0.0, 0.0)
                }
            }
            #[cfg(not(feature = "ml"))]
            {
                (0.0, 0.0, 0.0)
            }
        };

        // Track recent mids for return calculation.
        self.recent_mids.push_back(mid);
        if self.recent_mids.len() > 20 {
            self.recent_mids.pop_front();
        }

        let adjusted_mid = fair_value + trade_flow_adj + book_imb_adj + ml_adj;

        // 5. Avellaneda-Stoikov pricing.
        let half_spread;
        let reservation_price;

        if vol_bps < 0.01 {
            // Fallback: vol not yet estimated, use heuristic spread.
            // VPIN is not applied on this path (no vol estimate yet) — only regime multiplier.
            half_spread = (mid * self.base_spread_bps / 10_000.0 / 2.0).max(min_spread_floor)
                * self.regime_detector.spread_multiplier();
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

            // Regime multiplier: widen spread during trending markets.
            let regime_mult = self.regime_detector.spread_multiplier();

            // Floor at min_spread, then apply combined multiplier (capped to avoid excessive width).
            let combined_mult = (vpin_multiplier * regime_mult).min(self.regime_max_combined_mult);
            half_spread = (as_spread / 2.0).max(min_spread_floor) * combined_mult;

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

        // 7. Compute trade imbalance asymmetric spread adjustment.
        // Buy pressure → widen ask (raise ask, adversely selecting buy aggressor).
        // Sell pressure → widen bid (lower bid, adversely selecting sell aggressor).
        let (imb_bid_extra, imb_ask_extra) =
            if self.imbalance_factor > 0.0 && self.imbalance_tracker.window_size() > 0 {
                let imb = self.imbalance_tracker.imbalance();
                if imb > self.imbalance_threshold {
                    (0.0, self.imbalance_factor * imb * half_spread)
                } else if imb < -self.imbalance_threshold {
                    (self.imbalance_factor * imb.abs() * half_spread, 0.0)
                } else {
                    (0.0, 0.0)
                }
            } else {
                (0.0, 0.0)
            };

        // 8. Submit quotes around reservation_price.
        for level in 0..self.num_levels {
            let level_offset = half_spread * level as f64;

            if bid_size > 1e-8 {
                let bid =
                    reservation_price - half_spread - level_offset - ml_bid_extra - imb_bid_extra;
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
                let ask =
                    reservation_price + half_spread + level_offset + ml_ask_extra + imb_ask_extra;
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
        self.imbalance_tracker
            .update(is_buy, notional, trade.timestamp.as_nanos());
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

    fn on_timer(&mut self, ctx: &mut TradingContext, _timestamp: Timestamp) {
        self.flush_tick_counter = self.flush_tick_counter.wrapping_add(1);

        // Disabled when interval=0.
        if self.flush_interval_ticks == 0 {
            return;
        }
        // Not yet time to flush.
        if !self
            .flush_tick_counter
            .is_multiple_of(self.flush_interval_ticks)
        {
            return;
        }

        // Need exchange/symbol/mid from last on_book_update.
        let (exchange, symbol) = match (&self.last_exchange, &self.last_symbol) {
            (Some(e), Some(s)) => (*e, s.clone()),
            _ => return,
        };
        if self.last_mid <= 0.0 {
            return;
        }

        let net_pos = ctx.net_position(&exchange, &symbol);
        if net_pos.abs() <= self.flush_threshold {
            return;
        }

        // Submit market-crossing flush order. Pricing above mid for sell (buy trades at ask
        // sweep through it) and below mid for buy (sell trades at bid sweep through it).
        let (side, flush_price) = if net_pos > 0.0 {
            (Side::Sell, self.last_mid * 1.0001)
        } else {
            (Side::Buy, self.last_mid * 0.9999)
        };

        // Limit (not PostOnly): flush must execute as taker to cross spread
        ctx.submit_order(
            exchange,
            symbol,
            side,
            OrderType::Limit,
            Price::from(flush_price),
            Quantity::from(net_pos.abs()),
        );

        self.needs_requote = true;
    }

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
        if let Some(v) = params.get_f64("ml_factor") {
            self.ml_factor = v;
        }
        if let Some(v) = params.get_f64("ml_threshold") {
            self.ml_threshold = v;
        }
        if let Some(v) = params.get_str("ml_mode") {
            self.ml_mode = MlMode::from_str(v);
        }
        if params.get_object("ml_overrides").is_some() {
            self.ml_overrides = Self::parse_ml_overrides(params);
        }
        if let Some(v) = params.get_i64("flush_interval_ticks") {
            self.flush_interval_ticks = v.max(0) as u64;
        }
        if let Some(v) = params.get_f64("flush_threshold") {
            self.flush_threshold = v;
        }
        if let Some(v) = params.get_i64("imbalance_window") {
            self.imbalance_tracker = TradeImbalanceTracker::new(v.max(0) as usize);
        }
        if let Some(v) = params.get_f64("imbalance_threshold") {
            self.imbalance_threshold = v;
        }
        if let Some(v) = params.get_f64("imbalance_factor") {
            self.imbalance_factor = v;
        }
        // Regime detector: reconstruct on any param change (changing window_secs requires
        // clearing the stale VecDeque — same pattern as imbalance_tracker above).
        // Fall back to current runtime values (self.regime_*), not compile-time defaults,
        // so partial updates preserve non-provided params.
        let regime_changed = params.get_i64("regime_window_secs").is_some()
            || params.get_f64("regime_drift_enter_bps_hr").is_some()
            || params.get_f64("regime_drift_exit_bps_hr").is_some()
            || params.get_f64("regime_max_mult").is_some();
        if regime_changed {
            if let Some(v) = params.get_i64("regime_window_secs") {
                self.regime_window_secs = v.max(0) as u64;
            }
            if let Some(v) = params.get_f64("regime_drift_enter_bps_hr") {
                self.regime_drift_enter = v;
            }
            if let Some(v) = params.get_f64("regime_drift_exit_bps_hr") {
                self.regime_drift_exit = v;
            }
            if let Some(v) = params.get_f64("regime_max_mult") {
                self.regime_max_mult = v;
            }
            self.regime_detector = RegimeDetector::new(
                self.regime_window_secs,
                self.regime_drift_enter,
                self.regime_drift_exit,
                self.regime_max_mult,
            );
        }
        if let Some(v) = params.get_f64("regime_max_combined_mult") {
            self.regime_max_combined_mult = v;
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

    // ── Flush tests ──────────────────────────────────────────────────────────

    fn make_flush_params(interval: i64, threshold: f64) -> StrategyParams {
        StrategyParams {
            params: serde_json::json!({
                "flush_interval_ticks": interval,
                "flush_threshold": threshold,
            }),
        }
    }

    #[test]
    fn test_on_timer_noop_when_disabled() {
        // Default flush_interval_ticks=0 → disabled; no flush order submitted
        let mut strat = AdaptiveMarketMaker::from_params(&default_params());
        let book = make_book(50000.0, 50001.0);
        let mut setup_ctx = make_context();
        strat.on_book_update(&mut setup_ctx, &book);

        let mut ctx = make_context_with_position(0.05);
        strat.on_timer(&mut ctx, Timestamp::from_millis(1000));
        let submits: Vec<_> = ctx
            .drain_actions()
            .into_iter()
            .filter(|a| matches!(a, crate::context::OrderAction::Submit { .. }))
            .collect();
        assert_eq!(submits.len(), 0, "disabled flush should produce no orders");
    }

    #[test]
    fn test_on_timer_submits_sell_when_long() {
        // flush_interval_ticks=1 → flush every tick; long position → sell order
        let mut strat = AdaptiveMarketMaker::from_params(&make_flush_params(1, 0.0));
        let book = make_book(50000.0, 50001.0);
        let mut setup_ctx = make_context();
        strat.on_book_update(&mut setup_ctx, &book);

        let mut ctx = make_context_with_position(0.05); // long
        strat.on_timer(&mut ctx, Timestamp::from_millis(1000));
        let actions = ctx.drain_actions();
        let submits: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, crate::context::OrderAction::Submit { .. }))
            .collect();
        assert_eq!(
            submits.len(),
            1,
            "long position should produce one flush order"
        );
        if let crate::context::OrderAction::Submit { side, .. } = submits[0] {
            assert_eq!(*side, Side::Sell, "flush order for long should be a sell");
        }
    }

    #[test]
    fn test_on_timer_submits_buy_when_short() {
        // flush_interval_ticks=1 → flush every tick; short position → buy order
        let mut strat = AdaptiveMarketMaker::from_params(&make_flush_params(1, 0.0));
        let book = make_book(50000.0, 50001.0);
        let mut setup_ctx = make_context();
        strat.on_book_update(&mut setup_ctx, &book);

        let mut ctx = make_context_with_position(-0.05); // short
        strat.on_timer(&mut ctx, Timestamp::from_millis(1000));
        let actions = ctx.drain_actions();
        let submits: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, crate::context::OrderAction::Submit { .. }))
            .collect();
        assert_eq!(
            submits.len(),
            1,
            "short position should produce one flush order"
        );
        if let crate::context::OrderAction::Submit { side, .. } = submits[0] {
            assert_eq!(*side, Side::Buy, "flush order for short should be a buy");
        }
    }

    #[test]
    fn test_on_timer_flush_sell_price_above_mid() {
        // Sell flush must be priced above mid so buy trades sweep through it in queue model
        let mut strat = AdaptiveMarketMaker::from_params(&make_flush_params(1, 0.0));
        let book = make_book(50000.0, 50001.0); // mid = 50000.5
        let mut setup_ctx = make_context();
        strat.on_book_update(&mut setup_ctx, &book);

        let mut ctx = make_context_with_position(0.05);
        strat.on_timer(&mut ctx, Timestamp::from_millis(1000));
        let sell_price = ctx.drain_actions().into_iter().find_map(|a| match a {
            crate::context::OrderAction::Submit {
                side: Side::Sell,
                price,
                ..
            } => Some(price.to_f64()),
            _ => None,
        });
        let mid = 50000.5;
        let price = sell_price.expect("no sell order found");
        assert!(
            price > mid,
            "sell flush price {price} should be above mid {mid}"
        );
    }

    #[test]
    fn test_on_timer_flush_buy_price_below_mid() {
        // Buy flush must be priced below mid so sell trades sweep through it in queue model
        let mut strat = AdaptiveMarketMaker::from_params(&make_flush_params(1, 0.0));
        let book = make_book(50000.0, 50001.0); // mid = 50000.5
        let mut setup_ctx = make_context();
        strat.on_book_update(&mut setup_ctx, &book);

        let mut ctx = make_context_with_position(-0.05);
        strat.on_timer(&mut ctx, Timestamp::from_millis(1000));
        let buy_price = ctx.drain_actions().into_iter().find_map(|a| match a {
            crate::context::OrderAction::Submit {
                side: Side::Buy,
                price,
                ..
            } => Some(price.to_f64()),
            _ => None,
        });
        let mid = 50000.5;
        let price = buy_price.expect("no buy order found");
        assert!(
            price < mid,
            "buy flush price {price} should be below mid {mid}"
        );
    }

    #[test]
    fn test_on_timer_respects_interval() {
        // flush_interval_ticks=3 → flush only on ticks 3, 6, 9...
        let mut strat = AdaptiveMarketMaker::from_params(&make_flush_params(3, 0.0));
        let book = make_book(50000.0, 50001.0);
        let mut setup_ctx = make_context();
        strat.on_book_update(&mut setup_ctx, &book);

        // Tick 1: no flush
        let mut ctx = make_context_with_position(0.05);
        strat.on_timer(&mut ctx, Timestamp::from_millis(1));
        assert_eq!(
            ctx.drain_actions()
                .iter()
                .filter(|a| matches!(a, crate::context::OrderAction::Submit { .. }))
                .count(),
            0,
            "tick 1 should not flush"
        );

        // Tick 2: no flush
        let mut ctx = make_context_with_position(0.05);
        strat.on_timer(&mut ctx, Timestamp::from_millis(2));
        assert_eq!(
            ctx.drain_actions()
                .iter()
                .filter(|a| matches!(a, crate::context::OrderAction::Submit { .. }))
                .count(),
            0,
            "tick 2 should not flush"
        );

        // Tick 3: flush fires
        let mut ctx = make_context_with_position(0.05);
        strat.on_timer(&mut ctx, Timestamp::from_millis(3));
        assert_eq!(
            ctx.drain_actions()
                .iter()
                .filter(|a| matches!(a, crate::context::OrderAction::Submit { .. }))
                .count(),
            1,
            "tick 3 should flush"
        );
    }

    #[test]
    fn test_on_timer_respects_threshold() {
        // flush_threshold=0.1 → position=0.05 < threshold → no flush
        let mut strat = AdaptiveMarketMaker::from_params(&make_flush_params(1, 0.1));
        let book = make_book(50000.0, 50001.0);
        let mut setup_ctx = make_context();
        strat.on_book_update(&mut setup_ctx, &book);

        let mut ctx = make_context_with_position(0.05); // below threshold
        strat.on_timer(&mut ctx, Timestamp::from_millis(1000));
        let submits: Vec<_> = ctx
            .drain_actions()
            .into_iter()
            .filter(|a| matches!(a, crate::context::OrderAction::Submit { .. }))
            .collect();
        assert_eq!(
            submits.len(),
            0,
            "position below threshold should not trigger flush"
        );
    }

    #[test]
    fn test_on_timer_negative_interval_disabled() {
        // Negative flush_interval_ticks param → clamps to 0 → disabled
        let params = StrategyParams {
            params: serde_json::json!({ "flush_interval_ticks": -1 }),
        };
        let mut strat = AdaptiveMarketMaker::from_params(&params);
        let book = make_book(50000.0, 50001.0);
        let mut setup_ctx = make_context();
        strat.on_book_update(&mut setup_ctx, &book);

        let mut ctx = make_context_with_position(0.05);
        strat.on_timer(&mut ctx, Timestamp::from_millis(1000));
        let submits: Vec<_> = ctx
            .drain_actions()
            .into_iter()
            .filter(|a| matches!(a, crate::context::OrderAction::Submit { .. }))
            .collect();
        assert_eq!(
            submits.len(),
            0,
            "negative flush_interval_ticks should be treated as disabled (0)"
        );
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

    /// Integration test: with imbalance_factor=1.0 and an all-buy window, the ask
    /// side should be wider than the bid side (asymmetric spread).
    /// Trade flow and book imbalance factors are set to 0 to isolate the imbalance effect.
    #[test]
    fn test_imbalance_factor_widens_ask_on_buy_pressure() {
        let params = StrategyParams {
            params: serde_json::json!({
                "imbalance_window": 50,
                "imbalance_threshold": 0.5,
                "imbalance_factor": 1.0,
                "trade_flow_factor": 0.0,      // isolate imbalance effect
                "book_imbalance_factor": 0.0,  // isolate imbalance effect
                "reprice_threshold_bps": 0.0, // always requote
                "base_spread_bps": 10.0,
                "risk_aversion": 0.3,
                "fill_intensity": 1.5,
                "time_horizon": 1.0,
            }),
        };

        let mut strat = AdaptiveMarketMaker::from_params(&params);
        let book = make_book(2000.0, 2001.0);
        let mid = 2000.5;

        // Flood the imbalance window with buy trades
        for i in 0..50_u64 {
            let trade = Trade {
                exchange: Exchange::Binance,
                symbol: Symbol::new("BTCUSDT"),
                price: Price::from(mid),
                quantity: Quantity::from(0.1),
                side: Side::Buy,
                timestamp: Timestamp::from_millis(i + 1),
                trade_id: i.to_string(),
            };
            strat.on_trade(&mut make_context(), &trade);
        }

        // Get quotes after all-buy imbalance
        let mut ctx = make_context();
        strat.on_book_update(&mut ctx, &book);
        let actions = ctx.drain_actions();

        let bid_price = actions
            .iter()
            .find_map(|a| match a {
                crate::context::OrderAction::Submit { side, price, .. } if *side == Side::Buy => {
                    Some(price.to_f64())
                }
                _ => None,
            })
            .expect("no bid submitted");
        let ask_price = actions
            .iter()
            .find_map(|a| match a {
                crate::context::OrderAction::Submit { side, price, .. } if *side == Side::Sell => {
                    Some(price.to_f64())
                }
                _ => None,
            })
            .expect("no ask submitted");

        let bid_dist = mid - bid_price;
        let ask_dist = ask_price - mid;
        eprintln!("mid={mid:.4}, bid={bid_price:.4} (dist={bid_dist:.4}), ask={ask_price:.4} (dist={ask_dist:.4})");
        assert!(
            ask_dist > bid_dist,
            "buy pressure should widen ask: ask_dist={:.4} should be > bid_dist={:.4}",
            ask_dist,
            bid_dist
        );
    }

    /// Integration test: with imbalance_factor=0.0, the spread is symmetric regardless
    /// of window contents (backward compatibility — disabled by default).
    /// Trade flow and book imbalance factors are set to 0 to isolate the imbalance effect.
    #[test]
    fn test_imbalance_factor_zero_produces_symmetric_spread() {
        let params = StrategyParams {
            params: serde_json::json!({
                "imbalance_window": 50,
                "imbalance_threshold": 0.5,
                "imbalance_factor": 0.0, // disabled
                "trade_flow_factor": 0.0,      // isolate imbalance effect
                "book_imbalance_factor": 0.0,  // isolate imbalance effect
                "reprice_threshold_bps": 0.0,
                "base_spread_bps": 10.0,
                "risk_aversion": 0.3,
                "fill_intensity": 1.5,
                "time_horizon": 1.0,
            }),
        };

        let mut strat = AdaptiveMarketMaker::from_params(&params);
        let book = make_book(2000.0, 2001.0);
        let mid = 2000.5;

        // Flood with buy trades — imbalance_factor=0 should ignore this
        for i in 0..50_u64 {
            let trade = Trade {
                exchange: Exchange::Binance,
                symbol: Symbol::new("BTCUSDT"),
                price: Price::from(mid),
                quantity: Quantity::from(0.1),
                side: Side::Buy,
                timestamp: Timestamp::from_millis(i + 1),
                trade_id: i.to_string(),
            };
            strat.on_trade(&mut make_context(), &trade);
        }

        let mut ctx = make_context();
        strat.on_book_update(&mut ctx, &book);
        let actions = ctx.drain_actions();

        let bid_price = actions
            .iter()
            .find_map(|a| match a {
                crate::context::OrderAction::Submit { side, price, .. } if *side == Side::Buy => {
                    Some(price.to_f64())
                }
                _ => None,
            })
            .expect("no bid submitted");
        let ask_price = actions
            .iter()
            .find_map(|a| match a {
                crate::context::OrderAction::Submit { side, price, .. } if *side == Side::Sell => {
                    Some(price.to_f64())
                }
                _ => None,
            })
            .expect("no ask submitted");

        let bid_dist = mid - bid_price;
        let ask_dist = ask_price - mid;
        eprintln!("factor=0 mid={mid:.4}, bid_dist={bid_dist:.4}, ask_dist={ask_dist:.4}");
        assert!(
            (ask_dist - bid_dist).abs() < 1e-8,
            "imbalance_factor=0 should produce symmetric spread: ask_dist={:.4}, bid_dist={:.4}",
            ask_dist,
            bid_dist
        );
    }

    // ── Regime detector integration tests ────────────────────────────────────

    fn extract_ask(actions: &[crate::context::OrderAction]) -> Option<f64> {
        actions.iter().find_map(|a| match a {
            crate::context::OrderAction::Submit { side, price, .. } if *side == Side::Sell => {
                Some(price.to_f64())
            }
            _ => None,
        })
    }

    fn extract_bid(actions: &[crate::context::OrderAction]) -> Option<f64> {
        actions.iter().find_map(|a| match a {
            crate::context::OrderAction::Submit { side, price, .. } if *side == Side::Buy => {
                Some(price.to_f64())
            }
            _ => None,
        })
    }

    /// regime_window_secs=0 (default): behavior identical to pre-regime code.
    #[test]
    fn test_regime_default_disabled_backward_compat() {
        let mut strat_default = AdaptiveMarketMaker::from_params(&default_params());
        let mut strat_explicit = AdaptiveMarketMaker::from_params(&StrategyParams {
            params: serde_json::json!({ "regime_window_secs": 0 }),
        });

        let book = make_book(50000.0, 50001.0);
        // Warm up both identically
        for price in [50000.0, 50001.0, 50000.5, 50001.5, 50000.8] {
            let b = make_book(price, price + 1.0);
            let mut c1 = make_context();
            strat_default.on_book_update(&mut c1, &b);
            strat_default.last_quoted_mid = None;
            let mut c2 = make_context();
            strat_explicit.on_book_update(&mut c2, &b);
            strat_explicit.last_quoted_mid = None;
        }

        let mut ctx1 = make_context();
        strat_default.on_book_update(&mut ctx1, &book);
        let mut ctx2 = make_context();
        strat_explicit.on_book_update(&mut ctx2, &book);

        let ask1 = extract_ask(&ctx1.drain_actions()).expect("no ask from default strat");
        let ask2 = extract_ask(&ctx2.drain_actions()).expect("no ask from explicit strat");
        assert!(
            (ask1 - ask2).abs() < 1e-8,
            "regime disabled should be identical to default: {ask1} vs {ask2}"
        );
    }

    /// With regime enabled and trending prices, spread should widen vs disabled.
    /// Tests the same trending price series fed to both strategies simultaneously,
    /// comparing quoted spreads at the final price level.
    #[test]
    fn test_regime_widens_spread_during_trend() {
        let common = serde_json::json!({
            "regime_drift_enter_bps_hr": 100.0,
            "regime_drift_exit_bps_hr": 50.0,
            "regime_max_mult": 3.0,
            "reprice_threshold_bps": 0.0,
            "trade_flow_factor": 0.0,
            "book_imbalance_factor": 0.0,
        });
        let mut params_no = common.clone();
        params_no["regime_window_secs"] = serde_json::json!(0_i64);
        let mut params_regime = common;
        params_regime["regime_window_secs"] = serde_json::json!(60_i64);

        let mut strat_no = AdaptiveMarketMaker::from_params(&StrategyParams { params: params_no });
        let mut strat_regime = AdaptiveMarketMaker::from_params(&StrategyParams {
            params: params_regime,
        });
        let ns_per_sec = 1_000_000_000u64;

        // Feed the SAME trending ticks to both: 50000 → 50010 over 60s = 120 bps/hr
        for i in 0..=10u64 {
            let t_ns = i * 6 * ns_per_sec;
            let price = 50000.0 + i as f64;
            let b = make_book(price - 0.5, price + 0.5);
            strat_no.last_quoted_mid = None;
            strat_regime.last_quoted_mid = None;
            let mut c1 = TradingContext::new(vec![], vec![], Timestamp(t_ns));
            strat_no.on_book_update(&mut c1, &b);
            let mut c2 = TradingContext::new(vec![], vec![], Timestamp(t_ns));
            strat_regime.on_book_update(&mut c2, &b);
        }

        assert!(
            strat_regime.regime_detector.is_trending(),
            "drift={:.1} bps/hr should trigger trending",
            strat_regime.regime_detector.drift_bps_hr()
        );

        // Compare spreads (bid-ask distance) at the same final price
        let trend_price = 50010.0;
        strat_no.last_quoted_mid = None;
        strat_regime.last_quoted_mid = None;
        let final_book = make_book(trend_price - 0.5, trend_price + 0.5);
        let ts = Timestamp(61 * ns_per_sec);

        let mut ctx_no = TradingContext::new(vec![], vec![], ts);
        strat_no.on_book_update(&mut ctx_no, &final_book);
        let mut ctx_regime = TradingContext::new(vec![], vec![], ts);
        strat_regime.on_book_update(&mut ctx_regime, &final_book);

        let actions_no = ctx_no.drain_actions();
        let actions_regime = ctx_regime.drain_actions();
        let ask_no = extract_ask(&actions_no).expect("no ask from no-regime");
        let bid_no = extract_bid(&actions_no).expect("no bid from no-regime");
        let ask_regime = extract_ask(&actions_regime).expect("no ask from regime");
        let bid_regime = extract_bid(&actions_regime).expect("no bid from regime");

        let spread_no = ask_no - bid_no;
        let spread_regime = ask_regime - bid_regime;
        assert!(
            spread_regime > spread_no * 1.1,
            "trending spread {spread_regime:.4} should be wider than no-regime spread {spread_no:.4}"
        );
    }

    /// With regime enabled and flat prices, spread matches the disabled version.
    #[test]
    fn test_regime_normal_spread_during_flat() {
        let common = serde_json::json!({
            "regime_drift_enter_bps_hr": 100.0,
            "regime_drift_exit_bps_hr": 50.0,
            "regime_max_mult": 3.0,
            "reprice_threshold_bps": 0.0,
            "trade_flow_factor": 0.0,
            "book_imbalance_factor": 0.0,
        });
        let mut params_no = common.clone();
        params_no["regime_window_secs"] = serde_json::json!(0_i64);
        let mut params_regime = common;
        params_regime["regime_window_secs"] = serde_json::json!(60_i64);

        let mut strat_no = AdaptiveMarketMaker::from_params(&StrategyParams { params: params_no });
        let mut strat_regime = AdaptiveMarketMaker::from_params(&StrategyParams {
            params: params_regime,
        });
        let ns_per_sec = 1_000_000_000u64;

        // Feed identical flat prices to both strategies for 60 seconds
        for i in 0..=10u64 {
            let t_ns = i * 6 * ns_per_sec;
            let b = make_book(50000.0, 50001.0);
            strat_no.last_quoted_mid = None;
            strat_regime.last_quoted_mid = None;
            let mut c1 = TradingContext::new(vec![], vec![], Timestamp(t_ns));
            strat_no.on_book_update(&mut c1, &b);
            let mut c2 = TradingContext::new(vec![], vec![], Timestamp(t_ns));
            strat_regime.on_book_update(&mut c2, &b);
        }

        assert!(
            !strat_regime.regime_detector.is_trending(),
            "flat prices should not trigger trending"
        );

        // Both should produce identical spreads
        strat_no.last_quoted_mid = None;
        strat_regime.last_quoted_mid = None;
        let ts = Timestamp(61 * ns_per_sec);
        let b = make_book(50000.0, 50001.0);
        let mut ctx1 = TradingContext::new(vec![], vec![], ts);
        strat_no.on_book_update(&mut ctx1, &b);
        let mut ctx2 = TradingContext::new(vec![], vec![], ts);
        strat_regime.on_book_update(&mut ctx2, &b);

        let ask_no = extract_ask(&ctx1.drain_actions()).unwrap();
        let ask_regime = extract_ask(&ctx2.drain_actions()).unwrap();
        assert!(
            (ask_no - ask_regime).abs() < 1e-8,
            "flat: regime should not change spread: no={ask_no:.4}, regime={ask_regime:.4}"
        );
    }

    /// on_params_update with new regime params reconstructs RegimeDetector (no stale entries).
    /// Uses non-default values (drift_enter=200, drift_exit=80) so partial update cannot
    /// accidentally pass by falling back to defaults that equal the original values.
    #[test]
    fn test_regime_params_update_reconstructs_detector() {
        let params = StrategyParams {
            params: serde_json::json!({
                "regime_window_secs": 60,
                "regime_drift_enter_bps_hr": 200.0,  // non-default (default=100)
                "regime_drift_exit_bps_hr": 80.0,    // non-default (default=50)
                "regime_max_mult": 3.0,
            }),
        };
        let mut strat = AdaptiveMarketMaker::from_params(&params);
        let ns_per_sec = 1_000_000_000u64;

        // Feed trending ticks: 50000 → 50030 over 60s.
        // drift = 30/50000 * 10_000 = 6 bps; bps/hr = 6 * 60 = 360 bps/hr > 200 → enters trending.
        for i in 0..=10u64 {
            let t_ns = i * 6 * ns_per_sec;
            let price = 50000.0 + i as f64 * 3.0;
            let mut c = TradingContext::new(vec![], vec![], Timestamp(t_ns));
            strat.last_quoted_mid = None;
            strat.on_book_update(&mut c, &make_book(price - 0.5, price + 0.5));
        }
        assert!(
            strat.regime_detector.is_trending(),
            "should be trending before params update; drift_bps_hr={}",
            strat.regime_detector.drift_bps_hr()
        );

        // Partial update: only change regime_window_secs — drift_enter/drift_exit/max_mult must
        // be preserved from runtime state (200/80/3.0), not reset to compile-time defaults.
        let new_params = StrategyParams {
            params: serde_json::json!({ "regime_window_secs": 900 }),
        };
        strat.on_params_update(&new_params);

        // After reconstruction, detector is fresh (no stale trending state).
        assert!(
            !strat.regime_detector.is_trending(),
            "reconstructed detector should not carry over stale trending state"
        );

        // Verify non-provided params were preserved (not reset to defaults).
        // Feed a price that would be trending at 150 bps/hr (> default 100, < our 200).
        // With the preserved drift_enter=200, the detector stays FLAT.
        // If it incorrectly reset to default drift_enter=100, it would enter trending.
        let base_time_ns = 10 * 6 * ns_per_sec;
        let base_price = 50010.0_f64;
        // Seed the 900s window: feed 2 ticks spanning 900s with 150 bps/hr drift
        // Drift = (end - start)/start * 10_000 = 0.0075 * 10_000 = 75 bps over window
        // 75 bps / (900s / 3600s/hr) = 75 / 0.25 = 300 bps/hr — above both 100 and 200
        // Instead use a mild drift: 25 bps over 900s = 25/(900/3600) = 100 bps/hr exactly
        // (between drift_enter=200 default and our 200 is same — use 150 bps/hr instead)
        // 150 bps/hr * (900/3600 hr) = 37.5 bps total drift → end = start * (1 + 37.5/10_000)
        let end_price = base_price * (1.0 + 150.0 * (900.0 / 3600.0) / 10_000.0);
        // Feed start tick
        let mut c1 = TradingContext::new(vec![], vec![], Timestamp(base_time_ns));
        strat.last_quoted_mid = None;
        strat.on_book_update(&mut c1, &make_book(base_price - 0.5, base_price + 0.5));
        // Feed end tick (900s later)
        let mut c2 =
            TradingContext::new(vec![], vec![], Timestamp(base_time_ns + 900 * ns_per_sec));
        strat.last_quoted_mid = None;
        strat.on_book_update(&mut c2, &make_book(end_price - 0.5, end_price + 0.5));

        // drift = 150 bps/hr. With preserved drift_enter=200 → NOT trending.
        // If drift_enter was reset to default=100 → WOULD be trending. This assertion catches that.
        assert!(
            !strat.regime_detector.is_trending(),
            "drift_enter must be preserved at 200 after partial update; 150 bps/hr should NOT trigger trending"
        );
        assert_eq!(strat.regime_window_secs, 900, "window_secs updated");
        assert!(
            (strat.regime_drift_enter - 200.0).abs() < 1e-10,
            "drift_enter must be preserved at 200, got {}",
            strat.regime_drift_enter
        );
    }

    /// Combined VPIN * regime multiplier is capped at max_combined_mult.
    #[test]
    fn test_regime_combined_multiplier_capped() {
        // High VPIN factor + high regime max_mult — combined should be capped at 5.0
        let params = StrategyParams {
            params: serde_json::json!({
                "vpin_factor": 10.0,         // vpin_multiplier can be up to 11x
                "vpin_bucket_size": 100.0,
                "vpin_n_buckets": 2,
                "regime_window_secs": 60,
                "regime_drift_enter_bps_hr": 100.0,
                "regime_drift_exit_bps_hr": 50.0,
                "regime_max_mult": 5.0,
                "regime_max_combined_mult": 5.0,
                "reprice_threshold_bps": 0.0,
                "trade_flow_factor": 0.0,
                "book_imbalance_factor": 0.0,
            }),
        };
        let mut strat = AdaptiveMarketMaker::from_params(&params);
        let ns_per_sec = 1_000_000_000u64;

        // Warm up vol
        for i in 0..5u64 {
            let b = make_book(50000.0, 50001.0);
            let mut c = TradingContext::new(vec![], vec![], Timestamp(i * 1_000_000));
            strat.on_book_update(&mut c, &b);
        }

        // Build extreme VPIN (all buys = VPIN → 1.0)
        for _ in 0..20 {
            let trade = Trade {
                exchange: Exchange::Binance,
                symbol: Symbol::new("BTCUSDT"),
                timestamp: Timestamp::from_millis(1000),
                price: Price::from(50000.0),
                quantity: Quantity::from(0.1),
                side: Side::Buy,
                trade_id: "1".to_string(),
            };
            strat.on_trade(&mut make_context(), &trade);
        }

        // Feed trending ticks to enter trending regime
        for i in 0..=10u64 {
            let t_ns = i * 6 * ns_per_sec;
            let price = 50000.0 + i as f64;
            strat.last_quoted_mid = None;
            let mut c = TradingContext::new(vec![], vec![], Timestamp(t_ns));
            strat.on_book_update(&mut c, &make_book(price - 0.5, price + 0.5));
        }
        assert!(strat.regime_detector.is_trending(), "should be trending");
        assert!(strat.vpin_tracker.vpin() > 0.5, "VPIN should be high");

        // Without cap: vpin_multiplier ≈ 1 + 10 * 1.0 = 11.0; regime_mult = 5.0; product = 55.0
        // With cap at 5.0, the product should not exceed 5.0
        strat.last_quoted_mid = None;
        let trend_price = 50010.0;
        let mut ctx = TradingContext::new(vec![], vec![], Timestamp(60 * ns_per_sec + 1_000_000));
        strat.on_book_update(&mut ctx, &make_book(trend_price - 0.5, trend_price + 0.5));
        let actions = ctx.drain_actions();
        let ask = extract_ask(&actions).unwrap();
        let bid = extract_bid(&actions).unwrap();
        let half_spread = (ask - bid) / 2.0;
        let min_half_spread = trend_price * 2.0 / 10_000.0 / 2.0; // min_spread_bps=2.0

        // The effective multiplier should be at most max_combined_mult=5.0
        // half_spread = base * combined_mult, so half_spread/base ≤ 5.0
        let effective_mult = half_spread / min_half_spread;
        eprintln!(
            "half_spread={half_spread:.4}, min_half={min_half_spread:.4}, effective_mult={effective_mult:.2}"
        );
        assert!(
            effective_mult <= 5.01, // small tolerance for float rounding
            "combined multiplier should be capped at 5.0, got {effective_mult:.2}"
        );
    }
}

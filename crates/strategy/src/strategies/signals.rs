//! Reusable signal components for adaptive strategies.
//!
//! Provides EMA, volatility tracking, and trade-flow imbalance signals.

use std::collections::VecDeque;

/// Exponential moving average.
#[derive(Debug, Clone)]
pub struct Ema {
    pub value: f64,
    alpha: f64,
    initialized: bool,
}

impl Ema {
    /// Create an EMA with the given span (number of periods).
    /// `alpha = 2 / (span + 1)`.
    pub fn new(span: usize) -> Self {
        assert!(span >= 1, "EMA span must be >= 1");
        Self {
            value: 0.0,
            alpha: 2.0 / (span as f64 + 1.0),
            initialized: false,
        }
    }

    /// Feed a new sample. Returns the updated EMA value.
    pub fn update(&mut self, x: f64) -> f64 {
        if !self.initialized {
            self.value = x;
            self.initialized = true;
        } else {
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value;
        }
        self.value
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

/// Tracks realised volatility as an EMA of absolute returns (in bps).
#[derive(Debug, Clone)]
pub struct VolatilityTracker {
    ema: Ema,
    last_price: Option<f64>,
}

impl VolatilityTracker {
    pub fn new(span: usize) -> Self {
        Self {
            ema: Ema::new(span),
            last_price: None,
        }
    }

    /// Feed a new mid price. Returns the updated volatility estimate in bps.
    pub fn update(&mut self, mid: f64) -> f64 {
        if let Some(last) = self.last_price {
            if last > 0.0 {
                let abs_return = ((mid - last) / last).abs();
                self.ema.update(abs_return);
            }
        }
        self.last_price = Some(mid);
        self.volatility_bps()
    }

    /// Current volatility estimate in basis points.
    pub fn volatility_bps(&self) -> f64 {
        if !self.ema.is_initialized() {
            return 0.0;
        }
        self.ema.value * 10_000.0
    }
}

/// Tracks trade-flow imbalance as the difference between buy and sell pressure EMAs.
#[derive(Debug, Clone)]
pub struct TradeFlowSignal {
    buy_pressure: Ema,
    sell_pressure: Ema,
}

impl TradeFlowSignal {
    pub fn new(span: usize) -> Self {
        Self {
            buy_pressure: Ema::new(span),
            sell_pressure: Ema::new(span),
        }
    }

    /// Feed a trade event. `is_buy` indicates aggressor side, `notional` is price * qty.
    pub fn update(&mut self, is_buy: bool, notional: f64) {
        if is_buy {
            self.buy_pressure.update(notional);
            self.sell_pressure.update(0.0);
        } else {
            self.buy_pressure.update(0.0);
            self.sell_pressure.update(notional);
        }
    }

    /// Normalised imbalance in range `[-1, +1]`.
    /// Positive = buy-heavy, negative = sell-heavy.
    pub fn imbalance(&self) -> f64 {
        if !self.buy_pressure.is_initialized() || !self.sell_pressure.is_initialized() {
            return 0.0;
        }
        let total = self.buy_pressure.value + self.sell_pressure.value;
        if total < 1e-12 {
            return 0.0;
        }
        (self.buy_pressure.value - self.sell_pressure.value) / total
    }
}

/// Short-term trade imbalance tracker using a bounded rolling window.
///
/// Tracks buy and sell notional volume over the most recent `max_size` trades.
/// Unlike `TradeFlowSignal` (which uses exponential decay with infinite memory),
/// this tracker uses a hard eviction window to capture directional pressure that
/// is meaningful only on a short time scale.
///
/// The `(is_buy, notional, timestamp_ns)` tuple stores timestamps to support
/// future time-based eviction without struct redesign.
#[derive(Debug, Clone)]
pub struct TradeImbalanceTracker {
    window: VecDeque<(bool, f64, u64)>,
    max_size: usize,
}

impl TradeImbalanceTracker {
    /// Create a tracker with the given lookback window size (number of trades).
    /// Use `max_size = 0` to create a disabled (no-op) tracker.
    pub fn new(max_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(max_size.min(4096)),
            max_size,
        }
    }

    /// Feed a trade event. `timestamp_ns` is stored for future time-based eviction.
    pub fn update(&mut self, is_buy: bool, notional: f64, timestamp_ns: u64) {
        if self.max_size == 0 {
            return;
        }
        self.window.push_back((is_buy, notional, timestamp_ns));
        if self.window.len() > self.max_size {
            self.window.pop_front();
        }
    }

    /// Number of trades currently in the window.
    pub fn window_size(&self) -> usize {
        self.window.len()
    }

    /// Normalised imbalance in range `[-1, +1]`.
    /// Positive = buy pressure, negative = sell pressure, 0 = balanced or empty.
    pub fn imbalance(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        let mut buy_vol = 0.0_f64;
        let mut sell_vol = 0.0_f64;
        for &(is_buy, notional, _) in &self.window {
            if is_buy {
                buy_vol += notional;
            } else {
                sell_vol += notional;
            }
        }
        let total = buy_vol + sell_vol;
        if total < 1e-12 {
            return 0.0;
        }
        (buy_vol - sell_vol) / total
    }

    /// Buy volume fraction in range `[0, 1]`.
    pub fn buy_ratio(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        let mut buy_vol = 0.0_f64;
        let mut total = 0.0_f64;
        for &(is_buy, notional, _) in &self.window {
            total += notional;
            if is_buy {
                buy_vol += notional;
            }
        }
        if total < 1e-12 {
            0.0
        } else {
            buy_vol / total
        }
    }
}

/// Volume-synchronized probability of informed trading (VPIN).
///
/// Divides trade flow into equal-volume buckets and measures the average
/// absolute order imbalance across buckets. High VPIN indicates toxic
/// (informed) flow that a market maker should avoid.
#[derive(Debug, Clone)]
pub struct VpinTracker {
    bucket_size: f64,
    n_buckets: usize,
    current_buy_notional: f64,
    current_sell_notional: f64,
    current_total: f64,
    buckets: VecDeque<f64>,
}

impl VpinTracker {
    pub fn new(bucket_size: f64, n_buckets: usize) -> Self {
        Self {
            bucket_size,
            n_buckets,
            current_buy_notional: 0.0,
            current_sell_notional: 0.0,
            current_total: 0.0,
            buckets: VecDeque::with_capacity(n_buckets),
        }
    }

    /// Feed a trade. Accumulates into volume buckets with overflow carry.
    pub fn update(&mut self, is_buy: bool, notional: f64) {
        let mut remaining = notional;
        while remaining > 0.0 {
            let space = self.bucket_size - self.current_total;
            let fill = remaining.min(space);

            if is_buy {
                self.current_buy_notional += fill;
            } else {
                self.current_sell_notional += fill;
            }
            self.current_total += fill;
            remaining -= fill;

            // Bucket complete
            if self.current_total >= self.bucket_size - 1e-12 {
                let imbalance = (self.current_buy_notional - self.current_sell_notional).abs()
                    / self.bucket_size;
                self.buckets.push_back(imbalance);
                if self.buckets.len() > self.n_buckets {
                    self.buckets.pop_front();
                }
                // Reset for next bucket
                self.current_buy_notional = 0.0;
                self.current_sell_notional = 0.0;
                self.current_total = 0.0;
            }
        }
    }

    /// Current VPIN estimate in [0, 1]. 0 = balanced, 1 = toxic.
    /// Returns 0.0 if no buckets completed yet.
    pub fn vpin(&self) -> f64 {
        if self.buckets.is_empty() {
            return 0.0;
        }
        self.buckets.iter().sum::<f64>() / self.buckets.len() as f64
    }
}

/// Classifies market regime as flat or trending based on mid-price drift rate (bps/hr).
///
/// Uses a time-based rolling window and Schmitt trigger hysteresis to avoid flip-flopping.
/// Outputs a spread multiplier in `[1.0, max_mult]` — 1.0 in flat regime, up to `max_mult`
/// in strong trending regime.
///
/// When `window_secs == 0`, the detector is disabled: `update()` is a no-op and
/// `spread_multiplier()` always returns 1.0.
#[derive(Debug, Clone)]
pub struct RegimeDetector {
    window: VecDeque<(f64, u64)>, // (mid_price, timestamp_ns)
    window_ns: u64,
    drift_enter_bps_hr: f64,
    drift_exit_bps_hr: f64,
    max_mult: f64,
    trending: bool,
}

impl RegimeDetector {
    /// Create a regime detector.
    ///
    /// - `window_secs`: rolling window size; use 0 to disable.
    /// - `drift_enter_bps_hr`: abs drift threshold to enter trending regime (bps/hr).
    /// - `drift_exit_bps_hr`: abs drift threshold to exit trending regime (bps/hr).
    /// - `max_mult`: spread multiplier ceiling in trending regime.
    pub fn new(
        window_secs: u64,
        drift_enter_bps_hr: f64,
        drift_exit_bps_hr: f64,
        max_mult: f64,
    ) -> Self {
        Self {
            window: VecDeque::new(),
            window_ns: window_secs.saturating_mul(1_000_000_000),
            drift_enter_bps_hr,
            drift_exit_bps_hr,
            max_mult,
            trending: false,
        }
    }

    /// Feed a mid-price observation. Evicts stale entries and updates regime state.
    pub fn update(&mut self, mid: f64, timestamp_ns: u64) {
        if self.window_ns == 0 {
            return;
        }
        // Evict entries outside the rolling window.
        while let Some(&(_, front_ts)) = self.window.front() {
            if timestamp_ns.saturating_sub(front_ts) > self.window_ns {
                self.window.pop_front();
            } else {
                break;
            }
        }
        self.window.push_back((mid, timestamp_ns));

        // Schmitt trigger hysteresis.
        let abs_drift = self.drift_bps_hr().abs();
        if self.trending {
            if abs_drift < self.drift_exit_bps_hr {
                self.trending = false;
            }
        } else if abs_drift > self.drift_enter_bps_hr {
            self.trending = true;
        }
    }

    /// Current drift rate in bps/hr. Positive = uptrend, negative = downtrend.
    /// Returns 0.0 when fewer than 2 entries are in the window.
    pub fn drift_bps_hr(&self) -> f64 {
        if self.window.len() < 2 {
            return 0.0;
        }
        let &(oldest_mid, oldest_ts) = self.window.front().unwrap();
        let &(newest_mid, newest_ts) = self.window.back().unwrap();
        let elapsed_ns = newest_ts.saturating_sub(oldest_ts);
        if elapsed_ns == 0 || oldest_mid <= 0.0 {
            return 0.0;
        }
        let drift_bps = (newest_mid - oldest_mid) / oldest_mid * 10_000.0;
        drift_bps * 3_600_000_000_000.0 / elapsed_ns as f64
    }

    /// Whether the detector is currently in the trending regime.
    pub fn is_trending(&self) -> bool {
        self.trending
    }

    /// Spread multiplier in `[1.0, max_mult]`.
    ///
    /// Returns 1.0 when disabled or not trending.
    /// When `drift_enter_bps_hr <= drift_exit_bps_hr`, uses binary mode (returns 1.0 or
    /// `max_mult`) to avoid divide-by-zero.
    pub fn spread_multiplier(&self) -> f64 {
        if self.window_ns == 0 || !self.trending {
            return 1.0;
        }
        // Guard: if thresholds equal or inverted, use binary mode to avoid divide-by-zero.
        if self.drift_enter_bps_hr <= self.drift_exit_bps_hr {
            return self.max_mult;
        }
        let abs_drift = self.drift_bps_hr().abs();
        let ramp = (abs_drift - self.drift_exit_bps_hr)
            / (self.drift_enter_bps_hr - self.drift_exit_bps_hr);
        let mult = 1.0 + (self.max_mult - 1.0) * ramp;
        mult.clamp(1.0, self.max_mult)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_basic() {
        let mut ema = Ema::new(10);
        assert!(!ema.is_initialized());
        let v = ema.update(100.0);
        assert!((v - 100.0).abs() < 1e-10);
        assert!(ema.is_initialized());

        // Second update moves toward new value
        let v2 = ema.update(200.0);
        assert!(v2 > 100.0 && v2 < 200.0);
    }

    #[test]
    fn test_ema_converges() {
        let mut ema = Ema::new(5);
        for _ in 0..1000 {
            ema.update(42.0);
        }
        assert!((ema.value - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_volatility_tracker() {
        let mut vt = VolatilityTracker::new(10);
        // First update — no return yet
        vt.update(100.0);
        assert!((vt.volatility_bps() - 0.0).abs() < 1e-10);

        // Second update — 1% move = 100 bps
        vt.update(101.0);
        assert!(vt.volatility_bps() > 0.0);
    }

    #[test]
    fn test_trade_flow_signal() {
        let mut tfs = TradeFlowSignal::new(10);
        assert!((tfs.imbalance() - 0.0).abs() < 1e-10);

        // All buys → positive imbalance
        for _ in 0..20 {
            tfs.update(true, 1000.0);
        }
        assert!(tfs.imbalance() > 0.5);

        // Feed sells to bring it back
        for _ in 0..100 {
            tfs.update(false, 1000.0);
        }
        assert!(tfs.imbalance() < 0.0);
    }

    #[test]
    fn test_vpin_balanced() {
        let mut vpin = VpinTracker::new(1000.0, 5);
        // Alternate equal buys and sells
        for _ in 0..50 {
            vpin.update(true, 500.0);
            vpin.update(false, 500.0);
        }
        assert!(
            vpin.vpin() < 0.1,
            "balanced flow should give low VPIN: {}",
            vpin.vpin()
        );
    }

    #[test]
    fn test_vpin_toxic() {
        let mut vpin = VpinTracker::new(1000.0, 5);
        // All buys = maximally toxic
        for _ in 0..10 {
            vpin.update(true, 1000.0);
        }
        assert!(
            vpin.vpin() > 0.9,
            "all-buy flow should give high VPIN: {}",
            vpin.vpin()
        );
    }

    #[test]
    fn test_vpin_bucket_rollover() {
        let mut vpin = VpinTracker::new(100.0, 3);
        // Fill exactly 3 buckets with buys
        vpin.update(true, 300.0);
        assert_eq!(vpin.buckets.len(), 3);
        assert!((vpin.vpin() - 1.0).abs() < 1e-10);

        // Add one balanced bucket
        vpin.update(true, 50.0);
        vpin.update(false, 50.0);
        // Now 3 buckets (window=3), first was dropped
        assert_eq!(vpin.buckets.len(), 3);
        // Average should be (1.0 + 1.0 + 0.0) / 3 ≈ 0.667
        assert!(
            (vpin.vpin() - 2.0 / 3.0).abs() < 0.01,
            "vpin={}",
            vpin.vpin()
        );
    }

    #[test]
    fn test_vpin_no_buckets() {
        let vpin = VpinTracker::new(1000.0, 5);
        assert!((vpin.vpin() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_trade_imbalance_empty_window() {
        let tracker = TradeImbalanceTracker::new(100);
        assert_eq!(tracker.window_size(), 0);
        assert!((tracker.imbalance() - 0.0).abs() < 1e-10);
        assert!((tracker.buy_ratio() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_trade_imbalance_all_buys() {
        let mut tracker = TradeImbalanceTracker::new(100);
        for i in 0..50 {
            tracker.update(true, 1000.0, i as u64);
        }
        assert!(
            (tracker.imbalance() - 1.0).abs() < 1e-10,
            "all-buy imbalance={}",
            tracker.imbalance()
        );
        assert!((tracker.buy_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_trade_imbalance_all_sells() {
        let mut tracker = TradeImbalanceTracker::new(100);
        for i in 0..50 {
            tracker.update(false, 1000.0, i as u64);
        }
        assert!(
            (tracker.imbalance() - (-1.0)).abs() < 1e-10,
            "all-sell imbalance={}",
            tracker.imbalance()
        );
        assert!((tracker.buy_ratio() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_trade_imbalance_window_eviction() {
        let mut tracker = TradeImbalanceTracker::new(10);
        // Fill with 10 sells
        for i in 0..10 {
            tracker.update(false, 1000.0, i as u64);
        }
        assert_eq!(tracker.window_size(), 10);
        assert!((tracker.imbalance() - (-1.0)).abs() < 1e-10);

        // Add 10 buys — sells should be evicted
        for i in 10..20 {
            tracker.update(true, 1000.0, i as u64);
        }
        assert_eq!(tracker.window_size(), 10);
        assert!(
            (tracker.imbalance() - 1.0).abs() < 1e-10,
            "after eviction imbalance={}",
            tracker.imbalance()
        );
    }

    #[test]
    fn test_trade_imbalance_disabled_max_size_zero() {
        let mut tracker = TradeImbalanceTracker::new(0);
        tracker.update(true, 1000.0, 0);
        tracker.update(true, 1000.0, 1);
        assert_eq!(tracker.window_size(), 0);
        assert!((tracker.imbalance() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_vpin_overflow_carry() {
        let mut vpin = VpinTracker::new(100.0, 10);
        // Single trade larger than bucket size should fill multiple buckets
        vpin.update(true, 250.0);
        assert_eq!(
            vpin.buckets.len(),
            2,
            "250 notional with 100 bucket size = 2 complete buckets"
        );
        // Both buckets should be 1.0 (pure buy)
        assert!((vpin.vpin() - 1.0).abs() < 1e-10);
        // Current bucket should have 50 remaining
    }

    // --- RegimeDetector tests ---

    #[test]
    fn test_regime_disabled() {
        let mut rd = RegimeDetector::new(0, 100.0, 50.0, 3.0);
        rd.update(1000.0, 0);
        rd.update(2000.0, 1_000_000_000); // huge jump — should be ignored
        assert!((rd.spread_multiplier() - 1.0).abs() < 1e-10);
        assert!(!rd.is_trending());
        assert!(rd.drift_bps_hr().abs() < 1e-10);
    }

    #[test]
    fn test_regime_flat_price() {
        let mut rd = RegimeDetector::new(3600, 100.0, 50.0, 3.0);
        let ns_per_sec = 1_000_000_000u64;
        for i in 0..10u64 {
            rd.update(1000.0, i * 360 * ns_per_sec);
        }
        assert!(rd.drift_bps_hr().abs() < 1e-6, "drift={}", rd.drift_bps_hr());
        assert!(!rd.is_trending());
        assert!((rd.spread_multiplier() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_regime_uptrend() {
        // 120 bps/hr uptrend over 60s window — well above drift_enter=100
        let mut rd = RegimeDetector::new(60, 100.0, 50.0, 3.0);
        let ns = 1_000_000_000u64;
        rd.update(1000.0, 0);
        // drift_bps = 0.2/1000 * 10000 = 2 bps; drift_bps_hr = 2 * 3600/60 = 120
        rd.update(1000.2, 60 * ns);
        assert!(rd.drift_bps_hr() > 100.0, "drift={}", rd.drift_bps_hr());
        assert!(rd.is_trending());
        assert!(rd.spread_multiplier() > 1.0);
        assert!(rd.spread_multiplier() <= 3.0);
    }

    #[test]
    fn test_regime_downtrend_same_as_uptrend() {
        // Abs drift is used — downtrend gives same multiplier as equivalent uptrend.
        let ns = 1_000_000_000u64;
        let mut rd_up = RegimeDetector::new(60, 100.0, 50.0, 3.0);
        rd_up.update(1000.0, 0);
        rd_up.update(1000.2, 60 * ns); // 120 bps/hr up

        let mut rd_down = RegimeDetector::new(60, 100.0, 50.0, 3.0);
        rd_down.update(1000.2, 0);
        rd_down.update(1000.0, 60 * ns); // ~120 bps/hr down

        assert!(rd_up.is_trending());
        assert!(rd_down.is_trending());
        // Both clamped to max_mult=3.0 (drift >> enter)
        assert!(
            (rd_up.spread_multiplier() - rd_down.spread_multiplier()).abs() < 1e-10,
            "up={}, down={}",
            rd_up.spread_multiplier(),
            rd_down.spread_multiplier()
        );
    }

    #[test]
    fn test_regime_hysteresis() {
        let mut rd = RegimeDetector::new(60, 100.0, 50.0, 3.0);
        let ns = 1_000_000_000u64;

        // Step 1: Enter trending at 120 bps/hr
        rd.update(1000.0, 0);
        rd.update(1000.2, 60 * ns);
        assert!(rd.is_trending(), "should enter trending");

        // Step 2: Drift drops to ~75 bps/hr (between exit=50 and enter=100) — stays trending.
        // At t=120s: t=0 evicted, window = [(1000.2, 60s), (?, 120s)].
        // drift_bps = 75*60/3600 = 1.25; newest = 1000.2 * (1 + 1.25/10000) ≈ 1000.325
        rd.update(1000.325, 120 * ns);
        assert!(
            rd.is_trending(),
            "hysteresis: drift between exit and enter, should stay trending"
        );

        // Step 3: Drift drops below exit threshold (< 50 bps/hr) — exits trending.
        // At t=180s: t=60s evicted, window = [(1000.325, 120s), (?, 180s)].
        // drift_bps = 18*60/3600 = 0.3; newest = 1000.325 * (1 + 0.3/10000) ≈ 1000.355
        rd.update(1000.355, 180 * ns);
        assert!(!rd.is_trending(), "should exit trending when drift < exit threshold");
        assert!((rd.spread_multiplier() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_regime_eviction() {
        let mut rd = RegimeDetector::new(60, 100.0, 50.0, 3.0);
        let ns = 1_000_000_000u64;

        // Establish trending
        rd.update(1000.0, 0);
        rd.update(1000.2, 60 * ns); // 120 bps/hr
        assert!(rd.is_trending());

        // Advance time past window — both prior entries evicted (130s and 70s > 60s window)
        rd.update(1000.2, 130 * ns);
        assert!(
            rd.drift_bps_hr().abs() < 1e-6,
            "trend data evicted; drift should be 0, got {}",
            rd.drift_bps_hr()
        );
        assert!(!rd.is_trending(), "should exit trending after window clears");
    }

    #[test]
    fn test_regime_equal_thresholds_no_nan() {
        // drift_enter == drift_exit → binary mode: no ramp, no NaN/panic.
        let mut rd = RegimeDetector::new(60, 100.0, 100.0, 3.0);
        let ns = 1_000_000_000u64;
        rd.update(1000.0, 0);
        rd.update(1000.2, 60 * ns); // 120 bps/hr > 100 → enters trending
        assert!(rd.is_trending());
        let mult = rd.spread_multiplier();
        assert!(mult.is_finite(), "must not produce NaN or infinity: {}", mult);
        assert!((mult - 3.0).abs() < 1e-10, "binary mode should return max_mult={}", mult);
    }

    #[test]
    fn test_regime_single_entry() {
        let mut rd = RegimeDetector::new(60, 100.0, 50.0, 3.0);
        rd.update(1000.0, 0);
        assert!((rd.spread_multiplier() - 1.0).abs() < 1e-10);
        assert!(rd.drift_bps_hr().abs() < 1e-10);
    }
}

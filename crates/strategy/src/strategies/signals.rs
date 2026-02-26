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
        assert!(vpin.vpin() < 0.1, "balanced flow should give low VPIN: {}", vpin.vpin());
    }

    #[test]
    fn test_vpin_toxic() {
        let mut vpin = VpinTracker::new(1000.0, 5);
        // All buys = maximally toxic
        for _ in 0..10 {
            vpin.update(true, 1000.0);
        }
        assert!(vpin.vpin() > 0.9, "all-buy flow should give high VPIN: {}", vpin.vpin());
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
        assert!((vpin.vpin() - 2.0/3.0).abs() < 0.01, "vpin={}", vpin.vpin());
    }

    #[test]
    fn test_vpin_no_buckets() {
        let vpin = VpinTracker::new(1000.0, 5);
        assert!((vpin.vpin() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_vpin_overflow_carry() {
        let mut vpin = VpinTracker::new(100.0, 10);
        // Single trade larger than bucket size should fill multiple buckets
        vpin.update(true, 250.0);
        assert_eq!(vpin.buckets.len(), 2, "250 notional with 100 bucket size = 2 complete buckets");
        // Both buckets should be 1.0 (pure buy)
        assert!((vpin.vpin() - 1.0).abs() < 1e-10);
        // Current bucket should have 50 remaining
    }
}

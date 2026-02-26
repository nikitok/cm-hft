//! Reusable signal components for adaptive strategies.
//!
//! Provides EMA, volatility tracking, and trade-flow imbalance signals.

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
}

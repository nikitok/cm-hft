//! Symbol analysis engine for HFT market-making suitability evaluation.
//!
//! [`SymbolAnalyzer`] accumulates orderbook and trade microstructure metrics
//! in constant memory using streaming algorithms (Welford's online variance,
//! VecDeque sliding window for percentiles). At the end of the collection
//! window, [`SymbolAnalyzer::report`] produces an [`AnalysisReport`] with
//! a go/no-go [`AnalysisVerdict`] based on configurable [`Thresholds`].

use std::collections::VecDeque;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use cm_core::types::Trade;
use cm_market_data::orderbook::OrderBook;

// ─── Thresholds ──────────────────────────────────────────────────────────────

/// Configurable thresholds for the go/no-go verdict.
///
/// All defaults are conservative but appropriate for liquid spot pairs
/// on Binance (e.g. BTCUSDT, ETHUSDT).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thresholds {
    /// Maximum acceptable mean spread in basis points. Default: 5.0 bps.
    pub max_spread_bps: f64,
    /// Minimum acceptable trade rate (trades per second). Default: 1.0.
    pub min_trade_rate: f64,
    /// Minimum mean top-5 depth per side in USD. Default: $10,000.
    pub min_depth_usd: f64,
    /// Maximum acceptable mid-price std dev in bps. Default: 50.0 bps.
    pub max_volatility_bps: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            max_spread_bps: 5.0,
            min_trade_rate: 1.0,
            min_depth_usd: 10_000.0,
            max_volatility_bps: 50.0,
        }
    }
}

// ─── Verdict ─────────────────────────────────────────────────────────────────

/// Go/no-go verdict for a symbol.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "PascalCase")]
pub enum AnalysisVerdict {
    /// Symbol passes all thresholds — suitable for market making.
    Go,
    /// Symbol fails one or more thresholds.
    NoGo {
        /// Human-readable reasons for each failed threshold.
        reasons: Vec<String>,
    },
    /// Insufficient data collected to make a determination.
    Insufficient {
        /// Why the verdict cannot be determined.
        reason: String,
    },
}

// ─── Report ──────────────────────────────────────────────────────────────────

/// Full microstructure report for the analyzed symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    /// Actual elapsed collection time in seconds.
    pub duration_secs: f64,
    /// Number of order book update samples processed.
    pub book_samples: u64,
    /// Total trades observed.
    pub trade_count: u64,
    /// Mean trades per second over the collection window.
    pub trade_rate_per_sec: f64,
    /// Total USD volume traded.
    pub trade_volume_usd: f64,
    /// Mean spread in basis points.
    pub spread_bps_mean: f64,
    /// Minimum observed spread in bps.
    pub spread_bps_min: f64,
    /// Maximum observed spread in bps.
    pub spread_bps_max: f64,
    /// Median (p50) spread in bps (from sliding window of last 1000 samples).
    pub spread_bps_p50: f64,
    /// 95th percentile spread in bps.
    pub spread_bps_p95: f64,
    /// Mean top-5 bid-side depth in USD (per-level notional: Σ price_i × qty_i).
    pub bid_depth_usd_mean: f64,
    /// Mean top-5 ask-side depth in USD.
    pub ask_depth_usd_mean: f64,
    /// Mid-price volatility: standard deviation in basis points over the window.
    pub mid_price_volatility_bps: f64,
    /// Mean bid/ask imbalance (0 = all ask, 0.5 = balanced, 1 = all bid).
    pub bid_ask_imbalance_mean: f64,
    /// Go/no-go verdict based on configured thresholds.
    pub verdict: AnalysisVerdict,
}

// ─── Analyzer ────────────────────────────────────────────────────────────────

/// Streaming microstructure metric accumulator.
///
/// Processes order book and trade events in constant memory. Designed to be
/// driven by a main event loop and queried once at shutdown via [`report`].
///
/// [`report`]: SymbolAnalyzer::report
pub struct SymbolAnalyzer {
    // Spread (bps) sliding window — last 1000 samples for p50/p95.
    spread_samples: VecDeque<f64>,
    spread_sum: f64,
    spread_count: u64,
    spread_min: f64,
    spread_max: f64,

    // Trade metrics.
    trade_count: u64,
    trade_volume_usd: f64,

    // Depth (USD) running averages — accumulated per book update.
    bid_depth_usd_sum: f64,
    ask_depth_usd_sum: f64,
    depth_sample_count: u64,

    // Bid/ask imbalance running average.
    imbalance_sum: f64,
    imbalance_count: u64,

    // Mid-price volatility via Welford's online algorithm: (count, mean, M2).
    mid_welford_n: u64,
    mid_welford_mean: f64,
    mid_welford_m2: f64,
}

impl SymbolAnalyzer {
    /// Create a new, empty analyzer.
    pub fn new() -> Self {
        Self {
            spread_samples: VecDeque::with_capacity(1000),
            spread_sum: 0.0,
            spread_count: 0,
            spread_min: f64::MAX,
            spread_max: f64::MIN,

            trade_count: 0,
            trade_volume_usd: 0.0,

            bid_depth_usd_sum: 0.0,
            ask_depth_usd_sum: 0.0,
            depth_sample_count: 0,

            imbalance_sum: 0.0,
            imbalance_count: 0,

            mid_welford_n: 0,
            mid_welford_mean: 0.0,
            mid_welford_m2: 0.0,
        }
    }

    /// Process an order book state after each update.
    ///
    /// Should only be called after a successful `OrderBook::apply_update` or
    /// `OrderBook::apply_snapshot` — never on a stale or uninitialized book.
    pub fn on_book_update(&mut self, book: &OrderBook) {
        // Spread in bps.
        if let Some(bps) = book.spread_bps() {
            // Sliding window: evict oldest if at capacity.
            if self.spread_samples.len() == 1000 {
                self.spread_samples.pop_front();
            }
            self.spread_samples.push_back(bps);
            self.spread_sum += bps;
            self.spread_count += 1;
            if bps < self.spread_min {
                self.spread_min = bps;
            }
            if bps > self.spread_max {
                self.spread_max = bps;
            }
        }

        // Depth in USD: Σ(price_i × qty_i) for top-5 levels per side.
        let bid_depth_usd: f64 = book
            .bid_depth(5)
            .iter()
            .map(|l| l.price.to_f64() * l.quantity.to_f64())
            .sum();
        let ask_depth_usd: f64 = book
            .ask_depth(5)
            .iter()
            .map(|l| l.price.to_f64() * l.quantity.to_f64())
            .sum();

        self.bid_depth_usd_sum += bid_depth_usd;
        self.ask_depth_usd_sum += ask_depth_usd;
        self.depth_sample_count += 1;

        // Bid/ask imbalance: bid_depth / (bid_depth + ask_depth).
        let total_depth = bid_depth_usd + ask_depth_usd;
        if total_depth > 0.0 {
            self.imbalance_sum += bid_depth_usd / total_depth;
            self.imbalance_count += 1;
        }

        // Mid-price volatility via Welford's online algorithm.
        if let Some(mid) = book.mid_price() {
            let x = mid.to_f64();
            self.mid_welford_n += 1;
            let delta = x - self.mid_welford_mean;
            self.mid_welford_mean += delta / self.mid_welford_n as f64;
            let delta2 = x - self.mid_welford_mean;
            self.mid_welford_m2 += delta * delta2;
        }
    }

    /// Record a trade event.
    pub fn on_trade(&mut self, trade: &Trade) {
        self.trade_count += 1;
        self.trade_volume_usd += trade.price.to_f64() * trade.quantity.to_f64();
    }

    /// Produce the final analysis report.
    ///
    /// `duration` is the **actual** elapsed time (not configured target), so
    /// that trade_rate is correctly computed when the feed ends early.
    pub fn report(&self, duration: Duration, thresholds: &Thresholds) -> AnalysisReport {
        let duration_secs = duration.as_secs_f64();

        // Spread percentiles from the sliding window.
        let (spread_bps_p50, spread_bps_p95) = if self.spread_samples.is_empty() {
            (0.0, 0.0)
        } else {
            let mut sorted: Vec<f64> = self.spread_samples.iter().copied().collect();
            // Filter NaN values defensively before sorting to avoid unwrap panic.
            sorted.retain(|x| x.is_finite());
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if sorted.is_empty() {
                (0.0, 0.0)
            } else {
                let n = sorted.len();
                // True median: average of two middle elements for even n.
                let p50 = if n.is_multiple_of(2) {
                    (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                } else {
                    sorted[n / 2]
                };
                let p95_idx = ((n as f64 * 0.95) as usize).min(n - 1);
                let p95 = sorted[p95_idx];
                (p50, p95)
            }
        };

        let spread_bps_mean = if self.spread_count > 0 {
            self.spread_sum / self.spread_count as f64
        } else {
            0.0
        };
        let spread_bps_min = if self.spread_count > 0 {
            self.spread_min
        } else {
            0.0
        };
        let spread_bps_max = if self.spread_count > 0 {
            self.spread_max
        } else {
            0.0
        };

        let trade_rate_per_sec = if duration_secs > 0.0 {
            self.trade_count as f64 / duration_secs
        } else {
            0.0
        };

        let bid_depth_usd_mean = if self.depth_sample_count > 0 {
            self.bid_depth_usd_sum / self.depth_sample_count as f64
        } else {
            0.0
        };
        let ask_depth_usd_mean = if self.depth_sample_count > 0 {
            self.ask_depth_usd_sum / self.depth_sample_count as f64
        } else {
            0.0
        };

        let bid_ask_imbalance_mean = if self.imbalance_count > 0 {
            self.imbalance_sum / self.imbalance_count as f64
        } else {
            0.5
        };

        // Mid-price volatility in bps: (std_dev / mean) * 10_000.
        let mid_price_volatility_bps = if self.mid_welford_n >= 2 {
            let variance = self.mid_welford_m2 / (self.mid_welford_n - 1) as f64;
            let std_dev = variance.sqrt();
            if self.mid_welford_mean > 0.0 {
                (std_dev / self.mid_welford_mean) * 10_000.0
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Verdict logic.
        let verdict = self.compute_verdict(
            spread_bps_mean,
            trade_rate_per_sec,
            bid_depth_usd_mean,
            ask_depth_usd_mean,
            mid_price_volatility_bps,
            thresholds,
        );

        AnalysisReport {
            duration_secs,
            book_samples: self.spread_count,
            trade_count: self.trade_count,
            trade_rate_per_sec,
            trade_volume_usd: self.trade_volume_usd,
            spread_bps_mean,
            spread_bps_min,
            spread_bps_max,
            spread_bps_p50,
            spread_bps_p95,
            bid_depth_usd_mean,
            ask_depth_usd_mean,
            mid_price_volatility_bps,
            bid_ask_imbalance_mean,
            verdict,
        }
    }

    /// Evaluate the collected metrics against the thresholds.
    fn compute_verdict(
        &self,
        spread_bps_mean: f64,
        trade_rate_per_sec: f64,
        bid_depth_usd_mean: f64,
        ask_depth_usd_mean: f64,
        mid_price_volatility_bps: f64,
        thresholds: &Thresholds,
    ) -> AnalysisVerdict {
        // Insufficient data check.
        if self.spread_count < 10 {
            return AnalysisVerdict::Insufficient {
                reason: format!(
                    "only {} book samples collected (minimum 10 required)",
                    self.spread_count
                ),
            };
        }
        if self.trade_count < 5 {
            return AnalysisVerdict::Insufficient {
                reason: format!(
                    "only {} trades observed (minimum 5 required)",
                    self.trade_count
                ),
            };
        }

        let mut reasons = Vec::new();

        if spread_bps_mean > thresholds.max_spread_bps {
            reasons.push(format!(
                "mean spread {:.2} bps exceeds max {:.2} bps",
                spread_bps_mean, thresholds.max_spread_bps
            ));
        }

        if trade_rate_per_sec < thresholds.min_trade_rate {
            reasons.push(format!(
                "trade rate {:.3} trades/sec below min {:.3}",
                trade_rate_per_sec, thresholds.min_trade_rate
            ));
        }

        if bid_depth_usd_mean < thresholds.min_depth_usd {
            reasons.push(format!(
                "mean bid depth ${:.0} below min ${:.0}",
                bid_depth_usd_mean, thresholds.min_depth_usd
            ));
        }

        if ask_depth_usd_mean < thresholds.min_depth_usd {
            reasons.push(format!(
                "mean ask depth ${:.0} below min ${:.0}",
                ask_depth_usd_mean, thresholds.min_depth_usd
            ));
        }

        if mid_price_volatility_bps > thresholds.max_volatility_bps {
            reasons.push(format!(
                "mid-price volatility {:.2} bps exceeds max {:.2} bps",
                mid_price_volatility_bps, thresholds.max_volatility_bps
            ));
        }

        if reasons.is_empty() {
            AnalysisVerdict::Go
        } else {
            AnalysisVerdict::NoGo { reasons }
        }
    }
}

impl Default for SymbolAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use cm_core::types::{Exchange, Price, Quantity, Side, Symbol, Timestamp, Trade};
    use cm_market_data::orderbook::OrderBook;

    // ─── Helpers ─────────────────────────────────────────────────────────────

    /// Create an initialized OrderBook with a single bid and ask at the given prices.
    fn make_book_with_spread(bid: f64, ask: f64) -> OrderBook {
        let mut book = OrderBook::new(Exchange::Binance, Symbol::new("BTCUSDT"));
        let bid_price = Price::from(bid);
        let ask_price = Price::from(ask);
        let qty = Quantity::from(1.0_f64);
        book.apply_snapshot(&[(bid_price, qty)], &[(ask_price, qty)], 1);
        book
    }

    /// Create an initialized OrderBook with 5 bid and 5 ask levels.
    ///
    /// Bid prices: [base, base-1, base-2, base-3, base-4]
    /// Ask prices: [base+spread, base+spread+1, ...]
    /// Each level has the given quantity.
    fn make_book_with_depth(base_bid: f64, spread: f64, qty_per_level: f64) -> OrderBook {
        let mut book = OrderBook::new(Exchange::Binance, Symbol::new("BTCUSDT"));
        let bids: Vec<_> = (0..5)
            .map(|i| {
                (
                    Price::from(base_bid - i as f64),
                    Quantity::from(qty_per_level),
                )
            })
            .collect();
        let asks: Vec<_> = (0..5)
            .map(|i| {
                (
                    Price::from(base_bid + spread + i as f64),
                    Quantity::from(qty_per_level),
                )
            })
            .collect();
        book.apply_snapshot(&bids, &asks, 1);
        book
    }

    fn make_trade(price: f64, qty: f64) -> Trade {
        Trade {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            timestamp: Timestamp::from_millis(0),
            price: Price::from(price),
            quantity: Quantity::from(qty),
            side: Side::Buy,
            trade_id: "1".to_string(),
        }
    }

    // ─── Test 1: Empty analyzer → Insufficient ───────────────────────────────

    #[test]
    fn test_empty_analyzer_returns_insufficient() {
        let analyzer = SymbolAnalyzer::new();
        let report = analyzer.report(Duration::from_secs(30), &Thresholds::default());
        assert!(
            matches!(report.verdict, AnalysisVerdict::Insufficient { .. }),
            "expected Insufficient, got {:?}",
            report.verdict
        );
    }

    // ─── Test 2: Spread accumulation ─────────────────────────────────────────

    #[test]
    fn test_spread_accumulation() {
        let mut analyzer = SymbolAnalyzer::new();

        // Feed 100 book updates with a constant 1-bps spread.
        // bid=50000, ask=50001 → spread=1, mid=50000.5 → bps ≈ 0.19998
        let book = make_book_with_spread(50000.0, 50001.0);
        for _ in 0..100 {
            analyzer.on_book_update(&book);
        }

        let report = analyzer.report(Duration::from_secs(60), &Thresholds::default());
        assert_eq!(report.book_samples, 100);

        let expected_bps = book.spread_bps().unwrap();
        assert!(
            (report.spread_bps_mean - expected_bps).abs() < 1e-6,
            "mean spread"
        );
        assert!(
            (report.spread_bps_min - expected_bps).abs() < 1e-6,
            "min spread"
        );
        assert!(
            (report.spread_bps_max - expected_bps).abs() < 1e-6,
            "max spread"
        );
        assert!(
            (report.spread_bps_p50 - expected_bps).abs() < 1e-4,
            "p50 spread"
        );
        assert!(
            (report.spread_bps_p95 - expected_bps).abs() < 1e-4,
            "p95 spread"
        );
    }

    // ─── Test 3: Trade accumulation ──────────────────────────────────────────

    #[test]
    fn test_trade_accumulation() {
        let mut analyzer = SymbolAnalyzer::new();

        // Also need some book updates for the Insufficient threshold.
        let book = make_book_with_spread(50000.0, 50001.0);
        for _ in 0..20 {
            analyzer.on_book_update(&book);
        }

        // 10 trades at $50,000 × 0.001 BTC = $50 each.
        for _ in 0..10 {
            analyzer.on_trade(&make_trade(50_000.0, 0.001));
        }

        let report = analyzer.report(Duration::from_secs(10), &Thresholds::default());
        assert_eq!(report.trade_count, 10);
        assert!((report.trade_volume_usd - 500.0).abs() < 1e-3, "volume_usd");
        assert!((report.trade_rate_per_sec - 1.0).abs() < 1e-6, "trade_rate");
    }

    // ─── Test 4: Depth USD formula (per-level notional) ──────────────────────

    #[test]
    fn test_depth_usd_per_level_notional() {
        let mut analyzer = SymbolAnalyzer::new();

        // Book with 5 bid levels at different prices, 1 BTC each.
        // Bid prices: 50000, 49999, 49998, 49997, 49996
        // Expected depth USD = sum(price_i * qty_i) = 50000+49999+49998+49997+49996 = 249990
        let book = make_book_with_depth(50000.0, 5.0, 1.0);

        let expected_bid_depth: f64 = book
            .bid_depth(5)
            .iter()
            .map(|l| l.price.to_f64() * l.quantity.to_f64())
            .sum();

        analyzer.on_book_update(&book);
        let report = analyzer.report(
            Duration::from_secs(60),
            &Thresholds {
                min_trade_rate: 0.0, // allow no trades
                min_depth_usd: 0.0,
                ..Default::default()
            },
        );

        // mean_bid_depth == single sample value
        assert!(
            (report.bid_depth_usd_mean - expected_bid_depth).abs() < 1.0,
            "bid depth USD: got {}, expected {}",
            report.bid_depth_usd_mean,
            expected_bid_depth
        );
    }

    // ─── Test 5: Volatility = 0 with constant mid ────────────────────────────

    #[test]
    fn test_volatility_zero_constant_mid() {
        let mut analyzer = SymbolAnalyzer::new();
        let book = make_book_with_spread(50000.0, 50001.0);
        for _ in 0..50 {
            analyzer.on_book_update(&book);
        }
        let report = analyzer.report(Duration::from_secs(60), &Thresholds::default());
        // Constant mid → variance ≈ 0.
        assert!(
            report.mid_price_volatility_bps < 1e-6,
            "expected near-zero volatility, got {}",
            report.mid_price_volatility_bps
        );
    }

    // ─── Test 6: Volatility non-zero with alternating mid ────────────────────

    #[test]
    fn test_volatility_nonzero_alternating_mid() {
        let mut analyzer = SymbolAnalyzer::new();
        // Alternate between two different mid prices.
        let book_a = make_book_with_spread(50000.0, 50001.0); // mid ≈ 50000.5
        let book_b = make_book_with_spread(50100.0, 50101.0); // mid ≈ 50100.5
        for _ in 0..50 {
            analyzer.on_book_update(&book_a);
            analyzer.on_book_update(&book_b);
        }
        let report = analyzer.report(Duration::from_secs(60), &Thresholds::default());
        assert!(
            report.mid_price_volatility_bps > 0.1,
            "expected non-zero volatility, got {}",
            report.mid_price_volatility_bps
        );
    }

    // ─── Test 7: Go verdict (all thresholds pass) ─────────────────────────────

    #[test]
    fn test_go_verdict() {
        let mut analyzer = SymbolAnalyzer::new();

        // Tight spread book with good depth.
        let book = make_book_with_depth(50000.0, 1.0, 0.5);
        for _ in 0..20 {
            analyzer.on_book_update(&book);
        }
        // 10 trades — enough for the Insufficient threshold.
        for _ in 0..10 {
            analyzer.on_trade(&make_trade(50000.5, 0.1));
        }

        let thresholds = Thresholds {
            max_spread_bps: 1000.0, // very relaxed
            min_trade_rate: 0.0,
            min_depth_usd: 0.0,
            max_volatility_bps: 10000.0,
        };

        let report = analyzer.report(Duration::from_secs(10), &thresholds);
        assert_eq!(report.verdict, AnalysisVerdict::Go, "expected Go");
    }

    // ─── Test 8: NoGo verdict — wide spread ──────────────────────────────────

    #[test]
    fn test_nogo_wide_spread() {
        let mut analyzer = SymbolAnalyzer::new();

        // Very wide spread: bid=40000, ask=50000 → spread=10000, mid=45000 → bps ≈ 2222
        let book = make_book_with_spread(40000.0, 50000.0);
        for _ in 0..20 {
            analyzer.on_book_update(&book);
        }
        for _ in 0..10 {
            analyzer.on_trade(&make_trade(45000.0, 0.1));
        }

        let thresholds = Thresholds {
            max_spread_bps: 5.0, // far below actual 2222 bps
            min_trade_rate: 0.0,
            min_depth_usd: 0.0,
            max_volatility_bps: 10000.0,
        };

        let report = analyzer.report(Duration::from_secs(10), &thresholds);
        match &report.verdict {
            AnalysisVerdict::NoGo { reasons } => {
                assert!(
                    reasons.iter().any(|r| r.contains("spread")),
                    "expected spread in NoGo reasons: {:?}",
                    reasons
                );
            }
            other => panic!("expected NoGo, got {:?}", other),
        }
    }

    // ─── Test 9: NoGo verdict — low trade rate ───────────────────────────────

    #[test]
    fn test_nogo_low_trade_rate() {
        let mut analyzer = SymbolAnalyzer::new();

        let book = make_book_with_depth(50000.0, 1.0, 1.0);
        for _ in 0..20 {
            analyzer.on_book_update(&book);
        }
        // 5 trades in 300 seconds → 0.0167 trades/sec < threshold of 1.0
        for _ in 0..5 {
            analyzer.on_trade(&make_trade(50000.5, 0.01));
        }

        let thresholds = Thresholds {
            max_spread_bps: 1000.0,
            min_trade_rate: 1.0,
            min_depth_usd: 0.0,
            max_volatility_bps: 10000.0,
        };

        let report = analyzer.report(Duration::from_secs(300), &thresholds);
        match &report.verdict {
            AnalysisVerdict::NoGo { reasons } => {
                assert!(
                    reasons.iter().any(|r| r.contains("trade rate")),
                    "expected 'trade rate' in NoGo reasons: {:?}",
                    reasons
                );
            }
            other => panic!("expected NoGo (low trade_rate), got {:?}", other),
        }
    }

    // ─── Test 10: Custom thresholds override ─────────────────────────────────

    #[test]
    fn test_custom_thresholds_override() {
        let mut analyzer = SymbolAnalyzer::new();

        let book = make_book_with_depth(50000.0, 1.0, 0.1);
        for _ in 0..20 {
            analyzer.on_book_update(&book);
        }
        for _ in 0..10 {
            analyzer.on_trade(&make_trade(50000.5, 0.01));
        }

        // With default thresholds: depth might fail ($10k min vs small levels).
        // With relaxed thresholds: should pass.
        let relaxed = Thresholds {
            max_spread_bps: 1000.0,
            min_trade_rate: 0.0,
            min_depth_usd: 0.0, // 0 → always passes
            max_volatility_bps: 10000.0,
        };

        let report_relaxed = analyzer.report(Duration::from_secs(10), &relaxed);
        assert_eq!(
            report_relaxed.verdict,
            AnalysisVerdict::Go,
            "relaxed should be Go"
        );

        // Tight thresholds should fail.
        let tight = Thresholds {
            max_spread_bps: 0.0001, // impossibly tight
            min_trade_rate: 0.0,
            min_depth_usd: 0.0,
            max_volatility_bps: 10000.0,
        };
        let report_tight = analyzer.report(Duration::from_secs(10), &tight);
        assert!(
            matches!(report_tight.verdict, AnalysisVerdict::NoGo { .. }),
            "tight spread threshold should produce NoGo"
        );
    }

    // ─── Test 11: Insufficient — too few trades ───────────────────────────────

    #[test]
    fn test_insufficient_too_few_trades() {
        let mut analyzer = SymbolAnalyzer::new();
        let book = make_book_with_spread(50000.0, 50001.0);
        for _ in 0..20 {
            analyzer.on_book_update(&book);
        }
        // Only 4 trades — below the minimum of 5.
        for _ in 0..4 {
            analyzer.on_trade(&make_trade(50000.5, 0.01));
        }
        let report = analyzer.report(Duration::from_secs(30), &Thresholds::default());
        assert!(
            matches!(report.verdict, AnalysisVerdict::Insufficient { .. }),
            "4 trades should be Insufficient"
        );
    }

    // ─── Test 12: Sliding window for p50/p95 ─────────────────────────────────

    #[test]
    fn test_sliding_window_p50_p95() {
        let mut analyzer = SymbolAnalyzer::new();

        // First 1000 updates: narrow spread (bid=50000, ask=50000.5)
        let narrow_book = make_book_with_spread(50000.0, 50000.5);
        for _ in 0..1000 {
            analyzer.on_book_update(&narrow_book);
        }

        // Next 1000 updates: wide spread (bid=40000, ask=50000)
        let wide_book = make_book_with_spread(40000.0, 50000.0);
        for _ in 0..1000 {
            analyzer.on_book_update(&wide_book);
        }

        // The sliding window retains only the last 1000 (all wide).
        // So p50 and p95 should reflect the wide spread, not the narrow one.
        let report = analyzer.report(Duration::from_secs(60), &Thresholds::default());

        let wide_bps = wide_book.spread_bps().unwrap();
        let narrow_bps = narrow_book.spread_bps().unwrap();

        // p50 should be close to wide spread, not narrow.
        assert!(
            (report.spread_bps_p50 - wide_bps).abs() < 1.0,
            "sliding window: p50 should be wide ({:.2}), got {:.2} (narrow={:.2})",
            wide_bps,
            report.spread_bps_p50,
            narrow_bps
        );
    }

    // ─── Test 13: JSON serialization roundtrip ───────────────────────────────

    #[test]
    fn test_report_json_roundtrip() {
        let mut analyzer = SymbolAnalyzer::new();
        let book = make_book_with_spread(50000.0, 50001.0);
        for _ in 0..15 {
            analyzer.on_book_update(&book);
        }
        for _ in 0..10 {
            analyzer.on_trade(&make_trade(50000.5, 0.01));
        }
        let report = analyzer.report(Duration::from_secs(30), &Thresholds::default());
        let json = serde_json::to_string(&report).expect("serialize report");
        let deserialized: AnalysisReport = serde_json::from_str(&json).expect("deserialize report");
        assert!((deserialized.spread_bps_mean - report.spread_bps_mean).abs() < 1e-10);
        assert_eq!(deserialized.trade_count, report.trade_count);
    }

    // ─── Test 14: Thresholds JSON roundtrip ──────────────────────────────────

    #[test]
    fn test_thresholds_json_roundtrip() {
        let thresholds = Thresholds {
            max_spread_bps: 3.5,
            min_trade_rate: 2.0,
            min_depth_usd: 50_000.0,
            max_volatility_bps: 25.0,
        };
        let json = serde_json::to_string(&thresholds).expect("serialize");
        let parsed: Thresholds = serde_json::from_str(&json).expect("deserialize");
        assert!((parsed.max_spread_bps - 3.5).abs() < 1e-10);
        assert!((parsed.min_depth_usd - 50_000.0).abs() < 1e-10);
    }
}

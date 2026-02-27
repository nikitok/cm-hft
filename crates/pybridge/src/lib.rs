#![allow(clippy::useless_conversion)]
//! # cm-pybridge
//!
//! PyO3 bindings exposing the Rust trading core to Python for backtesting.
//! Provides the event replay engine, simulated exchange, and direct access
//! to strategy execution from Python orchestration code.

pub mod replay;
pub mod sim_exchange;

use pyo3::prelude::*;

use cm_core::types::*;
use cm_market_data::orderbook::OrderBook;
use cm_strategy::traits::StrategyParams;
use cm_strategy::{default_registry, OrderAction, TradingContext};

use crate::replay::{ReplayEngine, ReplayEvent};
use crate::sim_exchange::{FeeConfig, LatencyConfig, SimExchange};

/// Result of a backtest run, exposed to Python.
#[pyclass]
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Total realized PnL.
    #[pyo3(get)]
    pub total_pnl: f64,
    /// Total fees paid (negative = net rebate).
    #[pyo3(get)]
    pub total_fees: f64,
    /// Number of round-trip trades.
    #[pyo3(get)]
    pub trade_count: usize,
    /// Number of individual fills.
    #[pyo3(get)]
    pub fill_count: usize,
    /// Maximum absolute position reached.
    #[pyo3(get)]
    pub max_position: f64,
    /// PnL at each event (for equity curve).
    #[pyo3(get)]
    pub pnl_series: Vec<f64>,
    /// Timestamps for the PnL series.
    #[pyo3(get)]
    pub timestamp_series: Vec<u64>,
    /// All individual fill records.
    #[pyo3(get)]
    pub trades: Vec<TradeRecord>,
}

/// A single fill/trade record, exposed to Python.
#[pyclass]
#[derive(Debug, Clone)]
pub struct TradeRecord {
    /// Nanosecond timestamp.
    #[pyo3(get)]
    pub timestamp_ns: u64,
    /// Side as string ("Buy" or "Sell").
    #[pyo3(get)]
    pub side: String,
    /// Fill price.
    #[pyo3(get)]
    pub price: f64,
    /// Fill quantity.
    #[pyo3(get)]
    pub quantity: f64,
    /// Fee for this fill.
    #[pyo3(get)]
    pub fee: f64,
    /// Realized PnL from this fill.
    #[pyo3(get)]
    pub pnl: f64,
}

/// Configuration for a backtest run, exposed to Python.
#[pyclass]
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Strategy name (must be registered in the strategy registry).
    #[pyo3(get, set)]
    pub strategy_name: String,
    /// Strategy parameters as a JSON string.
    #[pyo3(get, set)]
    pub params: String,
    /// Maker fee rate (negative = rebate).
    #[pyo3(get, set)]
    pub maker_fee: f64,
    /// Taker fee rate.
    #[pyo3(get, set)]
    pub taker_fee: f64,
    /// Order latency in nanoseconds.
    #[pyo3(get, set)]
    pub latency_ns: u64,
}

#[pymethods]
impl BacktestConfig {
    /// Create a new backtest config with default fee/latency settings.
    #[new]
    fn new(strategy_name: String) -> Self {
        Self {
            strategy_name,
            params: "{}".to_string(),
            maker_fee: -0.0001,
            taker_fee: 0.0004,
            latency_ns: 1_000_000,
        }
    }
}

/// Run a backtest with synthetically generated market data.
///
/// Generates a random-walk price series around `initial_price` with the given
/// `volatility`, creates the strategy from the registry, runs the full replay
/// loop, and returns aggregated results.
#[pyfunction]
fn run_backtest_synthetic(
    config: &BacktestConfig,
    num_events: usize,
    initial_price: f64,
    volatility: f64,
) -> PyResult<BacktestResult> {
    let result = run_backtest_inner(config, num_events, initial_price, volatility)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:#}", e)))?;
    Ok(result)
}

/// Internal backtest implementation (not PyO3-dependent for testability).
pub fn run_backtest_inner(
    config: &BacktestConfig,
    num_events: usize,
    initial_price: f64,
    volatility: f64,
) -> anyhow::Result<BacktestResult> {
    // Create strategy from registry
    let registry = default_registry();
    let params: serde_json::Value =
        serde_json::from_str(&config.params).unwrap_or_else(|_| serde_json::json!({}));
    let strategy_params = StrategyParams { params };

    let mut strategy = registry
        .create(&config.strategy_name, &strategy_params)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "unknown strategy '{}', available: {:?}",
                config.strategy_name,
                registry.available_strategies()
            )
        })?;

    // Create sim exchange
    let mut sim = SimExchange::new(
        FeeConfig {
            maker_fee: config.maker_fee,
            taker_fee: config.taker_fee,
        },
        LatencyConfig {
            order_latency_ns: config.latency_ns,
            cancel_latency_ns: config.latency_ns / 2,
        },
    );

    // Create order book
    let mut book = OrderBook::new(Exchange::Binance, Symbol::new("BTCUSDT"));

    // Generate synthetic events using a simple random walk
    let events = generate_synthetic_events(num_events, initial_price, volatility);
    let replay = ReplayEngine::from_events(events);

    // Tracking state
    let mut net_position: f64 = 0.0;
    let mut avg_entry: f64 = 0.0;
    let mut realized_pnl: f64 = 0.0;
    let mut max_position: f64 = 0.0;
    let mut fill_count: usize = 0;
    let mut trade_count: usize = 0;
    let mut pnl_series = Vec::with_capacity(num_events);
    let mut timestamp_series = Vec::with_capacity(num_events);
    let mut trade_records = Vec::new();

    // Map from strategy order actions to sim exchange order IDs
    let mut _active_orders: Vec<OrderId> = Vec::new();

    for (ts, event) in replay.events() {
        sim.set_time(*ts);

        match event {
            ReplayEvent::BookUpdate(update) => {
                // Apply to our order book
                book.apply_snapshot(
                    &update.bids.iter().map(|&(p, q)| (p, q)).collect::<Vec<_>>(),
                    &update.asks.iter().map(|&(p, q)| (p, q)).collect::<Vec<_>>(),
                    *ts, // Use timestamp as update_id for synthetic data
                );

                // Feed book update to sim exchange for fill matching
                sim.on_book_update(update);

                // Process any fills from sim exchange
                let fills = sim.drain_fills();
                for fill in &fills {
                    let fill_price = fill.price.to_f64();
                    let fill_qty = fill.quantity.to_f64();
                    let signed_qty = match fill.side {
                        Side::Buy => fill_qty,
                        Side::Sell => -fill_qty,
                    };

                    // Calculate PnL
                    let fill_pnl =
                        calculate_fill_pnl(net_position, avg_entry, fill_price, signed_qty);

                    // Update position tracking
                    let (new_pos, new_avg) =
                        update_position(net_position, avg_entry, fill_price, signed_qty);
                    net_position = new_pos;
                    avg_entry = new_avg;

                    realized_pnl += fill_pnl;
                    fill_count += 1;
                    if fill_pnl.abs() > 1e-12 {
                        trade_count += 1;
                    }

                    let fee = fill_price * fill_qty * config.maker_fee;

                    trade_records.push(TradeRecord {
                        timestamp_ns: fill.timestamp.as_nanos(),
                        side: format!("{}", fill.side),
                        price: fill_price,
                        quantity: fill_qty,
                        fee,
                        pnl: fill_pnl,
                    });

                    if net_position.abs() > max_position {
                        max_position = net_position.abs();
                    }
                }

                // Feed to strategy
                let ctx_position = if net_position.abs() > 1e-12 {
                    vec![cm_oms::Position {
                        exchange: Exchange::Binance,
                        symbol: Symbol::new("BTCUSDT"),
                        net_quantity: Quantity::from(net_position),
                        avg_entry_price: Price::from(avg_entry),
                        realized_pnl: Price::from(realized_pnl),
                        fill_count: fill_count as u64,
                    }]
                } else {
                    vec![]
                };

                let mut ctx = TradingContext::new(ctx_position, vec![], Timestamp(*ts));
                strategy.on_book_update(&mut ctx, &book);

                // Process strategy order actions
                let actions = ctx.drain_actions();
                for action in actions {
                    match action {
                        OrderAction::Submit {
                            side,
                            price,
                            quantity,
                            ..
                        } => {
                            let oid = sim.submit_order(side, price, quantity);
                            _active_orders.push(oid);
                        }
                        OrderAction::Cancel { order_id } => {
                            sim.cancel_order(order_id);
                        }
                        OrderAction::CancelAll { .. } => {
                            sim.cancel_all();
                            _active_orders.clear();
                        }
                    }
                }

                // Deliver fills to strategy
                for fill in &fills {
                    let mut fill_ctx = TradingContext::new(vec![], vec![], Timestamp(*ts));
                    strategy.on_fill(&mut fill_ctx, fill);
                }
            }
            ReplayEvent::Trade(trade) => {
                let mut ctx = TradingContext::new(vec![], vec![], Timestamp(*ts));
                strategy.on_trade(&mut ctx, trade);
            }
        }

        // Calculate mark-to-market PnL
        let mark_price = book
            .mid_price()
            .map(|p| p.to_f64())
            .unwrap_or(initial_price);
        let unrealized = net_position * (mark_price - avg_entry);
        let total_pnl = realized_pnl + unrealized;

        pnl_series.push(total_pnl);
        timestamp_series.push(*ts);
    }

    Ok(BacktestResult {
        total_pnl: realized_pnl,
        total_fees: sim.total_fees(),
        trade_count,
        fill_count,
        max_position,
        pnl_series,
        timestamp_series,
        trades: trade_records,
    })
}

/// Generate synthetic book update events using a random walk.
fn generate_synthetic_events(
    num_events: usize,
    initial_price: f64,
    volatility: f64,
) -> Vec<(u64, ReplayEvent)> {
    let mut events = Vec::with_capacity(num_events);
    let mut price = initial_price;
    let base_ts: u64 = 1_706_000_000_000_000_000; // ~2024-01-23 in nanos
    let interval_ns: u64 = 100_000_000; // 100ms between events

    // Simple LCG random number generator (deterministic, no external deps)
    let mut rng_state: u64 = 42;

    for i in 0..num_events {
        // LCG: next = (a * state + c) mod m
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);

        // Convert to roughly normal distribution using Box-Muller approximation
        // Simplified: use uniform [-1, 1] scaled by volatility
        let uniform = ((rng_state >> 33) as f64 / (u32::MAX as f64)) * 2.0 - 1.0;
        let delta = uniform * volatility;

        price += delta;
        if price <= 0.0 {
            price = 1.0;
        }

        let spread = price * 0.0001; // 1 bps spread
        let ts = base_ts + (i as u64) * interval_ns;

        let bid = price - spread / 2.0;
        let ask = price + spread / 2.0;

        let update = BookUpdate {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            timestamp: Timestamp(ts),
            bids: vec![(Price::from(bid), Quantity::from(1.0))],
            asks: vec![(Price::from(ask), Quantity::from(1.0))],
            is_snapshot: true,
        };

        events.push((ts, ReplayEvent::BookUpdate(update)));
    }

    events
}

/// Calculate realized PnL from a fill.
fn calculate_fill_pnl(net_position: f64, avg_entry: f64, fill_price: f64, signed_qty: f64) -> f64 {
    if net_position.abs() < 1e-12 {
        // Opening a new position, no PnL
        return 0.0;
    }

    // Check if this fill is reducing the position
    let is_reducing =
        (net_position > 0.0 && signed_qty < 0.0) || (net_position < 0.0 && signed_qty > 0.0);

    if !is_reducing {
        return 0.0;
    }

    let reduce_qty = signed_qty.abs().min(net_position.abs());

    if net_position > 0.0 {
        // Was long, selling to reduce
        (fill_price - avg_entry) * reduce_qty
    } else {
        // Was short, buying to reduce
        (avg_entry - fill_price) * reduce_qty
    }
}

/// Update position tracking after a fill.
fn update_position(
    net_position: f64,
    avg_entry: f64,
    fill_price: f64,
    signed_qty: f64,
) -> (f64, f64) {
    let new_position = net_position + signed_qty;

    if net_position.abs() < 1e-12 {
        // Opening new position
        return (new_position, fill_price);
    }

    let same_direction =
        (net_position > 0.0 && signed_qty > 0.0) || (net_position < 0.0 && signed_qty < 0.0);

    if same_direction {
        // Adding to position: update VWAP
        let old_cost = avg_entry * net_position.abs();
        let new_cost = fill_price * signed_qty.abs();
        let total_qty = net_position.abs() + signed_qty.abs();
        let new_avg = if total_qty.abs() > 1e-12 {
            (old_cost + new_cost) / total_qty
        } else {
            fill_price
        };
        (new_position, new_avg)
    } else {
        // Reducing or flipping
        if new_position.abs() < 1e-12 {
            // Fully closed
            (0.0, 0.0)
        } else if (net_position > 0.0 && new_position < 0.0)
            || (net_position < 0.0 && new_position > 0.0)
        {
            // Flipped: new entry at fill price
            (new_position, fill_price)
        } else {
            // Partially reduced: avg_entry stays the same
            (new_position, avg_entry)
        }
    }
}

/// Python module definition.
#[pymodule]
fn cm_pybridge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BacktestResult>()?;
    m.add_class::<TradeRecord>()?;
    m.add_class::<BacktestConfig>()?;
    m.add_function(wrap_pyfunction!(run_backtest_synthetic, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> BacktestConfig {
        BacktestConfig {
            strategy_name: "market_making".to_string(),
            params: "{}".to_string(),
            maker_fee: -0.0001,
            taker_fee: 0.0004,
            latency_ns: 0, // Zero latency for deterministic tests
        }
    }

    // -- 1. run_backtest_inner with valid config --
    #[test]
    fn test_backtest_runs_to_completion() {
        let config = default_config();
        let result = run_backtest_inner(&config, 100, 50000.0, 1.0).unwrap();

        assert_eq!(result.pnl_series.len(), 100);
        assert_eq!(result.timestamp_series.len(), 100);
    }

    // -- 2. Unknown strategy returns error --
    #[test]
    fn test_unknown_strategy_error() {
        let config = BacktestConfig {
            strategy_name: "nonexistent".to_string(),
            params: "{}".to_string(),
            maker_fee: -0.0001,
            taker_fee: 0.0004,
            latency_ns: 0,
        };
        let result = run_backtest_inner(&config, 10, 50000.0, 1.0);
        assert!(result.is_err());
    }

    // -- 3. Zero events produces empty result --
    #[test]
    fn test_zero_events() {
        let config = default_config();
        let result = run_backtest_inner(&config, 0, 50000.0, 1.0).unwrap();
        assert_eq!(result.pnl_series.len(), 0);
        assert_eq!(result.fill_count, 0);
        assert_eq!(result.trade_count, 0);
    }

    // -- 4. Timestamps are monotonic --
    #[test]
    fn test_timestamps_monotonic() {
        let config = default_config();
        let result = run_backtest_inner(&config, 50, 50000.0, 1.0).unwrap();

        for i in 1..result.timestamp_series.len() {
            assert!(result.timestamp_series[i] >= result.timestamp_series[i - 1]);
        }
    }

    // -- 5. Max position is non-negative --
    #[test]
    fn test_max_position_non_negative() {
        let config = default_config();
        let result = run_backtest_inner(&config, 200, 50000.0, 5.0).unwrap();
        assert!(result.max_position >= 0.0);
    }

    // -- 6. Trade records have valid data --
    #[test]
    fn test_trade_records_valid() {
        let config = default_config();
        let result = run_backtest_inner(&config, 500, 50000.0, 10.0).unwrap();

        for trade in &result.trades {
            assert!(trade.price > 0.0);
            assert!(trade.quantity > 0.0);
            assert!(trade.side == "Buy" || trade.side == "Sell");
            assert!(trade.timestamp_ns > 0);
        }
    }

    // -- 7. PnL series length matches event count --
    #[test]
    fn test_pnl_series_length() {
        let config = default_config();
        let result = run_backtest_inner(&config, 75, 50000.0, 2.0).unwrap();
        assert_eq!(result.pnl_series.len(), 75);
        assert_eq!(result.timestamp_series.len(), 75);
    }

    // -- 8. Custom strategy params --
    #[test]
    fn test_custom_params() {
        let config = BacktestConfig {
            strategy_name: "market_making".to_string(),
            params: r#"{"spread_bps": 20.0, "order_size": 0.01}"#.to_string(),
            maker_fee: -0.0001,
            taker_fee: 0.0004,
            latency_ns: 0,
        };
        let result = run_backtest_inner(&config, 100, 50000.0, 5.0).unwrap();
        assert_eq!(result.pnl_series.len(), 100);
    }

    // -- 9. Generate synthetic events produces correct count --
    #[test]
    fn test_generate_synthetic_events_count() {
        let events = generate_synthetic_events(1000, 50000.0, 5.0);
        assert_eq!(events.len(), 1000);
    }

    // -- 10. Synthetic events have positive prices --
    #[test]
    fn test_synthetic_events_positive_prices() {
        let events = generate_synthetic_events(5000, 50000.0, 100.0);
        for (_, event) in &events {
            if let ReplayEvent::BookUpdate(update) = event {
                assert!(update.bids[0].0.to_f64() > 0.0);
                assert!(update.asks[0].0.to_f64() > 0.0);
            }
        }
    }

    // -- 11. calculate_fill_pnl opening position --
    #[test]
    fn test_fill_pnl_opening() {
        // No existing position -> no PnL
        let pnl = calculate_fill_pnl(0.0, 0.0, 50000.0, 1.0);
        assert!((pnl).abs() < 1e-12);
    }

    // -- 12. calculate_fill_pnl closing long --
    #[test]
    fn test_fill_pnl_closing_long() {
        // Long 1.0 at 50000, sell at 51000 -> PnL = 1000
        let pnl = calculate_fill_pnl(1.0, 50000.0, 51000.0, -1.0);
        assert!((pnl - 1000.0).abs() < 0.01);
    }

    // -- 13. calculate_fill_pnl closing short --
    #[test]
    fn test_fill_pnl_closing_short() {
        // Short 1.0 at 50000, buy at 49000 -> PnL = 1000
        let pnl = calculate_fill_pnl(-1.0, 50000.0, 49000.0, 1.0);
        assert!((pnl - 1000.0).abs() < 0.01);
    }

    // -- 14. calculate_fill_pnl adding to position --
    #[test]
    fn test_fill_pnl_adding() {
        // Long 1.0, buying more -> no PnL
        let pnl = calculate_fill_pnl(1.0, 50000.0, 51000.0, 1.0);
        assert!((pnl).abs() < 1e-12);
    }

    // -- 15. update_position opening --
    #[test]
    fn test_update_position_opening() {
        let (pos, avg) = update_position(0.0, 0.0, 50000.0, 1.0);
        assert!((pos - 1.0).abs() < 1e-12);
        assert!((avg - 50000.0).abs() < 0.01);
    }

    // -- 16. update_position adding --
    #[test]
    fn test_update_position_adding() {
        // Long 1.0 at 50000, buy 1.0 at 52000 -> VWAP = 51000
        let (pos, avg) = update_position(1.0, 50000.0, 52000.0, 1.0);
        assert!((pos - 2.0).abs() < 1e-12);
        assert!((avg - 51000.0).abs() < 0.01);
    }

    // -- 17. update_position reducing --
    #[test]
    fn test_update_position_reducing() {
        // Long 2.0 at 50000, sell 1.0 -> pos=1.0, avg stays 50000
        let (pos, avg) = update_position(2.0, 50000.0, 51000.0, -1.0);
        assert!((pos - 1.0).abs() < 1e-12);
        assert!((avg - 50000.0).abs() < 0.01);
    }

    // -- 18. update_position closing --
    #[test]
    fn test_update_position_closing() {
        let (pos, avg) = update_position(1.0, 50000.0, 51000.0, -1.0);
        assert!((pos).abs() < 1e-12);
        assert!((avg).abs() < 0.01);
    }

    // -- 19. update_position flipping --
    #[test]
    fn test_update_position_flipping() {
        // Long 1.0 at 50000, sell 2.0 at 51000 -> short 1.0 at 51000
        let (pos, avg) = update_position(1.0, 50000.0, 51000.0, -2.0);
        assert!((pos - (-1.0)).abs() < 1e-12);
        assert!((avg - 51000.0).abs() < 0.01);
    }

    // -- 20. Synthetic events are sorted --
    #[test]
    fn test_synthetic_events_sorted() {
        let events = generate_synthetic_events(100, 50000.0, 5.0);
        for i in 1..events.len() {
            assert!(events[i].0 >= events[i - 1].0);
        }
    }

    // -- 21. BacktestConfig new defaults --
    #[test]
    fn test_backtest_config_defaults() {
        let config = BacktestConfig::new("market_making".to_string());
        assert_eq!(config.strategy_name, "market_making");
        assert_eq!(config.params, "{}");
        assert!((config.maker_fee - (-0.0001)).abs() < 1e-12);
        assert!((config.taker_fee - 0.0004).abs() < 1e-12);
        assert_eq!(config.latency_ns, 1_000_000);
    }

    // -- 22. Large backtest doesn't panic --
    #[test]
    fn test_large_backtest() {
        let config = default_config();
        let result = run_backtest_inner(&config, 10_000, 50000.0, 10.0).unwrap();
        assert_eq!(result.pnl_series.len(), 10_000);
    }

    // -- 23. Invalid JSON params fall back to empty --
    #[test]
    fn test_invalid_json_params() {
        let config = BacktestConfig {
            strategy_name: "market_making".to_string(),
            params: "not valid json".to_string(),
            maker_fee: -0.0001,
            taker_fee: 0.0004,
            latency_ns: 0,
        };
        let result = run_backtest_inner(&config, 10, 50000.0, 1.0);
        assert!(result.is_ok());
    }

    // -- 24. Fill count matches trade records --
    #[test]
    fn test_fill_count_matches_records() {
        let config = default_config();
        let result = run_backtest_inner(&config, 500, 50000.0, 5.0).unwrap();
        assert_eq!(result.fill_count, result.trades.len());
    }
}

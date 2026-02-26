//! Replay-based strategy tests using recorded Bybit market data.
//!
//! These tests load JSONL.gz files from `testdata/` and replay them through
//! the strategy pipeline. Tests skip gracefully if data files are absent.

mod replay_harness;

use std::path::Path;

use cm_core::types::Exchange;
use cm_strategy::traits::StrategyParams;

use replay_harness::{default_params, load_events, ReplayTestHarness};

/// Helper: skip test if the data file doesn't exist.
/// Tries multiple duration suffixes (30m, 3m) and path locations.
fn data_path(symbol: &str) -> Option<String> {
    let durations = ["30m", "3m"];
    for dur in &durations {
        let candidates = [
            format!("testdata/bybit_{}_{}.jsonl.gz", symbol, dur),
            format!("../../testdata/bybit_{}_{}.jsonl.gz", symbol, dur),
        ];
        for path in &candidates {
            if Path::new(path).exists() {
                return Some(path.clone());
            }
        }
    }
    None
}

#[test]
fn test_mm_strategy_btc_30m() {
    let path = match data_path("btcusdt") {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no BTCUSDT data file found");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load events");
    assert!(!events.is_empty(), "data file should not be empty");

    let mut harness = ReplayTestHarness::new("simple_mm", default_params(), Exchange::Bybit, "BTCUSDT");
    let result = harness.run(&events);

    // Strategy generates orders on real data.
    assert!(result.order_count > 0, "strategy should generate orders");
    // Fills happen.
    assert!(result.fill_count > 0, "should have fills");
    // No catastrophic loss.
    assert!(
        result.total_pnl > -1000.0,
        "PnL shouldn't blow up: {}",
        result.total_pnl
    );
    // Position bounded by risk limits.
    assert!(
        result.max_position <= 0.1 + 1e-6,
        "position within limits: {}",
        result.max_position
    );
}

#[test]
fn test_mm_strategy_eth_30m() {
    let path = match data_path("ethusdt") {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no ETHUSDT data file found");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load events");
    assert!(!events.is_empty(), "data file should not be empty");

    let mut harness = ReplayTestHarness::new("simple_mm", default_params(), Exchange::Bybit, "ETHUSDT");
    let result = harness.run(&events);

    assert!(result.order_count > 0, "strategy should generate orders");
    assert!(result.fill_count > 0, "should have fills");
    assert!(
        result.total_pnl > -1000.0,
        "PnL shouldn't blow up: {}",
        result.total_pnl
    );
    assert!(
        result.max_position <= 0.1 + 1e-6,
        "position within limits: {}",
        result.max_position
    );
}

#[test]
fn test_mm_different_params() {
    let path = match data_path("btcusdt") {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no BTCUSDT data file found");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load events");

    // Wider spread, larger size.
    let params = StrategyParams {
        params: serde_json::json!({
            "spread_bps": 20.0,
            "order_size": 0.01
        }),
    };

    let mut harness = ReplayTestHarness::new("market_making", params, Exchange::Bybit, "BTCUSDT");
    let result = harness.run(&events);

    // Should still produce orders without panicking.
    assert!(result.order_count > 0, "wider params should still generate orders");
}

#[test]
fn test_no_panics_on_real_data() {
    let path = match data_path("btcusdt") {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no BTCUSDT data file found");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load events");

    // Just verify strategy doesn't crash on 30 min of real data.
    let mut harness = ReplayTestHarness::new("market_making", default_params(), Exchange::Bybit, "BTCUSDT");
    let _result = harness.run(&events);
    // If we get here without panicking, the test passes.
}

#[test]
fn test_adaptive_mm_btc_30m() {
    let path = match data_path("btcusdt") {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no BTCUSDT data file found");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load events");
    assert!(!events.is_empty(), "data file should not be empty");

    let params = StrategyParams {
        params: serde_json::json!({"max_position": 0.01}),
    };
    let mut harness = ReplayTestHarness::new("adaptive_mm", params, Exchange::Bybit, "BTCUSDT");
    let result = harness.run(&events);

    assert!(result.order_count > 0, "strategy should generate orders");
    assert!(result.fill_count > 0, "should have fills");

    // Compare against the old strategy's unbounded position (0.129 BTC).
    // Adaptive MM should have significantly lower max position.
    let mut old_harness = ReplayTestHarness::new("simple_mm", default_params(), Exchange::Bybit, "BTCUSDT");
    let old_result = old_harness.run(&events);
    assert!(
        result.max_position < old_result.max_position,
        "adaptive_mm max_pos ({:.6}) should be less than old mm ({:.6})",
        result.max_position,
        old_result.max_position,
    );
    // Should not have catastrophic loss
    assert!(
        result.total_pnl > -500.0,
        "PnL shouldn't blow up: {}",
        result.total_pnl
    );
}

#[test]
fn test_adaptive_mm_eth_30m() {
    let path = match data_path("ethusdt") {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no ETHUSDT data file found");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load events");
    assert!(!events.is_empty(), "data file should not be empty");

    let params = StrategyParams {
        params: serde_json::json!({"max_position": 0.1}),
    };
    let mut harness = ReplayTestHarness::new("adaptive_mm", params, Exchange::Bybit, "ETHUSDT");
    let result = harness.run(&events);

    assert!(result.order_count > 0, "strategy should generate orders");
    assert!(result.fill_count > 0, "should have fills");

    // Compare against old strategy's unbounded position (0.352 ETH).
    let mut old_harness = ReplayTestHarness::new("simple_mm", default_params(), Exchange::Bybit, "ETHUSDT");
    let old_result = old_harness.run(&events);
    assert!(
        result.max_position < old_result.max_position,
        "adaptive_mm max_pos ({:.6}) should be less than old mm ({:.6})",
        result.max_position,
        old_result.max_position,
    );
    assert!(
        result.total_pnl > -500.0,
        "PnL shouldn't blow up: {}",
        result.total_pnl
    );
}

#[test]
fn test_pnl_series_monotonic_events() {
    let path = match data_path("btcusdt") {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no BTCUSDT data file found");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load events");

    let mut harness = ReplayTestHarness::new("simple_mm", default_params(), Exchange::Bybit, "BTCUSDT");
    let result = harness.run(&events);

    // PnL series should have one entry per book update event.
    let book_count = events
        .iter()
        .filter(|(_, e)| matches!(e, replay_harness::ReplayEvent::Book(_)))
        .count();
    assert_eq!(
        result.pnl_series.len(),
        book_count,
        "PnL series length should match book update count"
    );

    // All PnL values should be finite.
    for (i, pnl) in result.pnl_series.iter().enumerate() {
        assert!(pnl.is_finite(), "PnL at index {} is not finite: {}", i, pnl);
    }
}

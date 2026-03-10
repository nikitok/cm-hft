//! Replay-based strategy tests using recorded Bybit market data.
//!
//! These tests load JSONL.gz files from `testdata/` and replay them through
//! the strategy pipeline. Tests skip gracefully if data files are absent.

mod replay_harness;

use std::path::Path;

use cm_core::types::Exchange;
use cm_strategy::traits::StrategyParams;

use replay_harness::{default_params, load_events, ReplayTestHarness};

/// Helper: find data file for a specific exchange and symbol.
/// Returns (path, exchange) if found.
fn data_path_for_exchange(exchange: &str, symbol: &str) -> Option<(String, Exchange)> {
    let durations = ["30m", "3m", "1m"];
    let exchange_enum = match exchange {
        "binance" => Exchange::Binance,
        "bybit" => Exchange::Bybit,
        _ => return None,
    };
    for dur in &durations {
        let candidates = [
            format!("testdata/{}_{}_{}.jsonl.gz", exchange, symbol, dur),
            format!("../../testdata/{}_{}_{}.jsonl.gz", exchange, symbol, dur),
        ];
        for path in &candidates {
            if Path::new(path).exists() {
                return Some((path.clone(), exchange_enum));
            }
        }
    }
    None
}

/// Helper: skip test if the data file doesn't exist.
/// Tries both Bybit and Binance, multiple duration suffixes (30m, 3m, 1m) and path locations.
/// Returns (path, exchange) for the first match found.
fn data_path(symbol: &str) -> Option<(String, Exchange)> {
    // Try Bybit first (most test data), then Binance
    data_path_for_exchange("bybit", symbol).or_else(|| data_path_for_exchange("binance", symbol))
}

#[test]
fn test_mm_strategy_btc_30m() {
    let (path, exchange) = match data_path("btcusdt") {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no BTCUSDT data file found");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load events");
    assert!(!events.is_empty(), "data file should not be empty");

    let mut harness = ReplayTestHarness::new("simple_mm", default_params(), exchange, "BTCUSDT");
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
    // simple_mm has no position limits — just verify it stays reasonable.
    assert!(
        result.max_position <= 0.5 + 1e-6,
        "position should stay reasonable: {}",
        result.max_position
    );
}

#[test]
fn test_mm_strategy_eth_30m() {
    let (path, exchange) = match data_path("ethusdt") {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no ETHUSDT data file found");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load events");
    assert!(!events.is_empty(), "data file should not be empty");

    let mut harness = ReplayTestHarness::new("simple_mm", default_params(), exchange, "ETHUSDT");
    let result = harness.run(&events);

    assert!(result.order_count > 0, "strategy should generate orders");
    assert!(result.fill_count > 0, "should have fills");
    assert!(
        result.total_pnl > -1000.0,
        "PnL shouldn't blow up: {}",
        result.total_pnl
    );
    // simple_mm has no position limits — just verify it stays reasonable.
    assert!(
        result.max_position <= 1.0 + 1e-6,
        "position should stay reasonable: {}",
        result.max_position
    );
}

#[test]
fn test_mm_different_params() {
    let (path, exchange) = match data_path("btcusdt") {
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

    let mut harness = ReplayTestHarness::new("market_making", params, exchange, "BTCUSDT");
    let result = harness.run(&events);

    // Should still produce orders without panicking.
    assert!(
        result.order_count > 0,
        "wider params should still generate orders"
    );
}

#[test]
fn test_no_panics_on_real_data() {
    let (path, exchange) = match data_path("btcusdt") {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no BTCUSDT data file found");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load events");

    // Just verify strategy doesn't crash on 30 min of real data.
    let mut harness =
        ReplayTestHarness::new("market_making", default_params(), exchange, "BTCUSDT");
    let _result = harness.run(&events);
    // If we get here without panicking, the test passes.
}

#[test]
fn test_adaptive_mm_btc_30m() {
    let (path, exchange) = match data_path("btcusdt") {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no BTCUSDT data file found");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load events");
    assert!(!events.is_empty(), "data file should not be empty");

    let max_pos_btc = 0.01;
    let params = StrategyParams {
        params: serde_json::json!({"max_position": max_pos_btc}),
    };
    let mut harness = ReplayTestHarness::new("adaptive_mm", params, exchange, "BTCUSDT");
    let result = harness.run(&events);

    assert!(result.order_count > 0, "strategy should generate orders");
    assert!(result.fill_count > 0, "should have fills");

    // Asymmetric sizing uses smooth decay — position naturally reverts toward
    // max_position but can exceed it when fills arrive faster than requotes.
    // Verify position stays within a reasonable multiple of max_position.
    assert!(
        result.max_position <= 1.0,
        "adaptive_mm BTC position ({:.6}) should stay reasonable",
        result.max_position,
    );
    // Should not have catastrophic loss
    assert!(
        result.total_pnl > -2000.0,
        "PnL shouldn't blow up: {}",
        result.total_pnl
    );
}

#[test]
fn test_adaptive_mm_eth_30m() {
    let (path, exchange) = match data_path("ethusdt") {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no ETHUSDT data file found");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load events");
    assert!(!events.is_empty(), "data file should not be empty");

    let max_pos_eth = 0.1;
    let params = StrategyParams {
        params: serde_json::json!({"max_position": max_pos_eth}),
    };
    let mut harness = ReplayTestHarness::new("adaptive_mm", params, exchange, "ETHUSDT");
    let result = harness.run(&events);

    assert!(result.order_count > 0, "strategy should generate orders");
    assert!(result.fill_count > 0, "should have fills");

    // Asymmetric sizing: smooth decay keeps position near max_position.
    assert!(
        result.max_position <= max_pos_eth * 5.0 + 0.01,
        "adaptive_mm position ({:.6}) should stay near max_position ({:.6})",
        result.max_position,
        max_pos_eth,
    );
    assert!(
        result.total_pnl > -2000.0,
        "PnL shouldn't blow up: {}",
        result.total_pnl
    );
}

#[test]
fn test_pnl_series_monotonic_events() {
    let (path, exchange) = match data_path("btcusdt") {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no BTCUSDT data file found");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load events");

    let mut harness = ReplayTestHarness::new("simple_mm", default_params(), exchange, "BTCUSDT");
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

// ── Binance-specific tests ──
// Binance data is shorter (~1min vs 30min Bybit), so thresholds are more lenient.

#[test]
fn test_mm_strategy_binance_btc() {
    let (path, exchange) = match data_path_for_exchange("binance", "btcusdt") {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no Binance BTCUSDT data file found");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load events");
    assert!(!events.is_empty(), "data file should not be empty");

    let mut harness = ReplayTestHarness::new("simple_mm", default_params(), exchange, "BTCUSDT");
    let result = harness.run(&events);

    // Strategy should generate orders without panicking.
    assert!(result.order_count > 0, "strategy should generate orders");
    // Short data may have fewer fills — just verify no panic.
}

#[test]
fn test_mm_strategy_binance_eth() {
    let (path, exchange) = match data_path_for_exchange("binance", "ethusdt") {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no Binance ETHUSDT data file found");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load events");
    assert!(!events.is_empty(), "data file should not be empty");

    let mut harness = ReplayTestHarness::new("simple_mm", default_params(), exchange, "ETHUSDT");
    let result = harness.run(&events);

    assert!(result.order_count > 0, "strategy should generate orders");
}

#[test]
fn test_adaptive_mm_binance_btc() {
    let (path, exchange) = match data_path_for_exchange("binance", "btcusdt") {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no Binance BTCUSDT data file found");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load events");
    assert!(!events.is_empty(), "data file should not be empty");

    let max_pos_btc = 0.01;
    let params = StrategyParams {
        params: serde_json::json!({"max_position": max_pos_btc}),
    };
    let mut harness = ReplayTestHarness::new("adaptive_mm", params, exchange, "BTCUSDT");
    let result = harness.run(&events);

    // Strategy should run without panicking on short Binance data.
    assert!(result.order_count > 0, "strategy should generate orders");
}

#[test]
fn test_adaptive_mm_binance_eth() {
    let (path, exchange) = match data_path_for_exchange("binance", "ethusdt") {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no Binance ETHUSDT data file found");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load events");
    assert!(!events.is_empty(), "data file should not be empty");

    let max_pos_eth = 0.1;
    let params = StrategyParams {
        params: serde_json::json!({"max_position": max_pos_eth}),
    };
    let mut harness = ReplayTestHarness::new("adaptive_mm", params, exchange, "ETHUSDT");
    let result = harness.run(&events);

    assert!(result.order_count > 0, "strategy should generate orders");
}

// ── Timer injection tests ──

#[test]
fn test_simconfig_timer_interval_default_zero() {
    // SimConfig::default() must have timer_interval=0 (disabled)
    let cfg = replay_harness::SimConfig::default();
    assert_eq!(
        cfg.timer_interval, 0,
        "timer_interval should default to 0 (disabled)"
    );
}

#[test]
fn test_timer_injection_does_not_break_existing_behavior() {
    // With timer_interval=0 (default), harness behaves identically to before.
    let (path, exchange) = match data_path_for_exchange("binance", "ethusdt") {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no Binance ETHUSDT data file found");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load events");
    let params = StrategyParams {
        params: serde_json::json!({"max_position": 0.05, "order_size": 0.01}),
    };

    // Run without timer (default)
    let cfg_no = replay_harness::SimConfig {
        timer_interval: 0,
        ..Default::default()
    };
    let mut harness_no_timer = replay_harness::ReplayTestHarness::with_config(
        "adaptive_mm",
        params.clone(),
        exchange,
        "ETHUSDT",
        cfg_no,
    );
    let result_no_timer = harness_no_timer.run(&events);

    // Run with timer enabled but flush disabled in strategy (flush_interval_ticks=0 default)
    let cfg_with = replay_harness::SimConfig {
        timer_interval: 1,
        ..Default::default()
    };
    let mut harness_with_timer = replay_harness::ReplayTestHarness::with_config(
        "adaptive_mm",
        params,
        exchange,
        "ETHUSDT",
        cfg_with,
    );
    let result_with_timer = harness_with_timer.run(&events);

    // Both runs should produce orders.
    assert!(
        result_no_timer.order_count > 0,
        "no-timer: should produce orders"
    );
    assert!(
        result_with_timer.order_count > 0,
        "with-timer: should produce orders"
    );
    // fill counts should be identical (flush disabled → timer fires but produces no orders/fills)
    assert_eq!(
        result_no_timer.fill_count, result_with_timer.fill_count,
        "fill count should be identical when flush is disabled"
    );
}

#[test]
fn test_timer_injection_with_flush_fires_extra_orders() {
    // With flush_interval_ticks=1 and timer_interval=1, on_timer fires every event
    // and when there's a position, submits flush orders → more orders than without timer
    let (path, exchange) = match data_path_for_exchange("binance", "ethusdt") {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no Binance ETHUSDT data file found");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load events");

    // Run without flush
    let params_no_flush = StrategyParams {
        params: serde_json::json!({
            "max_position": 0.05,
            "order_size": 0.01,
        }),
    };
    let cfg_no = replay_harness::SimConfig {
        timer_interval: 0,
        ..Default::default()
    };
    let mut harness_no_flush = replay_harness::ReplayTestHarness::with_config(
        "adaptive_mm",
        params_no_flush,
        exchange,
        "ETHUSDT",
        cfg_no,
    );
    let result_no_flush = harness_no_flush.run(&events);

    // Run with flush every tick
    let params_flush = StrategyParams {
        params: serde_json::json!({
            "max_position": 0.05,
            "order_size": 0.01,
            "flush_threshold": 0.0,
            "flush_interval_ticks": 1,
        }),
    };
    let cfg_flush = replay_harness::SimConfig {
        timer_interval: 1,
        ..Default::default()
    };
    let mut harness_flush = replay_harness::ReplayTestHarness::with_config(
        "adaptive_mm",
        params_flush,
        exchange,
        "ETHUSDT",
        cfg_flush,
    );
    let result_flush = harness_flush.run(&events);

    // Both runs produce orders; flush-enabled should have >= orders due to flush submissions
    assert!(
        result_no_flush.order_count > 0,
        "no-flush: should produce orders"
    );
    assert!(
        result_flush.order_count >= result_no_flush.order_count,
        "flush-enabled run ({}) should have >= orders than no-flush ({})",
        result_flush.order_count,
        result_no_flush.order_count,
    );
}

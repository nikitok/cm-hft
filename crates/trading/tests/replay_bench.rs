//! Detailed replay benchmark — runs strategies on recorded data and prints stats.

mod replay_harness;

use std::path::Path;

use cm_core::types::Exchange;
use cm_strategy::traits::StrategyParams;
use replay_harness::{load_events, ReplayTestHarness};

fn data_path(symbol: &str) -> Option<String> {
    let durations = ["30m", "5m", "3m", "1h"];
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

fn run_and_report(
    strategy_name: &str,
    params: &StrategyParams,
    symbol: &str,
    path: &str,
) {
    let events = load_events(path).expect("failed to load");
    let book_count = events
        .iter()
        .filter(|(_, e)| matches!(e, replay_harness::ReplayEvent::Book(_)))
        .count();
    let trade_count = events
        .iter()
        .filter(|(_, e)| matches!(e, replay_harness::ReplayEvent::Trade(_)))
        .count();

    let start = std::time::Instant::now();
    let mut harness = ReplayTestHarness::new(strategy_name, params.clone(), Exchange::Bybit, symbol);
    let result = harness.run(&events);
    let elapsed = start.elapsed();

    // PnL stats
    let pnl_min = result.pnl_series.iter().cloned().fold(f64::INFINITY, f64::min);
    let pnl_max = result.pnl_series.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let pnl_final = result.pnl_series.last().copied().unwrap_or(0.0);

    // Drawdown
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0_f64;
    for &pnl in &result.pnl_series {
        if pnl > peak {
            peak = pnl;
        }
        let dd = peak - pnl;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│  {} / {}",  strategy_name, symbol);
    println!("├─────────────────────────────────────────────────────────┤");
    println!("│  Input events:  {} book + {} trade = {} total", book_count, trade_count, events.len());
    println!("│  Replay time:   {:?}", elapsed);
    println!("├─────────────────────────────────────────────────────────┤");
    println!("│  Orders placed: {}", result.order_count);
    println!("│  Fills:         {}", result.fill_count);
    println!("│  Max position:  {:.6}", result.max_position);
    println!("├─────────────────────────────────────────────────────────┤");
    println!("│  Peak notional: ${:.2}  (max |pos| × price)", result.peak_notional);
    println!("│  Margin 1x:     ${:.2}", result.peak_margin_1x);
    println!("│  Margin 5x:     ${:.2}", result.peak_notional / 5.0);
    println!("│  Margin 10x:    ${:.2}", result.peak_notional / 10.0);
    println!("├─────────────────────────────────────────────────────────┤");
    println!("│  Realized PnL:  ${:.4}", result.total_pnl);
    println!("│  Final M2M PnL: ${:.4}", pnl_final);
    println!("│  PnL range:     ${:.4} .. ${:.4}", pnl_min, pnl_max);
    println!("│  Max drawdown:  ${:.4}", max_dd);
    if result.peak_margin_1x > 0.0 {
        println!("│  ROC (1x):      {:.2}%", result.total_pnl / result.peak_margin_1x * 100.0);
        println!("│  ROC (10x):     {:.2}%", result.total_pnl / (result.peak_notional / 10.0) * 100.0);
    }
    println!("└─────────────────────────────────────────────────────────┘");
    println!();
}

#[test]
fn bench_all_strategies() {
    let symbols = ["btcusdt", "ethusdt"];
    let configs: Vec<(&str, &str, StrategyParams)> = vec![
        (
            "market_making",
            "default",
            StrategyParams { params: serde_json::json!({}) },
        ),
        (
            "market_making",
            "tight spread",
            StrategyParams { params: serde_json::json!({"spread_bps": 5.0}) },
        ),
        (
            "market_making",
            "wide spread",
            StrategyParams { params: serde_json::json!({"spread_bps": 20.0}) },
        ),
        (
            "market_making",
            "small size",
            StrategyParams { params: serde_json::json!({"order_size": 0.005}) },
        ),
        (
            "market_making",
            "large size",
            StrategyParams { params: serde_json::json!({"order_size": 0.05}) },
        ),
        // ── Adaptive MM variants ──
        (
            "adaptive_mm",
            "default",
            StrategyParams { params: serde_json::json!({}) },
        ),
        (
            "adaptive_mm",
            "aggressive skew",
            StrategyParams { params: serde_json::json!({"skew_intensity": 4.0, "max_position": 0.005}) },
        ),
        (
            "adaptive_mm",
            "wide spread",
            StrategyParams { params: serde_json::json!({"base_spread_bps": 15.0, "min_spread_bps": 5.0}) },
        ),
    ];

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  REPLAY STRATEGY BENCHMARK — real Bybit data");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    for sym in &symbols {
        let path = match data_path(sym) {
            Some(p) => p,
            None => {
                println!("SKIP: no data for {}", sym.to_uppercase());
                continue;
            }
        };

        for (strategy_name, label, params) in &configs {
            println!(">>> {} [{}]", strategy_name, label);
            run_and_report(strategy_name, params, &sym.to_uppercase(), &path);
        }
    }
}

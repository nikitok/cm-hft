//! Detailed replay benchmark — runs strategies on recorded data and prints stats.

mod replay_harness;

use std::path::Path;

use cm_core::types::Exchange;
use cm_strategy::traits::StrategyParams;
use replay_harness::{find_series_files, load_events, load_events_multi, ReplayTestHarness, SimConfig};

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
    run_and_report_with_config(strategy_name, params, symbol, path, SimConfig::default());
}

fn run_and_report_with_config(
    strategy_name: &str,
    params: &StrategyParams,
    symbol: &str,
    path: &str,
    sim_config: SimConfig,
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
    let mut harness = ReplayTestHarness::with_config(strategy_name, params.clone(), Exchange::Bybit, symbol, sim_config);
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
    println!("│  Fee total:     ${:.4}", result.fee_total);
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
            "high risk aversion",
            StrategyParams { params: serde_json::json!({"risk_aversion": 1.0, "max_position": 0.005}) },
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

#[test]
fn bench_improvement_stages() {
    let symbols = ["btcusdt", "ethusdt"];

    // Stage definitions: (name, SimConfig, params JSON)
    // To isolate each improvement, earlier stages disable features via params:
    // - risk_aversion=0.0001 → near-zero A-S effect (falls back to base_spread)
    // - vpin_factor=0.0 → VPIN disabled
    // - size_decay_power=0.0 → flat sizing (no asymmetric decay, (1-r)^0 = 1)
    //   reduce_boost=0.0 → no reducing side boost
    let stages: Vec<(&str, SimConfig, serde_json::Value)> = vec![
        (
            "Baseline",
            SimConfig { maker_fee_bps: 0.0, strict_crossing: false },
            serde_json::json!({
                "risk_aversion": 0.0001,
                "vpin_factor": 0.0,
                "size_decay_power": 0.0,
                "reduce_boost": 0.0
            }),
        ),
        (
            "Realistic Sim",
            SimConfig::default(),
            serde_json::json!({
                "risk_aversion": 0.0001,
                "vpin_factor": 0.0,
                "size_decay_power": 0.0,
                "reduce_boost": 0.0
            }),
        ),
        (
            "A-S Pricing",
            SimConfig::default(),
            serde_json::json!({
                "risk_aversion": 0.3,
                "fill_intensity": 1.5,
                "time_horizon": 1.0,
                "vpin_factor": 0.0,
                "size_decay_power": 0.0,
                "reduce_boost": 0.0
            }),
        ),
        (
            "+ VPIN",
            SimConfig::default(),
            serde_json::json!({
                "risk_aversion": 0.3,
                "fill_intensity": 1.5,
                "time_horizon": 1.0,
                "vpin_factor": 2.0,
                "vpin_bucket_size": 50000.0,
                "vpin_n_buckets": 20,
                "size_decay_power": 0.0,
                "reduce_boost": 0.0
            }),
        ),
        (
            "Full Stack",
            SimConfig::default(),
            serde_json::json!({
                "risk_aversion": 0.3,
                "fill_intensity": 1.5,
                "time_horizon": 1.0,
                "vpin_factor": 2.0,
                "vpin_bucket_size": 50000.0,
                "vpin_n_buckets": 20,
                "size_decay_power": 2.0,
                "reduce_boost": 0.5
            }),
        ),
    ];

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════════════════════");
    println!("  IMPROVEMENT STAGES — adaptive_mm on real Bybit data");
    println!("═══════════════════════════════════════════════════════════════════════════════════════════════");

    for sym in &symbols {
        let path = match data_path(sym) {
            Some(p) => p,
            None => {
                println!("SKIP: no data for {}", sym.to_uppercase());
                continue;
            }
        };

        let events = load_events(&path).expect("failed to load");

        println!();
        println!("  {} ({} events)", sym.to_uppercase(), events.len());
        println!("  {:-<95}", "");
        println!(
            "  {:>2} | {:<15} | {:>6} | {:>8} | {:>12} | {:>10} | {:>12} | {:>10}",
            "#", "Stage", "Fills", "Max Pos", "Realized $", "Max DD $", "Peak Not $", "Fee $"
        );
        println!("  {:-<95}", "");

        for (i, (name, sim_config, params_json)) in stages.iter().enumerate() {
            let params = StrategyParams { params: params_json.clone() };
            let mut harness = ReplayTestHarness::with_config(
                "adaptive_mm",
                params,
                Exchange::Bybit,
                &sym.to_uppercase(),
                sim_config.clone(),
            );
            let result = harness.run(&events);

            // Compute max drawdown
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

            println!(
                "  {:>2} | {:<15} | {:>6} | {:>8.6} | {:>12.4} | {:>10.4} | {:>12.2} | {:>10.4}",
                i, name, result.fill_count, result.max_position, result.total_pnl,
                max_dd, result.peak_notional, result.fee_total,
            );
        }

        println!("  {:-<95}", "");
    }
    println!();
}

#[test]
fn bench_diagnostic() {
    let path = match data_path("btcusdt") {
        Some(p) => p,
        None => {
            println!("SKIP: no BTCUSDT data");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load");

    // Check price trend in the data
    let mut first_mid: Option<f64> = None;
    let mut last_mid: Option<f64> = None;
    let mut high = f64::NEG_INFINITY;
    let mut low = f64::INFINITY;
    for (_, event) in &events {
        if let replay_harness::ReplayEvent::Book(update) = event {
            if let (Some(&(bid_p, _)), Some(&(ask_p, _))) = (update.bids.first(), update.asks.first()) {
                let mid = (bid_p.to_f64() + ask_p.to_f64()) / 2.0;
                if first_mid.is_none() {
                    first_mid = Some(mid);
                }
                last_mid = Some(mid);
                if mid > high { high = mid; }
                if mid < low { low = mid; }
            }
        }
    }

    let first = first_mid.unwrap_or(0.0);
    let last = last_mid.unwrap_or(0.0);
    let range_bps = (high - low) / first * 10_000.0;
    let drift_bps = (last - first) / first * 10_000.0;

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  DATA DIAGNOSTIC — BTCUSDT");
    println!("═══════════════════════════════════════════════════════════");
    println!("  First mid:  ${:.2}", first);
    println!("  Last mid:   ${:.2}", last);
    println!("  High:       ${:.2}", high);
    println!("  Low:        ${:.2}", low);
    println!("  Range:      {:.1} bps ({:.2}%)", range_bps, range_bps / 100.0);
    println!("  Net drift:  {:.1} bps ({:.2}%)", drift_bps, drift_bps / 100.0);
    println!();

    // Run with Full Stack params and track per-fill stats
    let params = StrategyParams {
        params: serde_json::json!({
            "risk_aversion": 0.3,
            "fill_intensity": 1.5,
            "time_horizon": 1.0,
            "vpin_factor": 2.0,
            "size_decay_power": 2.0,
            "reduce_boost": 0.5
        }),
    };

    let mut harness = ReplayTestHarness::with_config(
        "adaptive_mm", params, Exchange::Bybit, "BTCUSDT", SimConfig::default(),
    );
    let result = harness.run(&events);

    let pnl_final = result.pnl_series.last().copied().unwrap_or(0.0);
    let unrealized = pnl_final - result.total_pnl;
    let pnl_per_fill = if result.fill_count > 0 {
        result.total_pnl / result.fill_count as f64
    } else {
        0.0
    };

    println!("  FULL STACK RESULTS:");
    println!("  Fills:         {}", result.fill_count);
    println!("  Realized PnL:  ${:.4}", result.total_pnl);
    println!("  Unrealized:    ${:.4}", unrealized);
    println!("  Final M2M:     ${:.4}", pnl_final);
    println!("  PnL per fill:  ${:.6}", pnl_per_fill);
    println!("  Fee total:     ${:.4} (rebate)", result.fee_total);
    println!("  PnL ex-fees:   ${:.4}", result.total_pnl - result.fee_total);
    println!();
}

#[test]
fn bench_param_sweep() {
    let path = match data_path("btcusdt") {
        Some(p) => p,
        None => {
            println!("SKIP: no BTCUSDT data for param sweep");
            return;
        }
    };

    let events = load_events(&path).expect("failed to load");

    let risk_aversions = [0.1, 0.3, 0.5, 1.0];
    let fill_intensities = [0.5, 1.5, 3.0];
    let vpin_factors = [0.0, 1.0, 2.0, 4.0];
    let decay_powers = [0.0, 1.0, 2.0];
    let time_horizons = [0.5, 1.0, 2.0, 5.0];

    struct SweepResult {
        gamma: f64,
        kappa: f64,
        vpin_f: f64,
        decay: f64,
        tau: f64,
        pnl: f64,
        fills: usize,
        max_dd: f64,
        max_pos: f64,
    }

    let mut results = Vec::new();

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  PARAMETER SWEEP — adaptive_mm on BTCUSDT");
    println!("  {} combinations", risk_aversions.len() * fill_intensities.len()
        * vpin_factors.len() * decay_powers.len() * time_horizons.len());
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    for &gamma in &risk_aversions {
        for &kappa in &fill_intensities {
            for &vpin_f in &vpin_factors {
                for &decay in &decay_powers {
                    for &tau in &time_horizons {
                        let params = StrategyParams {
                            params: serde_json::json!({
                                "risk_aversion": gamma,
                                "fill_intensity": kappa,
                                "time_horizon": tau,
                                "vpin_factor": vpin_f,
                                "vpin_bucket_size": 50000.0,
                                "vpin_n_buckets": 20,
                                "size_decay_power": decay,
                                "reduce_boost": 0.5,
                            }),
                        };

                        let mut harness = ReplayTestHarness::with_config(
                            "adaptive_mm",
                            params,
                            Exchange::Bybit,
                            "BTCUSDT",
                            SimConfig::default(),
                        );
                        let result = harness.run(&events);

                        let mut peak = f64::NEG_INFINITY;
                        let mut max_dd = 0.0_f64;
                        for &pnl in &result.pnl_series {
                            if pnl > peak { peak = pnl; }
                            let dd = peak - pnl;
                            if dd > max_dd { max_dd = dd; }
                        }

                        results.push(SweepResult {
                            gamma,
                            kappa,
                            vpin_f,
                            decay,
                            tau,
                            pnl: result.total_pnl,
                            fills: result.fill_count,
                            max_dd,
                            max_pos: result.max_position,
                        });
                    }
                }
            }
        }
    }

    // Sort by PnL descending.
    results.sort_by(|a, b| b.pnl.partial_cmp(&a.pnl).unwrap_or(std::cmp::Ordering::Equal));

    // Print top 20 and bottom 5.
    println!(
        "  {:>4} | {:>5} | {:>5} | {:>5} | {:>5} | {:>6} | {:>6} | {:>12} | {:>10} | {:>8}",
        "Rank", "γ", "κ", "VPIN", "Decay", "τ", "Fills", "Realized $", "Max DD $", "Max Pos"
    );
    println!("  {:-<105}", "");

    let top_n = 20.min(results.len());
    for (i, r) in results.iter().take(top_n).enumerate() {
        println!(
            "  {:>4} | {:>5.2} | {:>5.1} | {:>5.1} | {:>5.1} | {:>6.0} | {:>6} | {:>12.4} | {:>10.4} | {:>8.6}",
            i + 1, r.gamma, r.kappa, r.vpin_f, r.decay, r.tau, r.fills, r.pnl, r.max_dd, r.max_pos,
        );
    }

    if results.len() > top_n {
        println!("  {:>4}   ...", "");
        let bottom_start = results.len().saturating_sub(5);
        for (i, r) in results.iter().skip(bottom_start).enumerate() {
            println!(
                "  {:>4} | {:>5.2} | {:>5.1} | {:>5.1} | {:>5.1} | {:>6.0} | {:>6} | {:>12.4} | {:>10.4} | {:>8.6}",
                bottom_start + i + 1, r.gamma, r.kappa, r.vpin_f, r.decay, r.tau, r.fills, r.pnl, r.max_dd, r.max_pos,
            );
        }
    }

    println!("  {:-<105}", "");
    println!("  Total configs: {}", results.len());
    println!();
}

/// Helper: load series events for a symbol, returns None if no files found.
fn load_series(sym: &str) -> Option<Vec<(u64, replay_harness::ReplayEvent)>> {
    let data_dirs = ["testdata", "../../testdata"];
    for dir in &data_dirs {
        let files = find_series_files(dir, sym);
        if !files.is_empty() {
            let events = load_events_multi(&files).expect("failed to load series");
            return Some(events);
        }
    }
    None
}

/// Helper: compute max drawdown from PnL series.
fn max_drawdown(pnl_series: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0_f64;
    for &pnl in pnl_series {
        if pnl > peak { peak = pnl; }
        let dd = peak - pnl;
        if dd > max_dd { max_dd = dd; }
    }
    max_dd
}

/// Helper: run a single strategy on events with given sim config.
fn run_strategy_on_events(
    strategy_name: &str,
    params: &StrategyParams,
    symbol: &str,
    events: &[(u64, replay_harness::ReplayEvent)],
    sim_config: SimConfig,
) -> replay_harness::ReplayResult {
    let mut harness = ReplayTestHarness::with_config(
        strategy_name,
        params.clone(),
        Exchange::Bybit,
        symbol,
        sim_config,
    );
    harness.run(events)
}

/// Run all strategies on 8h series data for each pair.
/// Requires timestamped files from `scripts/record-series.sh`.
#[test]
fn bench_series() {
    println!();
    println!("═════════════════════════════════════════════════════════════════════════════════════════════════════════════");
    println!("  8-HOUR SERIES BENCHMARK — all strategies on recorded Bybit data");
    println!("═════════════════════════════════════════════════════════════════════════════════════════════════════════════");

    let configs: Vec<(&str, &str, StrategyParams, SimConfig)> = vec![
        // ── Simple MM variants ──
        (
            "market_making", "simple_mm default",
            StrategyParams { params: serde_json::json!({}) },
            SimConfig::default(),
        ),
        (
            "market_making", "simple_mm tight",
            StrategyParams { params: serde_json::json!({"spread_bps": 5.0}) },
            SimConfig::default(),
        ),
        (
            "market_making", "simple_mm wide",
            StrategyParams { params: serde_json::json!({"spread_bps": 20.0}) },
            SimConfig::default(),
        ),
        // ── Adaptive MM improvement stages ──
        (
            "adaptive_mm", "baseline (no features)",
            StrategyParams { params: serde_json::json!({
                "risk_aversion": 0.0001,
                "vpin_factor": 0.0,
                "size_decay_power": 0.0,
                "reduce_boost": 0.0
            })},
            SimConfig { maker_fee_bps: 0.0, strict_crossing: false },
        ),
        (
            "adaptive_mm", "realistic sim only",
            StrategyParams { params: serde_json::json!({
                "risk_aversion": 0.0001,
                "vpin_factor": 0.0,
                "size_decay_power": 0.0,
                "reduce_boost": 0.0
            })},
            SimConfig::default(),
        ),
        (
            "adaptive_mm", "+ A-S pricing",
            StrategyParams { params: serde_json::json!({
                "risk_aversion": 0.3,
                "fill_intensity": 1.5,
                "time_horizon": 1.0,
                "vpin_factor": 0.0,
                "size_decay_power": 0.0,
                "reduce_boost": 0.0
            })},
            SimConfig::default(),
        ),
        (
            "adaptive_mm", "+ A-S + VPIN",
            StrategyParams { params: serde_json::json!({
                "risk_aversion": 0.3,
                "fill_intensity": 1.5,
                "time_horizon": 1.0,
                "vpin_factor": 2.0,
                "vpin_bucket_size": 50000.0,
                "vpin_n_buckets": 20,
                "size_decay_power": 0.0,
                "reduce_boost": 0.0
            })},
            SimConfig::default(),
        ),
        (
            "adaptive_mm", "FULL STACK",
            StrategyParams { params: serde_json::json!({
                "risk_aversion": 0.3,
                "fill_intensity": 1.5,
                "time_horizon": 1.0,
                "vpin_factor": 2.0,
                "vpin_bucket_size": 50000.0,
                "vpin_n_buckets": 20,
                "size_decay_power": 2.0,
                "reduce_boost": 0.5
            })},
            SimConfig::default(),
        ),
        // ── Tuned variants ──
        (
            "adaptive_mm", "tight: γ=0.1 κ=3 τ=0.5",
            StrategyParams { params: serde_json::json!({
                "risk_aversion": 0.1,
                "fill_intensity": 3.0,
                "time_horizon": 0.5,
                "vpin_factor": 2.0,
                "vpin_bucket_size": 50000.0,
                "vpin_n_buckets": 20,
                "size_decay_power": 2.0,
                "reduce_boost": 0.5
            })},
            SimConfig::default(),
        ),
        (
            "adaptive_mm", "wide: γ=0.5 κ=1.5 τ=2",
            StrategyParams { params: serde_json::json!({
                "risk_aversion": 0.5,
                "fill_intensity": 1.5,
                "time_horizon": 2.0,
                "vpin_factor": 2.0,
                "vpin_bucket_size": 50000.0,
                "vpin_n_buckets": 20,
                "size_decay_power": 2.0,
                "reduce_boost": 0.5
            })},
            SimConfig::default(),
        ),
        (
            "adaptive_mm", "aggr: γ=0.1 κ=3 τ=1",
            StrategyParams { params: serde_json::json!({
                "risk_aversion": 0.1,
                "fill_intensity": 3.0,
                "time_horizon": 1.0,
                "vpin_factor": 1.0,
                "vpin_bucket_size": 50000.0,
                "vpin_n_buckets": 20,
                "size_decay_power": 1.0,
                "reduce_boost": 0.5
            })},
            SimConfig::default(),
        ),
        (
            "adaptive_mm", "conserv: γ=1.0 κ=1.5 τ=2",
            StrategyParams { params: serde_json::json!({
                "risk_aversion": 1.0,
                "fill_intensity": 1.5,
                "time_horizon": 2.0,
                "vpin_factor": 2.0,
                "vpin_bucket_size": 50000.0,
                "vpin_n_buckets": 20,
                "size_decay_power": 2.0,
                "reduce_boost": 0.5,
                "max_position": 0.05
            })},
            SimConfig::default(),
        ),
    ];

    for sym in &["btcusdt", "ethusdt"] {
        let events = match load_series(sym) {
            Some(e) => e,
            None => {
                println!("\n  SKIP: no series data for {}\n", sym.to_uppercase());
                continue;
            }
        };

        let book_count = events.iter().filter(|(_, e)| matches!(e, replay_harness::ReplayEvent::Book(_))).count();
        let trade_count = events.iter().filter(|(_, e)| matches!(e, replay_harness::ReplayEvent::Trade(_))).count();

        // Data diagnostic: price range
        let mut first_mid: Option<f64> = None;
        let mut last_mid: Option<f64> = None;
        let mut high = f64::NEG_INFINITY;
        let mut low = f64::INFINITY;
        for (_, event) in &events {
            if let replay_harness::ReplayEvent::Book(update) = event {
                if let (Some(&(bid_p, _)), Some(&(ask_p, _))) = (update.bids.first(), update.asks.first()) {
                    let mid = (bid_p.to_f64() + ask_p.to_f64()) / 2.0;
                    if first_mid.is_none() { first_mid = Some(mid); }
                    last_mid = Some(mid);
                    if mid > high { high = mid; }
                    if mid < low { low = mid; }
                }
            }
        }
        let first = first_mid.unwrap_or(0.0);
        let last = last_mid.unwrap_or(0.0);
        let range_bps = if first > 0.0 { (high - low) / first * 10_000.0 } else { 0.0 };
        let drift_bps = if first > 0.0 { (last - first) / first * 10_000.0 } else { 0.0 };

        println!();
        println!("  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐");
        println!("  │  {} — 8h series: {} book + {} trade = {} events", sym.to_uppercase(), book_count, trade_count, events.len());
        println!("  │  Price: ${:.2} → ${:.2}  range={:.0}bps  drift={:+.0}bps ({:+.2}%)",
            first, last, range_bps, drift_bps, drift_bps / 100.0);
        println!("  ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤");
        println!(
            "  │ {:>2} | {:<28} | {:>6} | {:>8} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8}",
            "#", "Strategy", "Fills", "Max Pos", "Realized$", "M2M PnL$", "Max DD$", "Peak Not$", "Fee$", "$/fill"
        );
        println!("  │ {:-<123}", "");

        for (i, (strat_name, label, params, sim_config)) in configs.iter().enumerate() {
            let start = std::time::Instant::now();
            let result = run_strategy_on_events(
                strat_name, params, &sym.to_uppercase(), &events, sim_config.clone(),
            );
            let elapsed = start.elapsed();

            let pnl_final = result.pnl_series.last().copied().unwrap_or(0.0);
            let dd = max_drawdown(&result.pnl_series);
            let pnl_per_fill = if result.fill_count > 0 {
                result.total_pnl / result.fill_count as f64
            } else {
                0.0
            };

            println!(
                "  │ {:>2} | {:<28} | {:>6} | {:>8.5} | {:>10.2} | {:>10.2} | {:>10.2} | {:>10.2} | {:>10.2} | {:>8.4}  ({:.1?})",
                i, label, result.fill_count, result.max_position,
                result.total_pnl, pnl_final, dd, result.peak_notional,
                result.fee_total, pnl_per_fill, elapsed,
            );
        }

        println!("  │ {:-<123}", "");
        println!("  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘");
    }
    println!();
}

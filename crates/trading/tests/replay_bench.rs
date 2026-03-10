//! Detailed replay benchmark — runs strategies on recorded data and prints stats.

mod replay_harness;

use std::path::Path;

use cm_core::types::Exchange;
use cm_strategy::traits::StrategyParams;
use replay_harness::{
    find_series_files, load_events, load_events_multi, ReplayTestHarness, SimConfig,
};

/// Type alias for symbol data: (exchange, symbol, events)
type SymbolData = Vec<(Exchange, String, Vec<(u64, replay_harness::ReplayEvent)>)>;

/// Helper: find data file for a specific exchange and symbol.
/// Returns (path, exchange) if found.
fn data_path_for_exchange(exchange: &str, symbol: &str) -> Option<(String, Exchange)> {
    let durations = ["30m", "5m", "3m", "1h"];
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

fn data_path(symbol: &str) -> Option<(String, Exchange)> {
    // Try Bybit first (most test data), then Binance
    data_path_for_exchange("bybit", symbol).or_else(|| data_path_for_exchange("binance", symbol))
}

fn run_and_report(
    strategy_name: &str,
    params: &StrategyParams,
    exchange: Exchange,
    symbol: &str,
    path: &str,
) {
    run_and_report_with_config(
        strategy_name,
        params,
        exchange,
        symbol,
        path,
        SimConfig::default(),
    );
}

fn run_and_report_with_config(
    strategy_name: &str,
    params: &StrategyParams,
    exchange: Exchange,
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
    let mut harness =
        ReplayTestHarness::with_config(strategy_name, params.clone(), exchange, symbol, sim_config);
    let result = harness.run(&events);
    let elapsed = start.elapsed();

    // PnL stats
    let pnl_min = result
        .pnl_series
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let pnl_max = result
        .pnl_series
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
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
    println!("│  {} / {}", strategy_name, symbol);
    println!("├─────────────────────────────────────────────────────────┤");
    println!(
        "│  Input events:  {} book + {} trade = {} total",
        book_count,
        trade_count,
        events.len()
    );
    println!("│  Replay time:   {:?}", elapsed);
    println!("├─────────────────────────────────────────────────────────┤");
    println!("│  Orders placed: {}", result.order_count);
    println!("│  Fills:         {}", result.fill_count);
    println!("│  Max position:  {:.6}", result.max_position);
    println!("├─────────────────────────────────────────────────────────┤");
    println!(
        "│  Peak notional: ${:.2}  (max |pos| × price)",
        result.peak_notional
    );
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
        println!(
            "│  ROC (1x):      {:.2}%",
            result.total_pnl / result.peak_margin_1x * 100.0
        );
        println!(
            "│  ROC (10x):     {:.2}%",
            result.total_pnl / (result.peak_notional / 10.0) * 100.0
        );
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
            StrategyParams {
                params: serde_json::json!({}),
            },
        ),
        (
            "market_making",
            "tight spread",
            StrategyParams {
                params: serde_json::json!({"spread_bps": 5.0}),
            },
        ),
        (
            "market_making",
            "wide spread",
            StrategyParams {
                params: serde_json::json!({"spread_bps": 20.0}),
            },
        ),
        (
            "market_making",
            "small size",
            StrategyParams {
                params: serde_json::json!({"order_size": 0.005}),
            },
        ),
        (
            "market_making",
            "large size",
            StrategyParams {
                params: serde_json::json!({"order_size": 0.05}),
            },
        ),
        // ── Adaptive MM variants ──
        (
            "adaptive_mm",
            "default",
            StrategyParams {
                params: serde_json::json!({}),
            },
        ),
        (
            "adaptive_mm",
            "high risk aversion",
            StrategyParams {
                params: serde_json::json!({"risk_aversion": 1.0, "max_position": 0.005}),
            },
        ),
        (
            "adaptive_mm",
            "wide spread",
            StrategyParams {
                params: serde_json::json!({"base_spread_bps": 15.0, "min_spread_bps": 5.0}),
            },
        ),
    ];

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  REPLAY STRATEGY BENCHMARK — real market data");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    for sym in &symbols {
        let (path, exchange) = match data_path(sym) {
            Some(p) => p,
            None => {
                println!("SKIP: no data for {}", sym.to_uppercase());
                continue;
            }
        };
        let exchange_name = match exchange {
            Exchange::Binance => "Binance",
            Exchange::Bybit => "Bybit",
        };

        for (strategy_name, label, params) in &configs {
            println!(">>> {} [{}] on {}", strategy_name, label, exchange_name);
            run_and_report(strategy_name, params, exchange, &sym.to_uppercase(), &path);
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
            SimConfig {
                maker_fee_bps: 0.0,
                strict_crossing: false,
                ..SimConfig::legacy()
            },
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
    println!("  IMPROVEMENT STAGES — adaptive_mm on real market data");
    println!("═══════════════════════════════════════════════════════════════════════════════════════════════");

    for sym in &symbols {
        let (path, exchange) = match data_path(sym) {
            Some(p) => p,
            None => {
                println!("SKIP: no data for {}", sym.to_uppercase());
                continue;
            }
        };
        let exchange_name = match exchange {
            Exchange::Binance => "Binance",
            Exchange::Bybit => "Bybit",
        };

        let events = load_events(&path).expect("failed to load");

        println!();
        println!(
            "  {} / {} ({} events)",
            sym.to_uppercase(),
            exchange_name,
            events.len()
        );
        println!("  {:-<95}", "");
        println!(
            "  {:>2} | {:<15} | {:>6} | {:>8} | {:>12} | {:>10} | {:>12} | {:>10}",
            "#", "Stage", "Fills", "Max Pos", "Realized $", "Max DD $", "Peak Not $", "Fee $"
        );
        println!("  {:-<95}", "");

        for (i, (name, sim_config, params_json)) in stages.iter().enumerate() {
            let params = StrategyParams {
                params: params_json.clone(),
            };
            let mut harness = ReplayTestHarness::with_config(
                "adaptive_mm",
                params,
                exchange,
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
                i,
                name,
                result.fill_count,
                result.max_position,
                result.total_pnl,
                max_dd,
                result.peak_notional,
                result.fee_total,
            );
        }

        println!("  {:-<95}", "");
    }
    println!();
}

#[test]
fn bench_diagnostic() {
    let (path, exchange) = match data_path("btcusdt") {
        Some(p) => p,
        None => {
            println!("SKIP: no BTCUSDT data");
            return;
        }
    };
    let exchange_name = match exchange {
        Exchange::Binance => "Binance",
        Exchange::Bybit => "Bybit",
    };

    let events = load_events(&path).expect("failed to load");

    // Check price trend in the data
    let mut first_mid: Option<f64> = None;
    let mut last_mid: Option<f64> = None;
    let mut high = f64::NEG_INFINITY;
    let mut low = f64::INFINITY;
    for (_, event) in &events {
        if let replay_harness::ReplayEvent::Book(update) = event {
            if let (Some(&(bid_p, _)), Some(&(ask_p, _))) =
                (update.bids.first(), update.asks.first())
            {
                let mid = (bid_p.to_f64() + ask_p.to_f64()) / 2.0;
                if first_mid.is_none() {
                    first_mid = Some(mid);
                }
                last_mid = Some(mid);
                if mid > high {
                    high = mid;
                }
                if mid < low {
                    low = mid;
                }
            }
        }
    }

    let first = first_mid.unwrap_or(0.0);
    let last = last_mid.unwrap_or(0.0);
    let range_bps = (high - low) / first * 10_000.0;
    let drift_bps = (last - first) / first * 10_000.0;

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  DATA DIAGNOSTIC — BTCUSDT / {}", exchange_name);
    println!("═══════════════════════════════════════════════════════════");
    println!("  First mid:  ${:.2}", first);
    println!("  Last mid:   ${:.2}", last);
    println!("  High:       ${:.2}", high);
    println!("  Low:        ${:.2}", low);
    println!(
        "  Range:      {:.1} bps ({:.2}%)",
        range_bps,
        range_bps / 100.0
    );
    println!(
        "  Net drift:  {:.1} bps ({:.2}%)",
        drift_bps,
        drift_bps / 100.0
    );
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
        "adaptive_mm",
        params,
        exchange,
        "BTCUSDT",
        SimConfig::default(),
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
    println!(
        "  PnL ex-fees:   ${:.4}",
        result.total_pnl - result.fee_total
    );
    println!();
}

#[test]
fn bench_param_sweep() {
    let (path, exchange) = match data_path("btcusdt") {
        Some(p) => p,
        None => {
            println!("SKIP: no BTCUSDT data for param sweep");
            return;
        }
    };
    let exchange_name = match exchange {
        Exchange::Binance => "Binance",
        Exchange::Bybit => "Bybit",
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
    println!(
        "  PARAMETER SWEEP — adaptive_mm on BTCUSDT / {}",
        exchange_name
    );
    println!(
        "  {} combinations",
        risk_aversions.len()
            * fill_intensities.len()
            * vpin_factors.len()
            * decay_powers.len()
            * time_horizons.len()
    );
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
                            exchange,
                            "BTCUSDT",
                            SimConfig::default(),
                        );
                        let result = harness.run(&events);

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
    results.sort_by(|a, b| {
        b.pnl
            .partial_cmp(&a.pnl)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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

/// Helper: load series events for a symbol and exchange, returns None if no files found.
fn load_series(exchange: &str, sym: &str) -> Option<Vec<(u64, replay_harness::ReplayEvent)>> {
    let data_dirs = ["testdata", "../../testdata"];
    for dir in &data_dirs {
        let files = find_series_files(dir, exchange, sym);
        if !files.is_empty() {
            let events = load_events_multi(&files).expect("failed to load series");
            return Some(events);
        }
    }
    None
}

/// Discover all (exchange, symbol) pairs with series data in testdata/.
/// Looks for timestamped files matching `{exchange}_{symbol}_YYYY-MM-DD_HH:MM.jsonl.gz`.
fn discover_symbols() -> Vec<(Exchange, String)> {
    let data_dirs = ["testdata", "../../testdata"];
    let mut pairs = Vec::new();
    let re = regex::Regex::new(
        r"^(bybit|binance)_([a-z0-9]+)_\d{4}-\d{2}-\d{2}_\d{2}:\d{2}\.jsonl\.gz$",
    )
    .unwrap();
    for dir in &data_dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if let Some(caps) = re.captures(&name) {
                    let exchange = match &caps[1] {
                        "binance" => Exchange::Binance,
                        "bybit" => Exchange::Bybit,
                        _ => continue,
                    };
                    let symbol = caps[2].to_string();
                    let pair = (exchange, symbol.clone());
                    if !pairs.contains(&pair) {
                        pairs.push(pair);
                    }
                }
            }
        }
    }
    pairs.sort_by(|a, b| {
        let a_ex = match a.0 {
            Exchange::Binance => 0,
            Exchange::Bybit => 1,
        };
        let b_ex = match b.0 {
            Exchange::Binance => 0,
            Exchange::Bybit => 1,
        };
        a_ex.cmp(&b_ex).then_with(|| a.1.cmp(&b.1))
    });
    pairs
}

/// Helper: compute max drawdown from PnL series.
fn max_drawdown(pnl_series: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0_f64;
    for &pnl in pnl_series {
        if pnl > peak {
            peak = pnl;
        }
        let dd = peak - pnl;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

/// Helper: run a single strategy on events with given sim config.
fn run_strategy_on_events(
    strategy_name: &str,
    params: &StrategyParams,
    exchange: Exchange,
    symbol: &str,
    events: &[(u64, replay_harness::ReplayEvent)],
    sim_config: SimConfig,
) -> replay_harness::ReplayResult {
    let mut harness =
        ReplayTestHarness::with_config(strategy_name, params.clone(), exchange, symbol, sim_config);
    harness.run(events)
}

/// Run all strategies on 8h series data for each pair.
/// Requires timestamped files from `scripts/record-series.sh`.
#[test]
fn bench_series() {
    println!();
    println!("═════════════════════════════════════════════════════════════════════════════════════════════════════════════");
    println!("  8-HOUR SERIES BENCHMARK — all strategies on recorded market data");
    println!("═════════════════════════════════════════════════════════════════════════════════════════════════════════════");

    let configs: Vec<(&str, &str, StrategyParams, SimConfig)> = vec![
        // ── Simple MM variants ──
        (
            "market_making",
            "simple_mm default",
            StrategyParams {
                params: serde_json::json!({}),
            },
            SimConfig::default(),
        ),
        (
            "market_making",
            "simple_mm tight",
            StrategyParams {
                params: serde_json::json!({"spread_bps": 5.0}),
            },
            SimConfig::default(),
        ),
        (
            "market_making",
            "simple_mm wide",
            StrategyParams {
                params: serde_json::json!({"spread_bps": 20.0}),
            },
            SimConfig::default(),
        ),
        // ── Adaptive MM improvement stages ──
        (
            "adaptive_mm",
            "baseline (no features)",
            StrategyParams {
                params: serde_json::json!({
                    "risk_aversion": 0.0001,
                    "vpin_factor": 0.0,
                    "size_decay_power": 0.0,
                    "reduce_boost": 0.0
                }),
            },
            SimConfig {
                maker_fee_bps: 0.0,
                strict_crossing: false,
                ..SimConfig::legacy()
            },
        ),
        (
            "adaptive_mm",
            "realistic sim only",
            StrategyParams {
                params: serde_json::json!({
                    "risk_aversion": 0.0001,
                    "vpin_factor": 0.0,
                    "size_decay_power": 0.0,
                    "reduce_boost": 0.0
                }),
            },
            SimConfig::default(),
        ),
        (
            "adaptive_mm",
            "+ A-S pricing",
            StrategyParams {
                params: serde_json::json!({
                    "risk_aversion": 0.3,
                    "fill_intensity": 1.5,
                    "time_horizon": 1.0,
                    "vpin_factor": 0.0,
                    "size_decay_power": 0.0,
                    "reduce_boost": 0.0
                }),
            },
            SimConfig::default(),
        ),
        (
            "adaptive_mm",
            "+ A-S + VPIN",
            StrategyParams {
                params: serde_json::json!({
                    "risk_aversion": 0.3,
                    "fill_intensity": 1.5,
                    "time_horizon": 1.0,
                    "vpin_factor": 2.0,
                    "vpin_bucket_size": 50000.0,
                    "vpin_n_buckets": 20,
                    "size_decay_power": 0.0,
                    "reduce_boost": 0.0
                }),
            },
            SimConfig::default(),
        ),
        (
            "adaptive_mm",
            "FULL STACK (legacy sim)",
            StrategyParams {
                params: serde_json::json!({
                    "risk_aversion": 0.3,
                    "fill_intensity": 1.5,
                    "time_horizon": 1.0,
                    "vpin_factor": 2.0,
                    "vpin_bucket_size": 50000.0,
                    "vpin_n_buckets": 20,
                    "size_decay_power": 2.0,
                    "reduce_boost": 0.5
                }),
            },
            SimConfig::legacy(),
        ),
        (
            "adaptive_mm",
            "FULL STACK (queue sim)",
            StrategyParams {
                params: serde_json::json!({
                    "risk_aversion": 0.3,
                    "fill_intensity": 1.5,
                    "time_horizon": 1.0,
                    "vpin_factor": 2.0,
                    "vpin_bucket_size": 50000.0,
                    "vpin_n_buckets": 20,
                    "size_decay_power": 2.0,
                    "reduce_boost": 0.5
                }),
            },
            SimConfig::default(),
        ),
        // ── ML reference configs ──
        (
            "adaptive_mm",
            "ML width f=1.0 (global)",
            StrategyParams {
                params: serde_json::json!({
                    "risk_aversion": 0.3,
                    "fill_intensity": 1.5,
                    "time_horizon": 1.0,
                    "vpin_factor": 2.0,
                    "vpin_bucket_size": 50000.0,
                    "vpin_n_buckets": 20,
                    "size_decay_power": 2.0,
                    "reduce_boost": 0.5,
                    "ml_factor": 1.0,
                    "ml_threshold": 0.05,
                    "ml_mode": "width",
                    "ml_weights_path": "../../models/mid_predictor.safetensors"
                }),
            },
            SimConfig::default(),
        ),
        (
            "adaptive_mm",
            "ML shift f=0.3 (global)",
            StrategyParams {
                params: serde_json::json!({
                    "risk_aversion": 0.3,
                    "fill_intensity": 1.5,
                    "time_horizon": 1.0,
                    "vpin_factor": 2.0,
                    "vpin_bucket_size": 50000.0,
                    "vpin_n_buckets": 20,
                    "size_decay_power": 2.0,
                    "reduce_boost": 0.5,
                    "ml_factor": 0.3,
                    "ml_threshold": 0.05,
                    "ml_mode": "shift",
                    "ml_weights_path": "../../models/mid_predictor.safetensors"
                }),
            },
            SimConfig::default(),
        ),
        // ── ML per-symbol optimal: width for BTC, shift for ETH ──
        (
            "adaptive_mm",
            "ML per-symbol optimal",
            StrategyParams {
                params: serde_json::json!({
                    "risk_aversion": 0.3,
                    "fill_intensity": 1.5,
                    "time_horizon": 1.0,
                    "vpin_factor": 2.0,
                    "vpin_bucket_size": 50000.0,
                    "vpin_n_buckets": 20,
                    "size_decay_power": 2.0,
                    "reduce_boost": 0.5,
                    "ml_factor": 1.0,
                    "ml_threshold": 0.05,
                    "ml_mode": "width",
                    "ml_weights_path": "../../models/mid_predictor.safetensors",
                    "ml_overrides": {
                        "ETHUSDT": { "ml_mode": "shift", "ml_factor": 0.3, "ml_threshold": 0.05 }
                    }
                }),
            },
            SimConfig::default(),
        ),
        // ── Tuned variants ──
        (
            "adaptive_mm",
            "tight: γ=0.1 κ=3 τ=0.5",
            StrategyParams {
                params: serde_json::json!({
                    "risk_aversion": 0.1,
                    "fill_intensity": 3.0,
                    "time_horizon": 0.5,
                    "vpin_factor": 2.0,
                    "vpin_bucket_size": 50000.0,
                    "vpin_n_buckets": 20,
                    "size_decay_power": 2.0,
                    "reduce_boost": 0.5
                }),
            },
            SimConfig::default(),
        ),
        (
            "adaptive_mm",
            "wide: γ=0.5 κ=1.5 τ=2",
            StrategyParams {
                params: serde_json::json!({
                    "risk_aversion": 0.5,
                    "fill_intensity": 1.5,
                    "time_horizon": 2.0,
                    "vpin_factor": 2.0,
                    "vpin_bucket_size": 50000.0,
                    "vpin_n_buckets": 20,
                    "size_decay_power": 2.0,
                    "reduce_boost": 0.5
                }),
            },
            SimConfig::default(),
        ),
        (
            "adaptive_mm",
            "aggr: γ=0.1 κ=3 τ=1",
            StrategyParams {
                params: serde_json::json!({
                    "risk_aversion": 0.1,
                    "fill_intensity": 3.0,
                    "time_horizon": 1.0,
                    "vpin_factor": 1.0,
                    "vpin_bucket_size": 50000.0,
                    "vpin_n_buckets": 20,
                    "size_decay_power": 1.0,
                    "reduce_boost": 0.5
                }),
            },
            SimConfig::default(),
        ),
        (
            "adaptive_mm",
            "conserv: γ=1.0 κ=1.5 τ=2",
            StrategyParams {
                params: serde_json::json!({
                    "risk_aversion": 1.0,
                    "fill_intensity": 1.5,
                    "time_horizon": 2.0,
                    "vpin_factor": 2.0,
                    "vpin_bucket_size": 50000.0,
                    "vpin_n_buckets": 20,
                    "size_decay_power": 2.0,
                    "reduce_boost": 0.5,
                    "max_position": 0.05
                }),
            },
            SimConfig::default(),
        ),
    ];

    let pairs = discover_symbols();
    if pairs.is_empty() {
        println!("\n  SKIP: no series data found in testdata/\n");
        return;
    }
    println!(
        "  Discovered pairs: {}",
        pairs
            .iter()
            .map(|(exch, sym)| {
                let ex_name = match exch {
                    Exchange::Binance => "Binance",
                    Exchange::Bybit => "Bybit",
                };
                format!("{}/{}", sym.to_uppercase(), ex_name)
            })
            .collect::<Vec<_>>()
            .join(", ")
    );

    for (exchange, sym) in &pairs {
        let exchange_prefix = match exchange {
            Exchange::Binance => "binance",
            Exchange::Bybit => "bybit",
        };
        let exchange_name = match exchange {
            Exchange::Binance => "Binance",
            Exchange::Bybit => "Bybit",
        };
        let events = match load_series(exchange_prefix, sym) {
            Some(e) => e,
            None => {
                println!(
                    "\n  SKIP: no series data for {} / {}\n",
                    sym.to_uppercase(),
                    exchange_name
                );
                continue;
            }
        };

        let book_count = events
            .iter()
            .filter(|(_, e)| matches!(e, replay_harness::ReplayEvent::Book(_)))
            .count();
        let trade_count = events
            .iter()
            .filter(|(_, e)| matches!(e, replay_harness::ReplayEvent::Trade(_)))
            .count();

        // Data diagnostic: price range
        let mut first_mid: Option<f64> = None;
        let mut last_mid: Option<f64> = None;
        let mut high = f64::NEG_INFINITY;
        let mut low = f64::INFINITY;
        for (_, event) in &events {
            if let replay_harness::ReplayEvent::Book(update) = event {
                if let (Some(&(bid_p, _)), Some(&(ask_p, _))) =
                    (update.bids.first(), update.asks.first())
                {
                    let mid = (bid_p.to_f64() + ask_p.to_f64()) / 2.0;
                    if first_mid.is_none() {
                        first_mid = Some(mid);
                    }
                    last_mid = Some(mid);
                    if mid > high {
                        high = mid;
                    }
                    if mid < low {
                        low = mid;
                    }
                }
            }
        }
        let first = first_mid.unwrap_or(0.0);
        let last = last_mid.unwrap_or(0.0);
        let range_bps = if first > 0.0 {
            (high - low) / first * 10_000.0
        } else {
            0.0
        };
        let drift_bps = if first > 0.0 {
            (last - first) / first * 10_000.0
        } else {
            0.0
        };

        println!();
        println!("  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐");
        println!(
            "  │  {} / {} — 8h series: {} book + {} trade = {} events",
            sym.to_uppercase(),
            exchange_name,
            book_count,
            trade_count,
            events.len()
        );
        println!(
            "  │  Price: ${:.2} → ${:.2}  range={:.0}bps  drift={:+.0}bps ({:+.2}%)",
            first,
            last,
            range_bps,
            drift_bps,
            drift_bps / 100.0
        );
        println!("  ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤");
        println!(
            "  │ {:>2} | {:<28} | {:>6} | {:>8} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8}",
            "#", "Strategy", "Fills", "Max Pos", "Realized$", "M2M PnL$", "Max DD$", "Peak Not$", "Fees$", "$/fill"
        );
        println!("  │ {:-<123}", "");

        for (i, (strat_name, label, params, sim_config)) in configs.iter().enumerate() {
            let start = std::time::Instant::now();
            let result = run_strategy_on_events(
                strat_name,
                params,
                *exchange,
                &sym.to_uppercase(),
                &events,
                sim_config.clone(),
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

/// Grid-sweep optimizer using Calmar ratio (PnL / MaxDD) across multiple symbols.
///
/// Sweeps 6 adaptive_mm parameters over 3,840 combinations, evaluates on all
/// discovered symbols, and ranks by average Calmar ratio to find configs that
/// perform well everywhere without blowing up on any single pair.
#[test]
fn bench_optimizer() {
    let pairs = discover_symbols();
    if pairs.is_empty() {
        println!("\n  SKIP: no series data found for optimizer\n");
        return;
    }

    // Load events for each (exchange, symbol) pair upfront.
    let symbol_data: SymbolData = pairs
        .iter()
        .filter_map(|(exch, sym)| {
            let exchange_prefix = match exch {
                Exchange::Binance => "binance",
                Exchange::Bybit => "bybit",
            };
            load_series(exchange_prefix, sym).map(|events| (*exch, sym.clone(), events))
        })
        .collect();
    if symbol_data.is_empty() {
        println!("\n  SKIP: could not load series data\n");
        return;
    }

    // ── Parameter grid (6 dimensions, 3840 combos) ──
    let risk_aversions = [0.1, 0.2, 0.3, 0.5, 1.0];
    let fill_intensities = [0.5, 1.5, 3.0];
    let time_horizons = [0.5, 1.0, 2.0, 5.0];
    let vpin_factors = [0.0, 1.0, 2.0, 4.0];
    let decay_powers = [0.0, 1.0, 2.0, 3.0];
    let reduce_boosts = [0.0, 0.25, 0.5, 1.0];

    // Fixed params (not swept).
    let base_spread_bps = 8.0;
    let min_spread_bps = 2.0;
    let order_size = 0.001;
    let max_position = 0.1;

    let total_combos = risk_aversions.len()
        * fill_intensities.len()
        * time_horizons.len()
        * vpin_factors.len()
        * decay_powers.len()
        * reduce_boosts.len();

    struct SymbolMetrics {
        exchange: Exchange,
        symbol: String,
        pnl: f64,
        max_dd: f64,
        calmar: f64,
        fills: usize,
        max_pos: f64,
    }

    struct OptimizerResult {
        gamma: f64,
        kappa: f64,
        tau: f64,
        vpin_f: f64,
        decay: f64,
        boost: f64,
        per_symbol: Vec<SymbolMetrics>,
        avg_calmar: f64,
        total_pnl: f64,
    }

    fn calmar(pnl: f64, max_dd: f64) -> f64 {
        if max_dd > 0.0 {
            pnl / max_dd
        } else if pnl > 0.0 {
            f64::MAX
        } else {
            0.0
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════════════════════════════════════");
    println!("  GRID OPTIMIZER — adaptive_mm, Calmar ratio objective");
    println!("═══════════════════════════════════════════════════════════════════════════════════════════════════════════════");
    println!(
        "  Pairs: {}",
        symbol_data
            .iter()
            .map(|(exch, s, e)| {
                let ex_name = match exch {
                    Exchange::Binance => "Binance",
                    Exchange::Bybit => "Bybit",
                };
                format!("{}/{} ({} events)", s.to_uppercase(), ex_name, e.len())
            })
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "  Grid: {} combos × {} symbols = {} runs",
        total_combos,
        symbol_data.len(),
        total_combos * symbol_data.len()
    );
    println!(
        "  Fixed: base_spread_bps={}, min_spread_bps={}, order_size={}, max_position={}",
        base_spread_bps, min_spread_bps, order_size, max_position
    );
    println!("═══════════════════════════════════════════════════════════════════════════════════════════════════════════════");

    let sweep_start = std::time::Instant::now();
    let sim_config = SimConfig::default();
    let mut results: Vec<OptimizerResult> = Vec::with_capacity(total_combos);

    let mut combo_idx = 0usize;
    for &gamma in &risk_aversions {
        for &kappa in &fill_intensities {
            for &tau in &time_horizons {
                for &vpin_f in &vpin_factors {
                    for &decay in &decay_powers {
                        for &boost in &reduce_boosts {
                            let params = StrategyParams {
                                params: serde_json::json!({
                                    "risk_aversion": gamma,
                                    "fill_intensity": kappa,
                                    "time_horizon": tau,
                                    "vpin_factor": vpin_f,
                                    "vpin_bucket_size": 50000.0,
                                    "vpin_n_buckets": 20,
                                    "size_decay_power": decay,
                                    "reduce_boost": boost,
                                    "base_spread_bps": base_spread_bps,
                                    "min_spread_bps": min_spread_bps,
                                    "order_size": order_size,
                                    "max_position": max_position,
                                }),
                            };

                            let mut per_symbol = Vec::with_capacity(symbol_data.len());
                            let mut total_pnl = 0.0;
                            let mut calmar_sum = 0.0;

                            for (exchange, sym, events) in &symbol_data {
                                let result = run_strategy_on_events(
                                    "adaptive_mm",
                                    &params,
                                    *exchange,
                                    &sym.to_uppercase(),
                                    events,
                                    sim_config.clone(),
                                );
                                let dd = max_drawdown(&result.pnl_series);
                                let c = calmar(result.total_pnl, dd);
                                total_pnl += result.total_pnl;
                                calmar_sum += c;
                                per_symbol.push(SymbolMetrics {
                                    exchange: *exchange,
                                    symbol: sym.to_uppercase(),
                                    pnl: result.total_pnl,
                                    max_dd: dd,
                                    calmar: c,
                                    fills: result.fill_count,
                                    max_pos: result.max_position,
                                });
                            }

                            let avg_calmar = calmar_sum / symbol_data.len() as f64;
                            results.push(OptimizerResult {
                                gamma,
                                kappa,
                                tau,
                                vpin_f,
                                decay,
                                boost,
                                per_symbol,
                                avg_calmar,
                                total_pnl,
                            });

                            combo_idx += 1;
                            if combo_idx.is_multiple_of(500) {
                                let pct = combo_idx as f64 / total_combos as f64 * 100.0;
                                println!("  ... {}/{} ({:.0}%)", combo_idx, total_combos, pct);
                            }
                        }
                    }
                }
            }
        }
    }

    let sweep_elapsed = sweep_start.elapsed();

    // Sort by avg_calmar descending (treat MAX as very large but finite for sorting).
    results.sort_by(|a, b| {
        let ca = if a.avg_calmar == f64::MAX {
            1e18
        } else {
            a.avg_calmar
        };
        let cb = if b.avg_calmar == f64::MAX {
            1e18
        } else {
            b.avg_calmar
        };
        cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)
    });

    // ── Per-symbol top 10 ──
    for (exchange, sym, _) in &symbol_data {
        let sym_upper = sym.to_uppercase();
        let exchange_name = match exchange {
            Exchange::Binance => "Binance",
            Exchange::Bybit => "Bybit",
        };
        let mut by_symbol: Vec<(usize, f64)> = results
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                r.per_symbol
                    .iter()
                    .find(|m| m.symbol == sym_upper && m.exchange == *exchange)
                    .map(|m| (i, m.calmar))
            })
            .collect();
        by_symbol.sort_by(|a, b| {
            let ca = if a.1 == f64::MAX { 1e18 } else { a.1 };
            let cb = if b.1 == f64::MAX { 1e18 } else { b.1 };
            cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)
        });

        println!();
        println!("  ── {} / {} Top 10 by Calmar ──", sym_upper, exchange_name);
        println!(
            "  {:>4} | {:>5} | {:>5} | {:>5} | {:>5} | {:>5} | {:>5} | {:>10} | {:>10} | {:>10} | {:>6}",
            "Rank", "γ", "κ", "τ", "VPIN", "Decay", "Boost", "PnL $", "Max DD $", "Calmar", "Fills"
        );
        println!("  {:-<110}", "");

        for (rank, &(idx, _)) in by_symbol.iter().take(10).enumerate() {
            let r = &results[idx];
            let m = r
                .per_symbol
                .iter()
                .find(|m| m.symbol == sym_upper && m.exchange == *exchange)
                .unwrap();
            let calmar_str = if m.calmar == f64::MAX {
                "∞".to_string()
            } else {
                format!("{:.4}", m.calmar)
            };
            println!(
                "  {:>4} | {:>5.2} | {:>5.1} | {:>5.1} | {:>5.1} | {:>5.1} | {:>5.2} | {:>10.4} | {:>10.4} | {:>10} | {:>6}",
                rank + 1, r.gamma, r.kappa, r.tau, r.vpin_f, r.decay, r.boost,
                m.pnl, m.max_dd, calmar_str, m.fills,
            );
        }
    }

    // ── Aggregate top 20 with per-symbol breakdown ──
    println!();
    println!("  ══════════════════════════════════════════════════════════════════════════════════════════════════════════");
    println!("  AGGREGATE TOP 20 by average Calmar across all symbols");
    println!("  ══════════════════════════════════════════════════════════════════════════════════════════════════════════");

    let top_n = 20.min(results.len());
    for (rank, r) in results.iter().take(top_n).enumerate() {
        let calmar_str = if r.avg_calmar == f64::MAX {
            "∞".to_string()
        } else {
            format!("{:.4}", r.avg_calmar)
        };
        println!();
        println!(
            "  #{:<3}  γ={:.2} κ={:.1} τ={:.1} vpin={:.1} decay={:.1} boost={:.2}  |  avg_calmar={} total_pnl=${:.4}",
            rank + 1, r.gamma, r.kappa, r.tau, r.vpin_f, r.decay, r.boost,
            calmar_str, r.total_pnl,
        );
        println!(
            "  {:>15} | {:>10} | {:>10} | {:>10} | {:>6} | {:>8}",
            "Pair", "PnL $", "Max DD $", "Calmar", "Fills", "Max Pos"
        );
        for m in &r.per_symbol {
            let exchange_name = match m.exchange {
                Exchange::Binance => "Binance",
                Exchange::Bybit => "Bybit",
            };
            let pair_label = format!("{}/{}", m.symbol, exchange_name);
            let c_str = if m.calmar == f64::MAX {
                "∞".to_string()
            } else {
                format!("{:.4}", m.calmar)
            };
            println!(
                "  {:>15} | {:>10.4} | {:>10.4} | {:>10} | {:>6} | {:>8.6}",
                pair_label, m.pnl, m.max_dd, c_str, m.fills, m.max_pos,
            );
        }
    }

    // ── Bottom 5 (worst configs) ──
    println!();
    println!("  ── BOTTOM 5 (worst configs) ──");
    println!(
        "  {:>4} | {:>5} | {:>5} | {:>5} | {:>5} | {:>5} | {:>5} | {:>10} | {:>10}",
        "Rank", "γ", "κ", "τ", "VPIN", "Decay", "Boost", "Avg Calmar", "Total PnL$"
    );
    println!("  {:-<90}", "");
    let bottom_start = results.len().saturating_sub(5);
    for (i, r) in results.iter().skip(bottom_start).enumerate() {
        let calmar_str = if r.avg_calmar == f64::MAX {
            "∞".to_string()
        } else {
            format!("{:.4}", r.avg_calmar)
        };
        println!(
            "  {:>4} | {:>5.2} | {:>5.1} | {:>5.1} | {:>5.1} | {:>5.1} | {:>5.2} | {:>10} | {:>10.4}",
            bottom_start + i + 1, r.gamma, r.kappa, r.tau, r.vpin_f, r.decay, r.boost,
            calmar_str, r.total_pnl,
        );
    }

    // ── Summary ──
    let best_calmar = results.first().map(|r| r.avg_calmar).unwrap_or(0.0);
    let worst_calmar = results.last().map(|r| r.avg_calmar).unwrap_or(0.0);
    let best_str = if best_calmar == f64::MAX {
        "∞".to_string()
    } else {
        format!("{:.4}", best_calmar)
    };
    let worst_str = if worst_calmar == f64::MAX {
        "∞".to_string()
    } else {
        format!("{:.4}", worst_calmar)
    };

    println!();
    println!("  ══════════════════════════════════════════════════════════════════════════════════════════════════════════");
    println!("  Total configs:  {}", results.len());
    println!("  Symbols:        {}", symbol_data.len());
    println!("  Runtime:        {:.1?}", sweep_elapsed);
    println!("  Best avg Calmar:  {}", best_str);
    println!("  Worst avg Calmar: {}", worst_str);
    println!("  ══════════════════════════════════════════════════════════════════════════════════════════════════════════");
    println!();
}

/// Sim realism comparison: demonstrates fill count reduction with queue model.
#[test]
fn bench_sim_realism() {
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════════════════════════════════════");
    println!("  SIM REALISM COMPARISON — queue position + trade-based fill matching");
    println!("═══════════════════════════════════════════════════════════════════════════════════════════════════════════════");

    // Realism stages: progressively more realistic simulation.
    let stages: Vec<(&str, SimConfig)> = vec![
        ("Legacy (book-based)", SimConfig::legacy()),
        (
            "Queue only",
            SimConfig {
                queue_model: true,
                partial_fills: false,
                latency_ns: 0,
                fill_probability: 1.0,
                ..SimConfig::default()
            },
        ),
        (
            "+ Partial fills",
            SimConfig {
                queue_model: true,
                partial_fills: true,
                latency_ns: 0,
                fill_probability: 1.0,
                ..SimConfig::default()
            },
        ),
        (
            "+ Latency 5ms",
            SimConfig {
                queue_model: true,
                partial_fills: true,
                latency_ns: 5_000_000,
                fill_probability: 1.0,
                ..SimConfig::default()
            },
        ),
        (
            "+ Pessimistic 50%",
            SimConfig {
                queue_model: true,
                partial_fills: true,
                latency_ns: 5_000_000,
                fill_probability: 0.5,
                ..SimConfig::default()
            },
        ),
    ];

    // Strategy configs to test.
    let strat_configs: Vec<(&str, StrategyParams)> = vec![(
        "adaptive_mm",
        StrategyParams {
            params: serde_json::json!({
                "risk_aversion": 0.3,
                "fill_intensity": 1.5,
                "time_horizon": 1.0,
                "vpin_factor": 2.0,
                "vpin_bucket_size": 50000.0,
                "vpin_n_buckets": 20,
                "size_decay_power": 2.0,
                "reduce_boost": 0.5,
            }),
        },
    )];

    let pairs = discover_symbols();
    // Fallback to single-file data if no series.
    let symbol_sources: SymbolData = if pairs.is_empty() {
        let mut sources = Vec::new();
        for sym in &["btcusdt", "ethusdt"] {
            if let Some((path, exchange)) = data_path(sym) {
                let events = load_events(&path).expect("failed to load");
                sources.push((exchange, sym.to_string(), events));
            }
        }
        sources
    } else {
        pairs
            .iter()
            .filter_map(|(exch, sym)| {
                let exchange_prefix = match exch {
                    Exchange::Binance => "binance",
                    Exchange::Bybit => "bybit",
                };
                load_series(exchange_prefix, sym).map(|events| (*exch, sym.clone(), events))
            })
            .collect()
    };

    if symbol_sources.is_empty() {
        println!("\n  SKIP: no data found\n");
        return;
    }

    for (exchange, sym, events) in &symbol_sources {
        let exchange_name = match exchange {
            Exchange::Binance => "Binance",
            Exchange::Bybit => "Bybit",
        };
        let book_count = events
            .iter()
            .filter(|(_, e)| matches!(e, replay_harness::ReplayEvent::Book(_)))
            .count();
        let trade_count = events
            .iter()
            .filter(|(_, e)| matches!(e, replay_harness::ReplayEvent::Trade(_)))
            .count();

        for (strat_name, params) in &strat_configs {
            println!();
            println!(
                "  {} / {} / {} — {} book + {} trade events",
                strat_name,
                sym.to_uppercase(),
                exchange_name,
                book_count,
                trade_count
            );
            println!("  {:-<130}", "");
            println!(
                "  {:>2} | {:<22} | {:>6} | {:>8} | {:>8} | {:>10} | {:>8} | {:>10} | {:>10} | {:>10} | {:>10}",
                "#", "Stage", "Fills", "Partial", "Avg Size", "Realized$", "$/fill", "Max DD$", "Max Pos", "Peak Not$", "QueueMiss"
            );
            println!("  {:-<130}", "");

            let mut baseline_fills: Option<usize> = None;

            for (i, (name, sim_config)) in stages.iter().enumerate() {
                let start = std::time::Instant::now();
                let result = run_strategy_on_events(
                    strat_name,
                    params,
                    *exchange,
                    &sym.to_uppercase(),
                    events,
                    sim_config.clone(),
                );
                let elapsed = start.elapsed();

                let dd = max_drawdown(&result.pnl_series);
                let pnl_per_fill = if result.fill_count > 0 {
                    result.total_pnl / result.fill_count as f64
                } else {
                    0.0
                };
                // Approximate average fill size: total fills / fill_count not directly available,
                // but we can estimate from max_position changes. Use order_count as proxy.
                let avg_fill_size = if result.fill_count > 0 {
                    // Cannot easily get total fill volume without tracking it; skip for now.
                    format!("{:>8}", "-")
                } else {
                    format!("{:>8}", "0")
                };

                let ratio_str = match baseline_fills {
                    Some(base) if base > 0 && result.fill_count > 0 => {
                        format!("  ({:.1}x fewer)", base as f64 / result.fill_count as f64)
                    }
                    _ => String::new(),
                };
                if baseline_fills.is_none() {
                    baseline_fills = Some(result.fill_count);
                }

                println!(
                    "  {:>2} | {:<22} | {:>6} | {:>8} | {} | {:>10.2} | {:>8.4} | {:>10.2} | {:>10.5} | {:>10.2} | {:>10.2}  ({:.1?}){}",
                    i, name, result.fill_count, result.partial_fill_count,
                    avg_fill_size, result.total_pnl, pnl_per_fill,
                    dd, result.max_position, result.peak_notional,
                    result.queue_volume_missed, elapsed, ratio_str,
                );
            }

            println!("  {:-<130}", "");
        }
    }
    println!();
}

/// ETHUSDT optimization sweep: order_size × reprice_threshold × flush_interval.
///
/// Sweeps 4×5×4 = 80 configs per exchange on 7-day ETHUSDT data (Binance and Bybit),
/// ranks by Calmar ratio (PnL / MaxDD), and prints results table.
///
/// Requires March 2-8 ETHUSDT data symlinked into testdata/. Skips gracefully if absent.
///
/// To set up data:
///   for d in $(seq -w 2 8); do
///     for f in .data/recordings/binance/binance_ethusdt_2026-03-0${d}_*.jsonl.gz; do
///       ln -sf "$(pwd)/$f" testdata/$(basename "$f"); done; done
///   # repeat for bybit
#[test]
fn bench_ethusdt_optimization() {
    const DEFAULT_MAX_POSITION: f64 = 0.1;

    let base = serde_json::json!({
        "risk_aversion": 0.3,
        "fill_intensity": 1.5,
        "time_horizon": 1.0,
        "vpin_factor": 2.0,
        "vpin_bucket_size": 50000.0,
        "vpin_n_buckets": 20,
        "size_decay_power": 2.0,
        "reduce_boost": 0.5,
    });

    let order_sizes: &[f64] = &[0.01, 0.02, 0.05, 0.1];
    let reprice_bps_values: &[f64] = &[1.5, 3.0, 5.0, 8.0, 12.0];
    let flush_intervals: &[u64] = &[0, 500, 1000, 2000];

    struct SweepResult {
        order_size: f64,
        reprice_bps: f64,
        flush_int: u64,
        fills: usize,
        orders: usize,
        realized: f64,
        m2m_pnl: f64,
        fees: f64,
        max_dd: f64,
        calmar: f64,
        pnl_per_fill: f64,
    }

    fn compute_calmar(pnl: f64, dd: f64) -> f64 {
        if dd > 1e-6 {
            pnl / dd
        } else if pnl > 0.0 {
            f64::MAX
        } else {
            0.0
        }
    }

    println!();
    println!("══════════════════════════════════════════════════════════════════════════════════════════");
    println!("  ETHUSDT OPTIMIZATION SWEEP — order_size × reprice_bps × flush_interval");
    println!("  Fixed: FULL STACK (γ=0.3, κ=1.5, τ=1.0, vpin=2.0, decay=2.0, boost=0.5)");
    println!("══════════════════════════════════════════════════════════════════════════════════════════");

    for &(exchange, exchange_str, symbol_str) in &[
        (Exchange::Binance, "binance", "ETHUSDT"),
        (Exchange::Bybit, "bybit", "ETHUSDT"),
    ] {
        let data_dir_candidates = ["testdata", "../../testdata"];
        let mut events_opt: Option<Vec<(u64, replay_harness::ReplayEvent)>> = None;
        for dir in &data_dir_candidates {
            let files = find_series_files(dir, exchange_str, "ethusdt");
            if !files.is_empty() {
                match load_events_multi(&files) {
                    Ok(ev) => {
                        events_opt = Some(ev);
                        break;
                    }
                    Err(e) => eprintln!("WARN: failed to load {} ETHUSDT: {}", exchange_str, e),
                }
            }
        }
        let events = match events_opt {
            Some(e) => e,
            None => {
                println!(
                    "\n  SKIP: no {} ETHUSDT series data in testdata/\n",
                    exchange_str.to_uppercase()
                );
                continue;
            }
        };

        let event_count = events.len();
        let book_count = events
            .iter()
            .filter(|(_, e)| matches!(e, replay_harness::ReplayEvent::Book(_)))
            .count();
        println!(
            "\n  {} ETHUSDT — {} events ({} book updates)",
            exchange_str.to_uppercase(),
            event_count,
            book_count
        );

        // Baseline: FULL STACK, order_size=0.01, reprice=1.5bps, no flush
        let baseline_params = StrategyParams {
            params: {
                let mut p = base.clone();
                p["order_size"] = serde_json::json!(0.01);
                p["reprice_threshold_bps"] = serde_json::json!(1.5);
                p["max_position"] = serde_json::json!(DEFAULT_MAX_POSITION);
                p
            },
        };
        let baseline_result = run_strategy_on_events(
            "adaptive_mm",
            &baseline_params,
            exchange,
            symbol_str,
            &events,
            SimConfig::default(),
        );
        let baseline_m2m = baseline_result.pnl_series.last().copied().unwrap_or(0.0);
        let baseline_dd = max_drawdown(&baseline_result.pnl_series);
        let baseline_calmar = compute_calmar(baseline_result.total_pnl, baseline_dd);
        println!(
            "  BASELINE  order=0.01 reprice=1.5bps flush=0 → fills={} orders={} realized=${:.2} m2m=${:.2} fees=${:.2} dd=${:.2} calmar={:.3}",
            baseline_result.fill_count, baseline_result.order_count,
            baseline_result.total_pnl, baseline_m2m, baseline_result.fee_total, baseline_dd, baseline_calmar
        );

        // Run sweep
        let start_all = std::time::Instant::now();
        let mut results: Vec<SweepResult> = Vec::new();

        for &order_size in order_sizes {
            // Scale max_position proportionally so all order_size values share same relative risk
            let max_pos = DEFAULT_MAX_POSITION.max(5.0 * order_size);

            for &reprice_bps in reprice_bps_values {
                for &flush_int in flush_intervals {
                    let mut params_json = base.clone();
                    params_json["order_size"] = serde_json::json!(order_size);
                    params_json["reprice_threshold_bps"] = serde_json::json!(reprice_bps);
                    params_json["max_position"] = serde_json::json!(max_pos);
                    if flush_int > 0 {
                        params_json["flush_interval_ticks"] = serde_json::json!(flush_int);
                        params_json["flush_threshold"] = serde_json::json!(0.0);
                    }
                    let params = StrategyParams {
                        params: params_json,
                    };

                    let sim_cfg = SimConfig {
                        timer_interval: if flush_int > 0 { 1000 } else { 0 },
                        ..Default::default()
                    };

                    let result = run_strategy_on_events(
                        "adaptive_mm",
                        &params,
                        exchange,
                        symbol_str,
                        &events,
                        sim_cfg,
                    );

                    let m2m = result.pnl_series.last().copied().unwrap_or(0.0);
                    let dd = max_drawdown(&result.pnl_series);
                    let calmar = compute_calmar(result.total_pnl, dd);
                    let pnl_per_fill = if result.fill_count > 0 {
                        result.total_pnl / result.fill_count as f64
                    } else {
                        0.0
                    };

                    results.push(SweepResult {
                        order_size,
                        reprice_bps,
                        flush_int,
                        fills: result.fill_count,
                        orders: result.order_count,
                        realized: result.total_pnl,
                        m2m_pnl: m2m,
                        fees: result.fee_total,
                        max_dd: dd,
                        calmar,
                        pnl_per_fill,
                    });
                }
            }
        }

        // Sort by Calmar descending
        results.sort_by(|a, b| {
            b.calmar
                .partial_cmp(&a.calmar)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        println!(
            "  Sweep: {} configs in {:.1?}",
            results.len(),
            start_all.elapsed()
        );
        println!();
        println!(
            "  {:>4} | {:>7} | {:>10} | {:>8} | {:>6} | {:>7} | {:>10} | {:>10} | {:>8} | {:>8} | {:>8} | {:>7}",
            "Rank", "order_sz", "reprice_bps", "flush_int", "Fills", "Orders",
            "Realized$", "M2M PnL$", "Fees$", "Max DD$", "Calmar", "$/fill"
        );
        println!("  {:-<120}", "");

        let top_n = results.len().min(20);
        for (rank, r) in results.iter().take(top_n).enumerate() {
            let order_reduction = if baseline_result.order_count > 0 {
                r.orders as f64 / baseline_result.order_count as f64
            } else {
                1.0
            };
            let marker = if order_reduction <= 0.7 { " ★" } else { "" };
            println!(
                "  {:>4} | {:>7.4} | {:>10.1} | {:>8} | {:>6} | {:>7} | {:>10.2} | {:>10.2} | {:>8.2} | {:>8.2} | {:>8.3} | {:>7.4}{}",
                rank + 1, r.order_size, r.reprice_bps, r.flush_int,
                r.fills, r.orders, r.realized, r.m2m_pnl,
                r.fees, r.max_dd, r.calmar, r.pnl_per_fill, marker
            );
        }

        if results.len() > top_n {
            println!("  {:>4} | ...", "...");
            let bottom_start = results.len().saturating_sub(5);
            println!("  --- Bottom 5 ---");
            for (i, r) in results[bottom_start..].iter().enumerate() {
                println!(
                    "  {:>4} | {:>7.4} | {:>10.1} | {:>8} | {:>6} | {:>7} | {:>10.2} | {:>10.2} | {:>8.2} | {:>8.2} | {:>8.3} | {:>7.4}",
                    results.len() - 5 + i + 1, r.order_size, r.reprice_bps, r.flush_int,
                    r.fills, r.orders, r.realized, r.m2m_pnl,
                    r.fees, r.max_dd, r.calmar, r.pnl_per_fill
                );
            }
        }

        println!("  {:-<120}", "");
        println!(
            "  ★ = order count ≤ 70% of baseline ({} orders)",
            baseline_result.order_count
        );
        println!("  Baseline calmar: {:.3}", baseline_calmar);
    }

    println!();
}

/// VPIN factor sweep on ETHUSDT — fixed reprice_bps=12, order_size=0.01.
///
/// Sweeps vpin_factor × [0.0, 2.0, 4.0, 6.0, 8.0, 10.0] on 7-day ETHUSDT data
/// (Binance and Bybit), ranks by Calmar ratio. Hypothesis: higher VPIN factor
/// reduces adverse selection enough to improve realized PnL.
///
/// Requires March 2-8 ETHUSDT data symlinked into testdata/. Skips gracefully if absent.
#[test]
fn bench_vpin_factor_sweep() {
    let vpin_factors: &[f64] = &[0.0, 2.0, 4.0, 6.0, 8.0, 10.0];

    struct VpinResult {
        vpin_factor: f64,
        fills: usize,
        orders: usize,
        realized: f64,
        m2m_pnl: f64,
        fees: f64,
        max_dd: f64,
        calmar: f64,
        pnl_per_fill: f64,
    }

    fn compute_calmar_vpin(pnl: f64, dd: f64) -> f64 {
        if dd > 1e-6 {
            pnl / dd
        } else if pnl > 0.0 {
            f64::MAX
        } else {
            0.0
        }
    }

    println!();
    println!("══════════════════════════════════════════════════════════════════════════════════════════");
    println!("  ETHUSDT VPIN FACTOR SWEEP — vpin_factor × [0, 2, 4, 6, 8, 10]");
    println!("  Fixed: reprice_bps=12, order_size=0.01, flush=0, γ=0.3, κ=1.5, τ=1.0");
    println!("══════════════════════════════════════════════════════════════════════════════════════════");

    for &(exchange, exchange_str, symbol_str) in &[
        (Exchange::Binance, "binance", "ETHUSDT"),
        (Exchange::Bybit, "bybit", "ETHUSDT"),
    ] {
        let data_dir_candidates = ["testdata", "../../testdata"];
        let mut events_opt: Option<Vec<(u64, replay_harness::ReplayEvent)>> = None;
        for dir in &data_dir_candidates {
            let files = find_series_files(dir, exchange_str, "ethusdt");
            if !files.is_empty() {
                match load_events_multi(&files) {
                    Ok(ev) => {
                        events_opt = Some(ev);
                        break;
                    }
                    Err(e) => eprintln!("WARN: failed to load {} ETHUSDT: {}", exchange_str, e),
                }
            }
        }
        let events = match events_opt {
            Some(e) => e,
            None => {
                println!(
                    "\n  SKIP: no {} ETHUSDT series data in testdata/\n",
                    exchange_str.to_uppercase()
                );
                continue;
            }
        };

        let event_count = events.len();
        let book_count = events
            .iter()
            .filter(|(_, e)| matches!(e, replay_harness::ReplayEvent::Book(_)))
            .count();
        println!(
            "\n  {} ETHUSDT — {} events ({} book updates)",
            exchange_str.to_uppercase(),
            event_count,
            book_count
        );

        let start_all = std::time::Instant::now();
        let mut results: Vec<VpinResult> = Vec::new();

        for &vpin_factor in vpin_factors {
            let params = StrategyParams {
                params: serde_json::json!({
                    "risk_aversion": 0.3,
                    "fill_intensity": 1.5,
                    "time_horizon": 1.0,
                    "vpin_factor": vpin_factor,
                    "vpin_bucket_size": 50000.0,
                    "vpin_n_buckets": 20,
                    "size_decay_power": 2.0,
                    "reduce_boost": 0.5,
                    "order_size": 0.01,
                    "reprice_threshold_bps": 12.0,
                    "max_position": 0.5,
                }),
            };

            let result = run_strategy_on_events(
                "adaptive_mm",
                &params,
                exchange,
                symbol_str,
                &events,
                SimConfig::default(),
            );

            let m2m = result.pnl_series.last().copied().unwrap_or(0.0);
            let dd = max_drawdown(&result.pnl_series);
            let calmar = compute_calmar_vpin(result.total_pnl, dd);
            let pnl_per_fill = if result.fill_count > 0 {
                result.total_pnl / result.fill_count as f64
            } else {
                0.0
            };

            results.push(VpinResult {
                vpin_factor,
                fills: result.fill_count,
                orders: result.order_count,
                realized: result.total_pnl,
                m2m_pnl: m2m,
                fees: result.fee_total,
                max_dd: dd,
                calmar,
                pnl_per_fill,
            });
        }

        // Sort by Calmar descending
        results.sort_by(|a, b| {
            b.calmar
                .partial_cmp(&a.calmar)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        println!(
            "  Sweep: {} configs in {:.1?}",
            results.len(),
            start_all.elapsed()
        );
        println!();
        println!(
            "  {:>4} | {:>11} | {:>6} | {:>7} | {:>10} | {:>10} | {:>8} | {:>8} | {:>8} | {:>7}",
            "Rank",
            "vpin_factor",
            "Fills",
            "Orders",
            "Realized$",
            "M2M PnL$",
            "Fees$",
            "Max DD$",
            "Calmar",
            "$/fill"
        );
        println!("  {:-<105}", "");

        for (rank, r) in results.iter().enumerate() {
            println!(
                "  {:>4} | {:>11.1} | {:>6} | {:>7} | {:>10.2} | {:>10.2} | {:>8.2} | {:>8.2} | {:>8.3} | {:>7.4}",
                rank + 1,
                r.vpin_factor,
                r.fills,
                r.orders,
                r.realized,
                r.m2m_pnl,
                r.fees,
                r.max_dd,
                r.calmar,
                r.pnl_per_fill
            );
        }

        println!("  {:-<105}", "");
        println!("  vpin_factor=2.0 is current baseline");
    }

    println!();
}

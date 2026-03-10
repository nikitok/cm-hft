//! cm-analyze-symbol — Analyze a trading symbol for HFT market-making suitability.
//!
//! Connects to Binance WebSocket, collects orderbook and trade data for a
//! configurable duration (default 5 minutes), computes microstructure metrics,
//! and prints a JSON report with a go/no-go verdict to stdout.
//!
//! Usage:
//!   cm-analyze-symbol --symbol BTCUSDT --duration 5m
//!   cm-analyze-symbol --symbol ETHUSDT --duration 30s --thresholds '{"max_spread_bps":3.0}'

use std::process;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use clap::Parser;
use tokio::sync::mpsc;

use cm_core::config::ExchangeConfig;
use cm_core::types::{BookUpdate, Exchange, Symbol, Trade};
use cm_market_data::binance::{BinanceMessage, BinanceWsClient};
use cm_market_data::orderbook::OrderBook;
use cm_trading::analyzer::{AnalysisVerdict, SymbolAnalyzer, Thresholds};

#[derive(Parser)]
#[command(
    name = "cm-analyze-symbol",
    about = "Analyze a trading symbol for HFT market-making suitability via Binance WebSocket"
)]
struct Args {
    /// Symbol to analyze (e.g., BTCUSDT, ETHUSDT).
    #[arg(long)]
    symbol: String,

    /// Collection duration (e.g., 5m, 30s, 1h). Default: 5m.
    #[arg(long, default_value = "5m")]
    duration: String,

    /// JSON thresholds override (merged with defaults).
    /// Example: '{"max_spread_bps": 3.0, "min_trade_rate": 2.0}'
    #[arg(long)]
    thresholds: Option<String>,
}

fn parse_duration(s: &str) -> Result<Duration> {
    let s = s.trim();
    if let Some(minutes) = s.strip_suffix('m') {
        let m: u64 = minutes.parse()?;
        Ok(Duration::from_secs(m * 60))
    } else if let Some(hours) = s.strip_suffix('h') {
        let h: u64 = hours.parse()?;
        Ok(Duration::from_secs(h * 3600))
    } else if let Some(secs) = s.strip_suffix('s') {
        let s: u64 = secs.parse()?;
        Ok(Duration::from_secs(s))
    } else {
        anyhow::bail!("invalid duration '{}', use e.g. 30m, 1h, 300s", s);
    }
}

/// Convert `BinanceDepthSnapshot` string levels to `(Price, Quantity)` pairs.
///
/// Skips levels where price or quantity fails to parse or is zero — these
/// would corrupt depth USD calculations if injected into the order book.
fn parse_snapshot_levels(
    levels: &[[String; 2]],
) -> Vec<(cm_core::types::Price, cm_core::types::Quantity)> {
    levels
        .iter()
        .filter_map(|pair| {
            let p: f64 = match pair[0].parse() {
                Ok(v) => v,
                Err(_) => {
                    tracing::warn!(raw = %pair[0], "skipping snapshot level: invalid price string");
                    return None;
                }
            };
            let q: f64 = match pair[1].parse() {
                Ok(v) => v,
                Err(_) => {
                    tracing::warn!(raw = %pair[1], "skipping snapshot level: invalid qty string");
                    return None;
                }
            };
            if p == 0.0 || q == 0.0 {
                tracing::warn!(price = p, qty = q, "skipping zero-value snapshot level");
                return None;
            }
            Some((
                cm_core::types::Price::from(p),
                cm_core::types::Quantity::from(q),
            ))
        })
        .collect()
}

/// Run the Binance WS feed: connect, subscribe, pump events into channels.
///
/// Reconnects on error until the cancellation token is triggered.
async fn run_ws_feed(
    symbol: String,
    book_tx: mpsc::Sender<BookUpdate>,
    trade_tx: mpsc::Sender<Trade>,
    shutdown: tokio_util::sync::CancellationToken,
) {
    let config = ExchangeConfig {
        api_key: String::new(),
        api_secret: String::new(),
        testnet: false,
        ws_url: "wss://stream.binance.com:9443/ws".to_string(),
        rest_url: "https://api.binance.com".to_string(),
        timeout_ms: 5000,
    };
    let client = BinanceWsClient::new(config, vec![symbol]);

    loop {
        if shutdown.is_cancelled() {
            break;
        }

        let mut stream = match client.connect().await {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(error = %e, "WS connect failed, retrying in 5s");
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };

        if let Err(e) = client.subscribe(&mut stream).await {
            tracing::error!(error = %e, "WS subscribe failed, retrying in 5s");
            tokio::time::sleep(Duration::from_secs(5)).await;
            continue;
        }

        tracing::info!("Binance WS connected and subscribed");

        loop {
            if shutdown.is_cancelled() {
                break;
            }
            match BinanceWsClient::read_message(&mut stream).await {
                Ok(Some(BinanceMessage::Depth(update))) => {
                    if book_tx.send(update).await.is_err() {
                        return;
                    }
                }
                Ok(Some(BinanceMessage::Trade(trade))) => {
                    if trade_tx.send(trade).await.is_err() {
                        return;
                    }
                }
                Ok(None) => continue,
                Err(e) => {
                    tracing::warn!(error = %e, "WS read error, reconnecting");
                    break;
                }
            }
        }

        if shutdown.is_cancelled() {
            break;
        }
        tracing::warn!("WS disconnected, reconnecting in 5s");
        tokio::time::sleep(Duration::from_secs(5)).await;
    }

    tracing::info!("WS feed task stopped");
}

/// Fetch REST depth snapshot and initialize the order book.
///
/// Retries up to 3 times with 5s sleep between attempts to handle transient
/// rate limits or network errors at startup.
///
/// Returns the initialized book and the `last_update_id` (used as the starting
/// value for the monotonic update counter).
async fn init_order_book(client: &BinanceWsClient, symbol: &str) -> Result<(OrderBook, u64)> {
    let mut last_err = anyhow::anyhow!("snapshot fetch never attempted");
    for attempt in 1..=3u32 {
        match client.fetch_snapshot(symbol).await {
            Ok(snapshot) => {
                let bids = parse_snapshot_levels(&snapshot.bids);
                let asks = parse_snapshot_levels(&snapshot.asks);

                // NOTE: Use snapshot.last_update_id (NOT hardcoded 1) so that pre-snapshot
                // WS events buffered in the mpsc channel are correctly rejected as stale
                // when we call apply_update with a monotonic counter starting from last_update_id+1.
                let mut book = OrderBook::new(Exchange::Binance, Symbol::new(symbol));
                book.apply_snapshot(&bids, &asks, snapshot.last_update_id);

                tracing::info!(
                    symbol = %symbol,
                    last_update_id = snapshot.last_update_id,
                    bid_levels = bids.len(),
                    ask_levels = asks.len(),
                    "order book initialized from REST snapshot"
                );

                return Ok((book, snapshot.last_update_id));
            }
            Err(e) => {
                tracing::warn!(attempt, error = %e, "snapshot fetch failed, retrying in 5s");
                last_err = e;
                if attempt < 3 {
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
        }
    }
    Err(last_err).with_context(|| format!("snapshot fetch failed after 3 attempts for {symbol}"))
}

#[tokio::main]
async fn main() -> Result<()> {
    cm_core::logging::init_tracing(true);

    let args = Args::parse();
    let symbol = args.symbol.to_uppercase();
    let configured_duration = parse_duration(&args.duration)?;

    // Parse threshold overrides (merge with defaults).
    let thresholds: Thresholds = if let Some(json) = &args.thresholds {
        // Deserialize into a serde_json::Value so we can merge partial overrides.
        let defaults = serde_json::to_value(Thresholds::default())?;
        let overrides: serde_json::Value = serde_json::from_str(json)?;
        let mut merged = defaults;
        if let (serde_json::Value::Object(ref mut m), serde_json::Value::Object(ref o)) =
            (&mut merged, &overrides)
        {
            for (k, v) in o {
                m.insert(k.clone(), v.clone());
            }
        }
        serde_json::from_value(merged)?
    } else {
        Thresholds::default()
    };

    tracing::info!(
        symbol = %symbol,
        duration = ?configured_duration,
        "starting symbol analysis"
    );

    // Channels.
    let (book_tx, mut book_rx) = mpsc::channel::<BookUpdate>(4096);
    let (trade_tx, mut trade_rx) = mpsc::channel::<Trade>(4096);

    // Shutdown token.
    let shutdown = tokio_util::sync::CancellationToken::new();

    // Spawn WS feed task (subscribes and pumps events).
    let ws_shutdown = shutdown.clone();
    let ws_symbol = symbol.clone();
    let _ws_handle = tokio::spawn(async move {
        run_ws_feed(ws_symbol, book_tx, trade_tx, ws_shutdown).await;
    });

    // Initialize order book from REST snapshot (after WS is connected + subscribed).
    // Small delay to allow WS to connect before snapshot — any depth events arriving
    // between subscribe() and snapshot fetch will be buffered in the mpsc channel.
    // When we process them with our monotonic counter (> snapshot.last_update_id),
    // apply_update will accept them only if counter > last_update_id; pre-snapshot
    // events are correctly rejected as StaleUpdate if their implicit ID would be lower.
    // The monotonic counter approach handles this correctly.
    let config = ExchangeConfig {
        api_key: String::new(),
        api_secret: String::new(),
        testnet: false,
        ws_url: "wss://stream.binance.com:9443/ws".to_string(),
        rest_url: "https://api.binance.com".to_string(),
        timeout_ms: 5000,
    };
    let snapshot_client = BinanceWsClient::new(config, vec![symbol.clone()]);

    let (mut book, snapshot_update_id) = match init_order_book(&snapshot_client, &symbol).await {
        Ok(result) => result,
        Err(e) => {
            tracing::error!(error = %e, "failed to fetch order book snapshot");
            anyhow::bail!("cannot initialize order book: {}", e);
        }
    };

    // Monotonic counter starting at snapshot's last_update_id.
    // Each depth event increments by 1, ensuring in-order processing.
    let mut update_counter = snapshot_update_id;

    let mut analyzer = SymbolAnalyzer::new();
    let start = Instant::now();

    // Ctrl+C handler.
    let ctrlc_shutdown = shutdown.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        tracing::info!("Ctrl+C received, shutting down");
        ctrlc_shutdown.cancel();
    });

    // Duration timer.
    let timer_shutdown = shutdown.clone();
    tokio::spawn(async move {
        tokio::time::sleep(configured_duration).await;
        tracing::info!("collection duration elapsed, shutting down");
        timer_shutdown.cancel();
    });

    // Main event loop.
    let early_exit = loop {
        tokio::select! {
            _ = shutdown.cancelled() => {
                tracing::info!("shutdown signal received");
                break false;
            }
            msg = book_rx.recv() => {
                match msg {
                    Some(update) => {
                        update_counter += 1;
                        match book.apply_update(&update, update_counter) {
                            Ok(()) => analyzer.on_book_update(&book),
                            Err(e) => tracing::debug!(error = %e, "skipping stale/invalid book update"),
                        }
                    }
                    None => {
                        // WS feed task died.
                        tracing::warn!("book channel closed — WS feed task terminated");
                        break true;
                    }
                }
            }
            msg = trade_rx.recv() => {
                match msg {
                    Some(trade) => analyzer.on_trade(&trade),
                    None => {
                        tracing::warn!("trade channel closed — WS feed task terminated");
                        break true;
                    }
                }
            }
        }
    };

    shutdown.cancel();

    let elapsed = start.elapsed();
    let report = analyzer.report(elapsed, &thresholds);

    // Print JSON report to stdout (tracing goes to stderr).
    println!("{}", serde_json::to_string_pretty(&report)?);

    // Determine exit code.
    let exit_code = if early_exit && elapsed < configured_duration / 2 {
        tracing::warn!(
            elapsed_secs = elapsed.as_secs_f64(),
            "WS feed died before 50% of configured duration — Insufficient result"
        );
        2
    } else {
        match &report.verdict {
            AnalysisVerdict::Go => 0,
            AnalysisVerdict::NoGo { .. } => 1,
            AnalysisVerdict::Insufficient { .. } => 2,
        }
    };

    process::exit(exit_code);
}

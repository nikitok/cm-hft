//! cm-record — Record Bybit market data to JSONL files.
//!
//! Writes to plain `.jsonl` during recording for crash safety. On clean
//! shutdown, compresses to `.jsonl.gz` and removes the uncompressed file.
//! If the process is killed mid-session, the `.jsonl` file survives with
//! all data up to the last write — the shell script will compress it on
//! next run.
//!
//! Usage:
//!   cm-record --symbols BTCUSDT,ETHUSDT --duration 30m --output testdata/

use std::collections::HashMap;
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::Serialize;
use tokio::sync::mpsc;

use cm_core::types::{BookUpdate, Trade};
use cm_market_data::bybit::client::{BybitConfig, BybitWsClient};

/// A recorded market data event.
#[derive(Debug, Serialize)]
struct RecordedEvent {
    /// Nanosecond timestamp of the event.
    ts_ns: u64,
    /// Event kind: "book" or "trade".
    kind: &'static str,
    /// Serialized event data.
    data: serde_json::Value,
}

#[derive(Parser)]
#[command(name = "cm-record", about = "Record Bybit market data to JSONL.gz")]
struct Args {
    /// Comma-separated symbols to record (e.g., BTCUSDT,ETHUSDT).
    #[arg(long, value_delimiter = ',')]
    symbols: Vec<String>,

    /// Recording duration (e.g., 30m, 1h, 5m).
    #[arg(long, default_value = "30m")]
    duration: String,

    /// Output directory for JSONL.gz files.
    #[arg(long, default_value = "testdata")]
    output: PathBuf,

    /// Use timestamped filenames: bybit_btcusdt_2026-02-26_22:00.jsonl.gz
    #[arg(long, default_value_t = false)]
    timestamp: bool,
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
        anyhow::bail!("invalid duration format '{}', use e.g. 30m, 1h, 300s", s);
    }
}

/// Build the base filename (without extension) for a symbol.
fn base_filename(symbol: &str, duration_label: &str, timestamp: bool) -> String {
    if timestamp {
        let now = chrono::Local::now();
        format!(
            "bybit_{}_{}",
            symbol.to_lowercase(),
            now.format("%Y-%m-%d_%H:%M")
        )
    } else {
        format!("bybit_{}_{}", symbol.to_lowercase(), duration_label)
    }
}

/// Compress a .jsonl file to .jsonl.gz and remove the original.
fn compress_file(jsonl_path: &PathBuf, gz_path: &PathBuf) -> Result<()> {
    let input = std::fs::File::open(jsonl_path)?;
    let reader = std::io::BufReader::new(input);
    let output = std::fs::File::create(gz_path)?;
    let mut encoder = GzEncoder::new(output, Compression::default());
    std::io::copy(&mut reader.take(u64::MAX), &mut encoder)?;
    encoder.finish()?;
    std::fs::remove_file(jsonl_path)?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    cm_core::logging::init_tracing(true);
    let args = Args::parse();

    if args.symbols.is_empty() {
        anyhow::bail!("no symbols specified, use --symbols BTCUSDT,ETHUSDT");
    }

    let duration = parse_duration(&args.duration)?;
    let duration_label = &args.duration;

    // Create output directory.
    std::fs::create_dir_all(&args.output)?;

    tracing::info!(
        symbols = ?args.symbols,
        duration = ?duration,
        output = %args.output.display(),
        "starting recorder"
    );

    // Build paths and open plain JSONL writers (crash-safe).
    struct SymWriter {
        writer: BufWriter<std::fs::File>,
        jsonl_path: PathBuf,
        gz_path: PathBuf,
    }

    let mut writers: HashMap<String, SymWriter> = HashMap::new();
    for symbol in &args.symbols {
        let base = base_filename(symbol, duration_label, args.timestamp);
        let jsonl_path = args.output.join(format!("{}.jsonl", base));
        let gz_path = args.output.join(format!("{}.jsonl.gz", base));

        let file = std::fs::File::create(&jsonl_path)?;
        let writer = BufWriter::new(file);
        writers.insert(
            symbol.clone(),
            SymWriter {
                writer,
                jsonl_path: jsonl_path.clone(),
                gz_path,
            },
        );
        tracing::info!(path = %jsonl_path.display(), "writing to plain JSONL (will compress on clean shutdown)");
    }

    // Event counters per symbol.
    let counters: Arc<HashMap<String, AtomicUsize>> = Arc::new(
        args.symbols
            .iter()
            .map(|s| (s.clone(), AtomicUsize::new(0)))
            .collect(),
    );

    // Channels for book and trade events.
    let (book_tx, mut book_rx) = mpsc::channel::<BookUpdate>(4096);
    let (trade_tx, mut trade_rx) = mpsc::channel::<Trade>(4096);

    // Spawn Bybit WS client.
    let client = BybitWsClient::new(BybitConfig::default(), args.symbols.clone());
    let ws_handle = tokio::spawn(async move {
        if let Err(e) = client.run(book_tx, trade_tx).await {
            tracing::error!(error = %e, "Bybit WS client error");
        }
    });

    // Shutdown signal.
    let shutdown = tokio_util::sync::CancellationToken::new();
    let shutdown_clone = shutdown.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        tracing::info!("Ctrl+C received, shutting down...");
        shutdown_clone.cancel();
    });

    // SIGTERM handler for graceful shutdown in K8s.
    let shutdown_sigterm = shutdown.clone();
    tokio::spawn(async move {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to register SIGTERM handler");
        sigterm.recv().await;
        tracing::info!("SIGTERM received, shutting down...");
        shutdown_sigterm.cancel();
    });

    // Timer-based shutdown.
    let shutdown_timer = shutdown.clone();
    tokio::spawn(async move {
        tokio::time::sleep(duration).await;
        tracing::info!("recording duration elapsed, shutting down...");
        shutdown_timer.cancel();
    });

    // Progress reporting.
    let counters_progress = counters.clone();
    let symbols_progress = args.symbols.clone();
    let shutdown_progress = shutdown.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        loop {
            interval.tick().await;
            if shutdown_progress.is_cancelled() {
                break;
            }
            for sym in &symbols_progress {
                if let Some(c) = counters_progress.get(sym) {
                    tracing::info!(symbol = %sym, events = c.load(Ordering::Relaxed), "progress");
                }
            }
        }
    });

    // Periodic flush interval (every 5s) to minimize data loss on crash.
    let mut flush_counter: u64 = 0;
    let flush_interval: u64 = 5000; // events between flushes

    // Main event loop: drain both channels until shutdown.
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => {
                tracing::info!("shutdown signal received");
                break;
            }
            Some(book) = book_rx.recv() => {
                let sym = book.symbol.0.clone();
                if let Some(sw) = writers.get_mut(&sym) {
                    let event = RecordedEvent {
                        ts_ns: book.timestamp.as_nanos(),
                        kind: "book",
                        data: serde_json::to_value(&book)?,
                    };
                    serde_json::to_writer(&mut sw.writer, &event)?;
                    sw.writer.write_all(b"\n")?;
                    if let Some(c) = counters.get(&sym) {
                        c.fetch_add(1, Ordering::Relaxed);
                    }
                    flush_counter += 1;
                    if flush_counter % flush_interval == 0 {
                        sw.writer.flush()?;
                    }
                }
            }
            Some(trade) = trade_rx.recv() => {
                let sym = trade.symbol.0.clone();
                if let Some(sw) = writers.get_mut(&sym) {
                    let event = RecordedEvent {
                        ts_ns: trade.timestamp.as_nanos(),
                        kind: "trade",
                        data: serde_json::to_value(&trade)?,
                    };
                    serde_json::to_writer(&mut sw.writer, &event)?;
                    sw.writer.write_all(b"\n")?;
                    if let Some(c) = counters.get(&sym) {
                        c.fetch_add(1, Ordering::Relaxed);
                    }
                    flush_counter += 1;
                    if flush_counter % flush_interval == 0 {
                        sw.writer.flush()?;
                    }
                }
            }
        }
    }

    // Flush, compress, and close all writers.
    for (sym, mut sw) in writers {
        let count = counters
            .get(&sym)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0);

        // Flush the BufWriter.
        sw.writer.flush()?;
        drop(sw.writer); // close the file handle

        // Compress .jsonl → .jsonl.gz
        tracing::info!(symbol = %sym, events = count, src = %sw.jsonl_path.display(), "compressing");
        compress_file(&sw.jsonl_path, &sw.gz_path)?;
        tracing::info!(symbol = %sym, events = count, dst = %sw.gz_path.display(), "file closed");
    }

    ws_handle.abort();

    // Print summary.
    for sym in &args.symbols {
        let count = counters
            .get(sym)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0);
        tracing::info!(symbol = %sym, total_events = count, "recording complete");
    }

    Ok(())
}

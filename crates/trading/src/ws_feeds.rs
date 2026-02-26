//! WebSocket feed integration for Binance and Bybit.
//!
//! Each exchange feed runs as a tokio task, maintains a local [`OrderBook`],
//! sends [`MarketDataEvent`]s to the strategy thread, updates the
//! [`PaperExecutor`] (if in paper mode), and stores mid prices for risk checks.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossbeam::channel::Sender;
use tokio_util::sync::CancellationToken;

use cm_core::config::ExchangeConfig;
use cm_core::types::*;
use cm_market_data::binance::{BinanceMessage, BinanceWsClient, BinanceDepthSnapshot};
use cm_market_data::bybit::{BybitConfig, BybitWsClient};
use cm_market_data::orderbook::OrderBook;
use cm_market_data::ws::ReconnectConfig as WsReconnectConfig;

use crate::engine::SharedState;
use crate::event_loop::MarketDataEvent;
use crate::paper_executor::PaperExecutor;

/// Monotonic counter for book update sequencing across all feeds.
static FEED_BOOK_SEQ: AtomicU64 = AtomicU64::new(1);

// ────────────────────────────────────────────────────────────────────
// Binance feed
// ────────────────────────────────────────────────────────────────────

/// Run the Binance market data feed.
///
/// Connects to Binance WebSocket, subscribes to depth + trade streams,
/// fetches REST snapshots, and reads messages in a reconnecting loop.
pub async fn run_binance_feed(
    config: ExchangeConfig,
    symbols: Vec<String>,
    md_tx: Sender<MarketDataEvent>,
    state: Arc<SharedState>,
    paper_executor: Option<Arc<PaperExecutor>>,
    cancel: CancellationToken,
) {
    let client = BinanceWsClient::new(config, symbols.clone());
    let mut books: HashMap<(Exchange, Symbol), OrderBook> = HashMap::new();

    loop {
        if cancel.is_cancelled() {
            break;
        }

        // Connect
        let mut stream = match client.connect().await {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(error = %e, "Binance WS connect failed, retrying");
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };

        // Subscribe
        if let Err(e) = client.subscribe(&mut stream).await {
            tracing::error!(error = %e, "Binance WS subscribe failed, retrying");
            tokio::time::sleep(Duration::from_secs(5)).await;
            continue;
        }

        // Fetch snapshots for each symbol
        for symbol in &symbols {
            match client.fetch_snapshot(symbol).await {
                Ok(snapshot) => {
                    let book_update = snapshot_to_book_update(&snapshot, symbol);
                    apply_book_update(
                        &mut books,
                        &book_update,
                        &md_tx,
                        &state,
                        &paper_executor,
                    );
                }
                Err(e) => {
                    tracing::error!(symbol = %symbol, error = %e, "Binance snapshot fetch failed");
                }
            }
        }

        // Message read loop
        loop {
            if cancel.is_cancelled() {
                break;
            }
            match BinanceWsClient::read_message(&mut stream).await {
                Ok(Some(BinanceMessage::Depth(book_update))) => {
                    apply_book_update(
                        &mut books,
                        &book_update,
                        &md_tx,
                        &state,
                        &paper_executor,
                    );
                }
                Ok(Some(BinanceMessage::Trade(trade))) => {
                    let _ = md_tx.send(MarketDataEvent::Trade(trade));
                }
                Ok(None) => continue,
                Err(e) => {
                    tracing::error!(error = %e, "Binance WS read error, reconnecting");
                    break;
                }
            }
        }

        if cancel.is_cancelled() {
            break;
        }
        tracing::warn!("Binance WS disconnected, reconnecting in 5s");
        tokio::time::sleep(Duration::from_secs(5)).await;
    }

    tracing::info!("Binance feed task stopped");
}

/// Convert a REST depth snapshot into a [`BookUpdate`] with `is_snapshot = true`.
fn snapshot_to_book_update(snapshot: &BinanceDepthSnapshot, symbol: &str) -> BookUpdate {
    let bids: Vec<(Price, Quantity)> = snapshot
        .bids
        .iter()
        .map(|pair| {
            let p: f64 = pair[0].parse().unwrap_or(0.0);
            let q: f64 = pair[1].parse().unwrap_or(0.0);
            (Price::from(p), Quantity::from(q))
        })
        .collect();

    let asks: Vec<(Price, Quantity)> = snapshot
        .asks
        .iter()
        .map(|pair| {
            let p: f64 = pair[0].parse().unwrap_or(0.0);
            let q: f64 = pair[1].parse().unwrap_or(0.0);
            (Price::from(p), Quantity::from(q))
        })
        .collect();

    BookUpdate {
        exchange: Exchange::Binance,
        symbol: Symbol::new(symbol),
        timestamp: Timestamp::now(),
        bids,
        asks,
        is_snapshot: true,
    }
}

// ────────────────────────────────────────────────────────────────────
// Bybit feed
// ────────────────────────────────────────────────────────────────────

/// Run the Bybit market data feed.
///
/// Uses [`BybitWsClient`] with auto-reconnect, bridges tokio mpsc channels
/// to the crossbeam `md_tx` channel.
pub async fn run_bybit_feed(
    config: ExchangeConfig,
    reconnect: WsReconnectConfig,
    symbols: Vec<String>,
    md_tx: Sender<MarketDataEvent>,
    state: Arc<SharedState>,
    paper_executor: Option<Arc<PaperExecutor>>,
    cancel: CancellationToken,
) {
    let bybit_config = BybitConfig {
        testnet: config.testnet,
        ws_url: Some(config.ws_url),
        reconnect,
    };

    let client = BybitWsClient::new(bybit_config, symbols);
    let (book_tx, mut book_rx) = tokio::sync::mpsc::channel::<BookUpdate>(4096);
    let (trade_tx, mut trade_rx) = tokio::sync::mpsc::channel::<Trade>(4096);

    // Spawn the Bybit WS client (handles reconnection internally)
    let client_cancel = cancel.clone();
    tokio::spawn(async move {
        tokio::select! {
            result = client.run(book_tx, trade_tx) => {
                if let Err(e) = result {
                    tracing::error!(error = %e, "Bybit WS client error");
                }
            }
            _ = client_cancel.cancelled() => {
                tracing::info!("Bybit WS client cancelled");
            }
        }
    });

    // Bridge book updates: apply to OrderBook, update executor + mid prices
    let book_md_tx = md_tx.clone();
    let book_state = state.clone();
    let book_paper = paper_executor.clone();
    let book_cancel = cancel.clone();
    tokio::spawn(async move {
        let mut books: HashMap<(Exchange, Symbol), OrderBook> = HashMap::new();
        loop {
            tokio::select! {
                update = book_rx.recv() => {
                    match update {
                        Some(book_update) => {
                            apply_book_update(
                                &mut books,
                                &book_update,
                                &book_md_tx,
                                &book_state,
                                &book_paper,
                            );
                        }
                        None => break,
                    }
                }
                _ = book_cancel.cancelled() => break,
            }
        }
        tracing::info!("Bybit book bridge stopped");
    });

    // Bridge trades to md_tx
    let trade_md_tx = md_tx;
    let trade_cancel = cancel;
    tokio::spawn(async move {
        loop {
            tokio::select! {
                trade = trade_rx.recv() => {
                    match trade {
                        Some(t) => {
                            let _ = trade_md_tx.send(MarketDataEvent::Trade(t));
                        }
                        None => break,
                    }
                }
                _ = trade_cancel.cancelled() => break,
            }
        }
        tracing::info!("Bybit trade bridge stopped");
    });
}

// ────────────────────────────────────────────────────────────────────
// Shared helpers
// ────────────────────────────────────────────────────────────────────

/// Apply a book update to the local OrderBook, update mid price + executor,
/// and forward to the strategy thread.
fn apply_book_update(
    books: &mut HashMap<(Exchange, Symbol), OrderBook>,
    book_update: &BookUpdate,
    md_tx: &Sender<MarketDataEvent>,
    state: &SharedState,
    paper_executor: &Option<Arc<PaperExecutor>>,
) {
    let key = (book_update.exchange, book_update.symbol.clone());
    let book = books
        .entry(key.clone())
        .or_insert_with(|| OrderBook::new(book_update.exchange, book_update.symbol.clone()));

    if book_update.is_snapshot {
        let bids: Vec<_> = book_update.bids.iter().map(|(p, q)| (*p, *q)).collect();
        let asks: Vec<_> = book_update.asks.iter().map(|(p, q)| (*p, *q)).collect();
        book.apply_snapshot(&bids, &asks, 1);
    } else {
        let uid = FEED_BOOK_SEQ.fetch_add(1, Ordering::Relaxed);
        let _ = book.apply_update(book_update, uid);
    }

    // Update mid price in shared state
    if let (Some(bid), Some(ask)) = (book.best_bid(), book.best_ask()) {
        let mid = Price::from((bid.price.to_f64() + ask.price.to_f64()) / 2.0);
        state.mid_prices.insert(key, mid);
    }

    // Update paper executor for fill matching
    if let Some(pe) = paper_executor {
        pe.update_market_data(book);
    }

    // Forward to strategy thread
    let _ = md_tx.send(MarketDataEvent::BookUpdate {
        exchange: book_update.exchange,
        symbol: book_update.symbol.clone(),
        book_update: book_update.clone(),
    });
}

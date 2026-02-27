//! Event loop: strategy thread + fill processor + action processor.
//!
//! The strategy runs on a dedicated OS thread (not a tokio task) to avoid
//! jitter from the async runtime. Communication with the async world happens
//! through crossbeam channels.

use std::collections::HashMap;
use std::sync::Arc;

use crossbeam::channel::{Receiver, Sender, TryRecvError};
use tokio_util::sync::CancellationToken;

use cm_core::types::*;
use cm_execution::gateway::{ExchangeGateway, NewOrder};
use cm_market_data::orderbook::OrderBook;
use cm_oms::order::{Order, OrderStatus};
use cm_risk::pipeline::RiskContext;
use cm_strategy::context::OrderAction;
use cm_strategy::traits::{Fill, Strategy};
use cm_strategy::TradingContext;

use crate::engine::SharedState;
use crate::paper_executor::RawFill;

/// Simple monotonic counter for book update IDs.
static BOOK_UPDATE_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Events delivered to the strategy thread.
#[derive(Debug)]
pub enum MarketDataEvent {
    /// L2 order book update.
    BookUpdate {
        exchange: Exchange,
        symbol: Symbol,
        book_update: BookUpdate,
    },
    /// Individual trade.
    Trade(Trade),
    /// Periodic timer tick.
    Timer(Timestamp),
}

// ────────────────────────────────────────────────────────────────────
// Strategy loop — dedicated OS thread
// ────────────────────────────────────────────────────────────────────

/// Strategy loop — runs on a dedicated OS thread.
///
/// Drains fills, blocks on market data events, calls strategy callbacks,
/// and sends resulting actions to the action processor.
pub fn strategy_loop(
    mut strategy: Box<dyn Strategy>,
    state: Arc<SharedState>,
    md_rx: Receiver<MarketDataEvent>,
    action_tx: Sender<Vec<OrderAction>>,
    fill_rx: Receiver<Fill>,
    cancel: CancellationToken,
) {
    let mut books: HashMap<(Exchange, Symbol), OrderBook> = HashMap::new();

    tracing::info!(strategy = strategy.name(), "strategy thread started");

    loop {
        if cancel.is_cancelled() {
            break;
        }

        // 1. Drain fills non-blocking
        let mut fills = Vec::new();
        loop {
            match fill_rx.try_recv() {
                Ok(fill) => fills.push(fill),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    tracing::warn!("fill channel disconnected");
                    return;
                }
            }
        }

        // Deliver fills to strategy
        if !fills.is_empty() {
            let positions = state.position_tracker.all_positions();
            let open_orders = state.order_manager.get_open_orders();
            for fill in &fills {
                let mut ctx =
                    TradingContext::new(positions.clone(), open_orders.clone(), fill.timestamp);
                strategy.on_fill(&mut ctx, fill);
                let actions = ctx.drain_actions();
                if !actions.is_empty() {
                    let _ = action_tx.send(actions);
                }
            }
        }

        // 2. Block on next market data event (with timeout for cancellation check)
        let event = match md_rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(event) => event,
            Err(crossbeam::channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                tracing::warn!("market data channel disconnected");
                break;
            }
        };

        let positions = state.position_tracker.all_positions();
        let open_orders = state.order_manager.get_open_orders();

        match event {
            MarketDataEvent::BookUpdate {
                exchange,
                symbol,
                book_update,
            } => {
                let key = (exchange, symbol.clone());
                let book = books
                    .entry(key)
                    .or_insert_with(|| OrderBook::new(exchange, symbol.clone()));

                if book_update.is_snapshot {
                    let bids: Vec<_> = book_update.bids.iter().map(|(p, q)| (*p, *q)).collect();
                    let asks: Vec<_> = book_update.asks.iter().map(|(p, q)| (*p, *q)).collect();
                    book.apply_snapshot(&bids, &asks, 1);
                } else {
                    let uid =
                        BOOK_UPDATE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let _ = book.apply_update(&book_update, uid);
                }

                let mut ctx = TradingContext::new(positions, open_orders, book_update.timestamp);
                strategy.on_book_update(&mut ctx, book);
                let actions = ctx.drain_actions();
                if !actions.is_empty() {
                    let _ = action_tx.send(actions);
                }
            }
            MarketDataEvent::Trade(trade) => {
                let mut ctx = TradingContext::new(positions, open_orders, trade.timestamp);
                strategy.on_trade(&mut ctx, &trade);
                let actions = ctx.drain_actions();
                if !actions.is_empty() {
                    let _ = action_tx.send(actions);
                }
            }
            MarketDataEvent::Timer(ts) => {
                let mut ctx = TradingContext::new(positions, open_orders, ts);
                strategy.on_timer(&mut ctx, ts);
                let actions = ctx.drain_actions();
                if !actions.is_empty() {
                    let _ = action_tx.send(actions);
                }
            }
        }
    }

    tracing::info!("strategy thread stopped");
}

// ────────────────────────────────────────────────────────────────────
// Fill processor — tokio task
// ────────────────────────────────────────────────────────────────────

/// Fill processor — resolves `RawFill` (from executor) into `Fill` with the
/// correct `OrderId`, updates OMS and PositionTracker, then forwards to
/// strategy thread.
pub async fn fill_processor(
    state: Arc<SharedState>,
    raw_fill_rx: Receiver<RawFill>,
    fill_tx: Sender<Fill>,
    cancel: CancellationToken,
) {
    tracing::info!("fill processor started");

    loop {
        if cancel.is_cancelled() {
            break;
        }

        let raw = {
            let rx = raw_fill_rx.clone();
            tokio::task::spawn_blocking(move || {
                rx.recv_timeout(std::time::Duration::from_millis(100))
            })
            .await
        };

        let raw = match raw {
            Ok(Ok(r)) => r,
            Ok(Err(_)) => continue,
            Err(_) => break,
        };

        // Resolve internal OrderId from client_order_id
        let order_id = match state.order_id_map.get(&raw.client_order_id) {
            Some(entry) => *entry.value(),
            None => {
                tracing::warn!(
                    client_order_id = %raw.client_order_id,
                    "fill for unknown client_order_id — dropping"
                );
                continue;
            }
        };

        // Determine if this is a full fill
        let is_full = state
            .order_manager
            .get_order(&order_id)
            .map(|o| {
                let remaining = o.quantity.to_f64() - o.filled_quantity.to_f64();
                raw.quantity.to_f64() >= remaining - 1e-12
            })
            .unwrap_or(true);

        // Update OMS
        if let Err(e) = state
            .order_manager
            .on_fill(order_id, raw.price, raw.quantity, is_full)
        {
            tracing::error!(order_id = %order_id, error = %e, "OMS on_fill failed");
        }

        // Update PositionTracker
        state.position_tracker.on_fill(
            raw.exchange,
            raw.symbol.clone(),
            raw.side,
            raw.price,
            raw.quantity,
        );

        // Remove pending if fully filled
        if is_full {
            state.position_tracker.remove_pending(&order_id);
        }

        // Forward resolved Fill to strategy
        let fill = Fill {
            order_id,
            exchange: raw.exchange,
            symbol: raw.symbol,
            side: raw.side,
            price: raw.price,
            quantity: raw.quantity,
            timestamp: raw.timestamp,
            is_maker: raw.is_maker,
        };

        tracing::debug!(
            order_id = %order_id,
            side = ?fill.side,
            price = %fill.price,
            qty = %fill.quantity,
            maker = fill.is_maker,
            "fill processed"
        );

        let _ = fill_tx.send(fill);
    }

    tracing::info!("fill processor stopped");
}

// ────────────────────────────────────────────────────────────────────
// Action processor — tokio task
// ────────────────────────────────────────────────────────────────────

/// Action processor — runs as a tokio task.
///
/// Reads order actions from the strategy thread, runs risk checks, registers
/// with the OMS, and dispatches to the executor.
pub async fn action_processor(
    state: Arc<SharedState>,
    executor: Arc<dyn ExchangeGateway>,
    action_rx: Receiver<Vec<OrderAction>>,
    cancel: CancellationToken,
) {
    tracing::info!("action processor started");

    loop {
        if cancel.is_cancelled() {
            break;
        }

        // Bridge from crossbeam to tokio
        let actions = {
            let rx = action_rx.clone();
            tokio::task::spawn_blocking(move || {
                rx.recv_timeout(std::time::Duration::from_millis(100))
            })
            .await
        };

        let actions = match actions {
            Ok(Ok(actions)) => actions,
            Ok(Err(_)) => continue,
            Err(_) => break,
        };

        for action in actions {
            match action {
                OrderAction::Submit {
                    exchange,
                    symbol,
                    side,
                    order_type,
                    price,
                    quantity,
                } => {
                    // Check circuit breaker
                    if !state.circuit_breaker.is_trading_enabled() {
                        tracing::warn!("order rejected: circuit breaker active");
                        continue;
                    }

                    // Generate IDs
                    let order_id = state.next_oid();
                    let client_order_id = state.fill_dedup.next_client_order_id();

                    // Register mapping for fill resolution
                    state.order_id_map.insert(client_order_id.clone(), order_id);

                    // Create OMS order
                    let order = Order {
                        id: order_id,
                        client_order_id: client_order_id.clone(),
                        exchange_order_id: None,
                        exchange,
                        symbol: symbol.clone(),
                        side,
                        order_type,
                        price,
                        quantity,
                        filled_quantity: Quantity::zero(quantity.scale()),
                        status: OrderStatus::New,
                        created_at: Timestamp::now(),
                        updated_at: Timestamp::now(),
                    };

                    // Risk check
                    let mid_price = state
                        .mid_prices
                        .get(&(exchange, symbol.clone()))
                        .map(|e| *e.value());
                    let risk_ctx = RiskContext {
                        position_tracker: &state.position_tracker,
                        current_mid_price: mid_price,
                        daily_pnl: state.daily_pnl(),
                        open_order_count: state.order_manager.get_open_orders().len(),
                    };

                    if let Err(reject) = state.risk_pipeline.check_order(&order, &risk_ctx) {
                        tracing::warn!(
                            order_id = %order_id,
                            reason = %reject,
                            "order rejected by risk pipeline"
                        );
                        state.order_id_map.remove(&client_order_id);
                        continue;
                    }

                    // Register in OMS
                    if let Err(e) = state.order_manager.submit(order) {
                        tracing::error!(error = %e, "failed to submit order to OMS");
                        state.order_id_map.remove(&client_order_id);
                        continue;
                    }
                    if let Err(e) = state.order_manager.on_sent(order_id) {
                        tracing::error!(error = %e, "failed to transition order to Sent");
                        continue;
                    }

                    // Add pending position
                    state.position_tracker.add_pending(order_id, side, quantity);

                    // Dispatch to executor
                    let new_order = NewOrder {
                        exchange,
                        symbol: symbol.clone(),
                        side,
                        order_type,
                        price,
                        quantity,
                        client_order_id,
                    };

                    let om = state.order_manager.clone();
                    let pt = state.position_tracker.clone();
                    let exec = executor.clone();
                    tokio::spawn(async move {
                        match exec.place_order(&new_order).await {
                            Ok(ack) => {
                                tracing::debug!(
                                    order_id = %order_id,
                                    exchange_id = %ack.exchange_order_id,
                                    "order acknowledged"
                                );
                                let _ = om.on_ack(order_id, ExchangeOrderId(ack.exchange_order_id));
                            }
                            Err(e) => {
                                tracing::error!(
                                    order_id = %order_id,
                                    error = %e,
                                    "order placement failed"
                                );
                                let _ = om.on_reject(order_id, e.to_string());
                                pt.remove_pending(&order_id);
                            }
                        }
                    });
                }

                OrderAction::Cancel { order_id } => {
                    if let Some(order) = state.order_manager.get_order(&order_id) {
                        if let Some(eid) = &order.exchange_order_id {
                            let exec = executor.clone();
                            let om = state.order_manager.clone();
                            let pt = state.position_tracker.clone();
                            let exchange = order.exchange;
                            let symbol = order.symbol.0.clone();
                            let eid_str = eid.0.clone();
                            let oid = order_id;
                            tokio::spawn(async move {
                                match exec.cancel_order(exchange, &symbol, &eid_str).await {
                                    Ok(_) => {
                                        let _ = om.on_cancel_ack(oid);
                                        pt.remove_pending(&oid);
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            order_id = %oid,
                                            error = %e,
                                            "cancel failed"
                                        );
                                    }
                                }
                            });
                        }
                    }
                }

                OrderAction::CancelAll { exchange, symbol } => {
                    let open_orders = state.order_manager.get_open_orders();
                    for order in open_orders {
                        let matches_exchange = exchange.is_none_or(|e| order.exchange == e);
                        let matches_symbol = symbol.as_ref().is_none_or(|s| order.symbol == *s);

                        if matches_exchange && matches_symbol {
                            if let Some(eid) = &order.exchange_order_id {
                                let exec = executor.clone();
                                let om = state.order_manager.clone();
                                let pt = state.position_tracker.clone();
                                let exch = order.exchange;
                                let sym = order.symbol.0.clone();
                                let eid_str = eid.0.clone();
                                let oid = order.id;
                                tokio::spawn(async move {
                                    match exec.cancel_order(exch, &sym, &eid_str).await {
                                        Ok(_) => {
                                            let _ = om.on_cancel_ack(oid);
                                            pt.remove_pending(&oid);
                                        }
                                        Err(e) => {
                                            tracing::error!(
                                                order_id = %oid,
                                                error = %e,
                                                "cancel_all: cancel failed"
                                            );
                                        }
                                    }
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    tracing::info!("action processor stopped");
}

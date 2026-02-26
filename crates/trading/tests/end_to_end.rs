//! End-to-end integration test for the paper trading loop.
//!
//! Wires up all components manually (no network), feeds a synthetic book
//! update, verifies the strategy produces orders, the PaperExecutor fills
//! them, and positions update correctly.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crossbeam::channel;
use dashmap::DashMap;

use cm_core::config::PaperConfig;
use cm_core::types::*;
use cm_execution::gateway::{ExchangeGateway, NewOrder};
use cm_market_data::orderbook::OrderBook;
use cm_oms::order::{Order, OrderStatus};
use cm_oms::{FillDeduplicator, OrderManager, PositionTracker};

// ── Bring in the trading crate internals via the binary ────────────
// We test the paper executor and fill processing logic directly.

/// Simplified end-to-end test: manually drive the components without the
/// full TradingEngine (no HTTP server, no OS threads).
#[tokio::test]
async fn test_paper_trading_loop_end_to_end() {
    // ── 1. Build shared state ────────────────────────────────────
    let position_tracker = PositionTracker::new();
    let order_manager = OrderManager::new();
    let fill_dedup = FillDeduplicator::new("test".to_string());
    let order_id_map: DashMap<String, OrderId> = DashMap::new();
    let next_order_id = AtomicU64::new(1);

    let paper_config = PaperConfig {
        latency_ms: 0,
        maker_fee: -0.0001,
        taker_fee: 0.0004,
        max_fill_fraction: 1.0,
    };

    // ── 2. Create PaperExecutor with raw fill channel ────────────
    let (raw_fill_tx, raw_fill_rx) =
        channel::unbounded::<cm_trading::paper_executor::RawFill>();
    let executor = cm_trading::paper_executor::PaperExecutor::new(
        paper_config,
        raw_fill_tx,
    );

    // Set up market data: book with bid=49900, ask=50100
    let mut book = OrderBook::new(Exchange::Binance, Symbol::new("BTCUSDT"));
    book.apply_snapshot(
        &[(Price::from(49900.0), Quantity::from(10.0))],
        &[(Price::from(50100.0), Quantity::from(10.0))],
        1,
    );
    executor.update_market_data(&book);

    // ── 3. Simulate action processor: submit a market buy ────────
    let order_id = OrderId(next_order_id.fetch_add(1, Ordering::Relaxed));
    let client_order_id = fill_dedup.next_client_order_id();

    // Register in order_id_map
    order_id_map.insert(client_order_id.clone(), order_id);

    // Create and register OMS order
    let oms_order = Order {
        id: order_id,
        client_order_id: client_order_id.clone(),
        exchange_order_id: None,
        exchange: Exchange::Binance,
        symbol: Symbol::new("BTCUSDT"),
        side: Side::Buy,
        order_type: OrderType::Market,
        price: Price::from(50100.0),
        quantity: Quantity::from(0.1),
        filled_quantity: Quantity::zero(8),
        status: OrderStatus::New,
        created_at: Timestamp::now(),
        updated_at: Timestamp::now(),
    };

    order_manager.submit(oms_order).unwrap();
    order_manager.on_sent(order_id).unwrap();
    position_tracker.add_pending(order_id, Side::Buy, Quantity::from(0.1));

    // Place order through executor
    let new_order = NewOrder {
        exchange: Exchange::Binance,
        symbol: Symbol::new("BTCUSDT"),
        side: Side::Buy,
        order_type: OrderType::Market,
        price: Price::from(50100.0),
        quantity: Quantity::from(0.1),
        client_order_id: client_order_id.clone(),
    };

    let ack = executor.place_order(&new_order).await.unwrap();
    assert!(ack.exchange_order_id.starts_with("PAPER-"));

    // Ack the order in OMS
    order_manager
        .on_ack(order_id, ExchangeOrderId(ack.exchange_order_id))
        .unwrap();

    // ── 4. Verify raw fill was produced ──────────────────────────
    let raw_fill = raw_fill_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("expected raw fill from executor");
    assert_eq!(raw_fill.client_order_id, client_order_id);
    assert_eq!(raw_fill.side, Side::Buy);
    assert!(!raw_fill.is_maker); // Market order → taker

    // ── 5. Simulate fill processor: resolve OrderId, update OMS/positions
    let resolved_order_id = *order_id_map
        .get(&raw_fill.client_order_id)
        .expect("client_order_id should be in map")
        .value();
    assert_eq!(resolved_order_id, order_id);

    // Determine full fill
    let oms_order = order_manager.get_order(&resolved_order_id).unwrap();
    let remaining = oms_order.quantity.to_f64() - oms_order.filled_quantity.to_f64();
    let is_full = raw_fill.quantity.to_f64() >= remaining - 1e-12;
    assert!(is_full, "market order should be fully filled");

    // Update OMS
    order_manager
        .on_fill(resolved_order_id, raw_fill.price, raw_fill.quantity, is_full)
        .unwrap();

    // Update PositionTracker
    position_tracker.on_fill(
        raw_fill.exchange,
        raw_fill.symbol.clone(),
        raw_fill.side,
        raw_fill.price,
        raw_fill.quantity,
    );
    position_tracker.remove_pending(&resolved_order_id);

    // ── 6. Verify final state ────────────────────────────────────

    // OMS: order should be Filled
    let final_order = order_manager.get_order(&order_id).unwrap();
    assert!(
        final_order.status.is_terminal(),
        "order should be terminal after full fill"
    );

    // PositionTracker: should have a long position
    let positions = position_tracker.all_positions();
    assert_eq!(positions.len(), 1, "should have exactly 1 position");
    let pos = &positions[0];
    assert_eq!(pos.exchange, Exchange::Binance);
    assert_eq!(pos.symbol, Symbol::new("BTCUSDT"));
    assert!(
        pos.net_quantity.to_f64() > 0.09,
        "should have ~0.1 BTC long position, got {}",
        pos.net_quantity.to_f64()
    );
    assert_eq!(pos.fill_count, 1);

    // Open orders should be empty
    let open = order_manager.get_open_orders();
    assert!(open.is_empty(), "no open orders after full fill");

    // ── 7. Now test a resting limit order + book update fill ─────
    let order_id2 = OrderId(next_order_id.fetch_add(1, Ordering::Relaxed));
    let client_order_id2 = fill_dedup.next_client_order_id();
    order_id_map.insert(client_order_id2.clone(), order_id2);

    let oms_order2 = Order {
        id: order_id2,
        client_order_id: client_order_id2.clone(),
        exchange_order_id: None,
        exchange: Exchange::Binance,
        symbol: Symbol::new("BTCUSDT"),
        side: Side::Sell,
        order_type: OrderType::Limit,
        price: Price::from(51000.0), // Way above best bid → should rest
        quantity: Quantity::from(0.05),
        filled_quantity: Quantity::zero(8),
        status: OrderStatus::New,
        created_at: Timestamp::now(),
        updated_at: Timestamp::now(),
    };

    order_manager.submit(oms_order2).unwrap();
    order_manager.on_sent(order_id2).unwrap();
    position_tracker.add_pending(order_id2, Side::Sell, Quantity::from(0.05));

    let new_order2 = NewOrder {
        exchange: Exchange::Binance,
        symbol: Symbol::new("BTCUSDT"),
        side: Side::Sell,
        order_type: OrderType::Limit,
        price: Price::from(51000.0),
        quantity: Quantity::from(0.05),
        client_order_id: client_order_id2.clone(),
    };

    let ack2 = executor.place_order(&new_order2).await.unwrap();
    order_manager
        .on_ack(order_id2, ExchangeOrderId(ack2.exchange_order_id))
        .unwrap();

    // No immediate fill expected (price doesn't cross)
    assert!(
        raw_fill_rx.try_recv().is_err(),
        "limit order should rest, not fill immediately"
    );

    // Now move the book: bid jumps to 51000 → crosses our resting ask
    let mut book2 = OrderBook::new(Exchange::Binance, Symbol::new("BTCUSDT"));
    book2.apply_snapshot(
        &[(Price::from(51000.0), Quantity::from(10.0))],
        &[(Price::from(51100.0), Quantity::from(10.0))],
        2,
    );
    executor.update_market_data(&book2);

    // Should get a maker fill for the resting sell
    let raw_fill2 = raw_fill_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("expected fill after book crossed resting order");
    assert_eq!(raw_fill2.client_order_id, client_order_id2);
    assert_eq!(raw_fill2.side, Side::Sell);
    assert!(raw_fill2.is_maker, "resting order should be maker fill");

    // Process the fill
    let resolved_id2 = *order_id_map.get(&raw_fill2.client_order_id).unwrap().value();
    order_manager
        .on_fill(resolved_id2, raw_fill2.price, raw_fill2.quantity, true)
        .unwrap();
    position_tracker.on_fill(
        raw_fill2.exchange,
        raw_fill2.symbol.clone(),
        raw_fill2.side,
        raw_fill2.price,
        raw_fill2.quantity,
    );
    position_tracker.remove_pending(&resolved_id2);

    // Verify: position reduced (bought 0.1, sold 0.05 → net 0.05 long)
    let final_positions = position_tracker.all_positions();
    assert_eq!(final_positions.len(), 1);
    let final_pos = &final_positions[0];
    let net = final_pos.net_quantity.to_f64();
    assert!(
        (net - 0.05).abs() < 0.001,
        "expected ~0.05 net position, got {}",
        net
    );
    assert_eq!(final_pos.fill_count, 2);

    // Realized PnL: sold 0.05 at 51000 that was bought at 50100
    // PnL = (51000 - 50100) * 0.05 = 45.0
    let pnl = final_pos.realized_pnl.to_f64();
    assert!(
        (pnl - 45.0).abs() < 1.0,
        "expected ~45 USD realized PnL, got {}",
        pnl
    );
}

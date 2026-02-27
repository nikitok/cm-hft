//! Replay test harness for running strategies against recorded market data.
//!
//! Loads JSONL.gz files produced by `cm-record`, replays events through
//! an OrderBook + SimExchange + Strategy pipeline, and tracks PnL/positions.

use std::io::{BufRead, BufReader};

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use serde::Deserialize;

use cm_core::types::*;
use cm_market_data::orderbook::OrderBook;
use cm_strategy::traits::{Fill, StrategyParams};
use cm_strategy::{default_registry, OrderAction, Strategy, TradingContext};

// Re-use the SimExchange from pybridge (it's pub).
// Since pybridge depends on pyo3, we re-implement the sim exchange logic inline
// using the same pattern. This avoids pulling in pyo3 as a test dependency.

/// A recorded market data event (matches the JSONL format from cm-record).
#[derive(Debug, Deserialize)]
pub struct RecordedEvent {
    pub ts_ns: u64,
    pub kind: String,
    pub data: serde_json::Value,
}

/// Deserialized replay event.
#[derive(Debug)]
pub enum ReplayEvent {
    Book(BookUpdate),
    Trade(Trade),
}

/// Result of a replay run.
#[derive(Debug)]
pub struct ReplayResult {
    pub total_pnl: f64,
    pub fill_count: usize,
    pub partial_fill_count: usize,
    pub queue_volume_missed: f64,
    pub order_count: usize,
    pub max_position: f64,
    /// Peak notional exposure in USD (max |position| * mark_price at that moment).
    pub peak_notional: f64,
    /// Peak margin required assuming cross-margin at given leverage.
    /// Calculated as peak_notional / leverage.
    pub peak_margin_1x: f64,
    pub pnl_series: Vec<f64>,
    pub fee_total: f64,
}

/// Load events from multiple gzipped JSONL files, concatenated in file order.
///
/// Files should be passed in chronological order (sorted by name).
/// Events within each file are in receive order. Across files, we assign
/// a globally monotonic sequence number.
pub fn load_events_multi(paths: &[String]) -> Result<Vec<(u64, ReplayEvent)>> {
    let mut all_events = Vec::new();
    let mut global_seq: u64 = 0;
    for path in paths {
        let events = load_events(path)?;
        for (_, event) in events {
            all_events.push((global_seq, event));
            global_seq += 1;
        }
    }
    Ok(all_events)
}

/// Find all timestamped data files for a symbol in the given directory.
/// Matches pattern: bybit_{symbol}_YYYY-MM-DD_HH:MM.jsonl.gz
pub fn find_series_files(dir: &str, symbol: &str) -> Vec<String> {
    let prefix = format!("bybit_{}_", symbol.to_lowercase());
    let suffix = ".jsonl.gz";

    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(&prefix) && name.ends_with(suffix) {
                // Check it's a timestamped file (YYYY-MM-DD_HH:MM pattern after prefix)
                let middle = &name[prefix.len()..name.len() - suffix.len()];
                // Timestamped files have format: 2026-02-26_22:00 (16 chars)
                if middle.len() == 16
                    && middle.chars().nth(4) == Some('-')
                    && middle.chars().nth(10) == Some('_')
                {
                    files.push(entry.path().to_string_lossy().to_string());
                }
            }
        }
    }
    files.sort();
    files
}

/// Load events from a JSONL file (plain or gzipped), preserving file order.
///
/// Events are NOT sorted by `ts_ns` because book and trade timestamps may
/// use different clock bases (exchange-relative vs UNIX epoch). File order
/// corresponds to receive order, which naturally interleaves both streams.
pub fn load_events(path: &str) -> Result<Vec<(u64, ReplayEvent)>> {
    let file = std::fs::File::open(path).with_context(|| format!("failed to open {}", path))?;

    // Support both .jsonl.gz and plain .jsonl files.
    let reader: Box<dyn BufRead> = if path.ends_with(".gz") {
        Box::new(BufReader::new(GzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let mut events = Vec::new();
    let mut seq: u64 = 0;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let recorded: RecordedEvent =
            serde_json::from_str(&line).with_context(|| format!("failed to parse JSONL line"))?;

        let event = match recorded.kind.as_str() {
            "book" => {
                let book: BookUpdate = serde_json::from_value(recorded.data)?;
                ReplayEvent::Book(book)
            }
            "trade" => {
                let trade: Trade = serde_json::from_value(recorded.data)?;
                ReplayEvent::Trade(trade)
            }
            other => {
                anyhow::bail!("unknown event kind: {}", other);
            }
        };

        // Use monotonic sequence number as the event timestamp.
        // The original ts_ns has inconsistent clock bases across event types.
        events.push((seq, event));
        seq += 1;
    }

    Ok(events)
}

/// Replay test harness that runs a strategy against recorded market data.
///
/// Uses a simplified SimExchange (inline) matching the pybridge pattern.
pub struct ReplayTestHarness {
    strategy: Box<dyn Strategy>,
    book: OrderBook,
    exchange: Exchange,
    symbol: Symbol,
    // Position tracking (same as pybridge/lib.rs).
    net_position: f64,
    avg_entry: f64,
    realized_pnl: f64,
    fill_count: usize,
    partial_fill_count: usize,
    queue_volume_missed: f64,
    order_count: usize,
    max_position: f64,
    peak_notional: f64,
    pnl_series: Vec<f64>,
    // Inline sim exchange state.
    sim_bids: Vec<SimOrder>,
    sim_asks: Vec<SimOrder>,
    next_order_id: u64,
    sim_config: SimConfig,
    fee_total: f64,
    /// Fills accumulated from Trade events, delivered on next Book event.
    pending_fills: Vec<Fill>,
    /// Nanosecond timestamp of the last event (for latency activation).
    last_event_ts_ns: u64,
}

#[derive(Debug, Clone)]
struct SimOrder {
    id: OrderId,
    side: Side,
    price: Price,
    original_qty: f64,
    remaining: f64,
    /// L2 volume ahead of us in queue when order was placed.
    queue_ahead: f64,
    /// Event timestamp when order was placed (for latency modeling).
    placed_ts: u64,
    /// Whether the order is active (false if latency hasn't elapsed yet).
    active: bool,
}

/// Simulation configuration for realistic backtesting.
#[derive(Debug, Clone)]
pub struct SimConfig {
    /// Maker fee in basis points. Negative = rebate (e.g., -0.25 for Bybit -0.025%).
    pub maker_fee_bps: f64,
    /// If true, track L2 queue position for realistic fill simulation.
    pub queue_model: bool,
    /// If true, allow partial fills based on available trade volume.
    pub partial_fills: bool,
    /// Order placement latency in nanoseconds. 0 = instant activation.
    pub latency_ns: u64,
    /// Fill probability multiplier. 1.0 = standard, <1.0 = pessimistic.
    pub fill_probability: f64,
    /// If true (legacy mode only), fills only when price crosses THROUGH our level.
    pub strict_crossing: bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            maker_fee_bps: -0.25,
            queue_model: true,
            partial_fills: true,
            latency_ns: 0,
            fill_probability: 1.0,
            strict_crossing: true,
        }
    }
}

impl SimConfig {
    /// Legacy configuration matching the old book-based matching behavior.
    pub fn legacy() -> Self {
        Self {
            maker_fee_bps: -0.25,
            queue_model: false,
            partial_fills: false,
            latency_ns: 0,
            fill_probability: 1.0,
            strict_crossing: true,
        }
    }
}

impl ReplayTestHarness {
    /// Create a new harness with a strategy from the default registry.
    pub fn new(
        strategy_name: &str,
        params: StrategyParams,
        exchange: Exchange,
        symbol: &str,
    ) -> Self {
        Self::with_config(
            strategy_name,
            params,
            exchange,
            symbol,
            SimConfig::default(),
        )
    }

    /// Create a new harness with a custom simulation configuration.
    pub fn with_config(
        strategy_name: &str,
        params: StrategyParams,
        exchange: Exchange,
        symbol: &str,
        sim_config: SimConfig,
    ) -> Self {
        let registry = default_registry();
        let strategy = registry
            .create(strategy_name, &params)
            .unwrap_or_else(|| panic!("unknown strategy: {}", strategy_name));

        Self {
            strategy,
            book: OrderBook::new(exchange, Symbol::new(symbol)),
            exchange,
            symbol: Symbol::new(symbol),
            net_position: 0.0,
            avg_entry: 0.0,
            realized_pnl: 0.0,
            fill_count: 0,
            partial_fill_count: 0,
            queue_volume_missed: 0.0,
            order_count: 0,
            max_position: 0.0,
            peak_notional: 0.0,
            pnl_series: Vec::new(),
            sim_bids: Vec::new(),
            sim_asks: Vec::new(),
            next_order_id: 1,
            sim_config,
            fee_total: 0.0,
            pending_fills: Vec::new(),
            last_event_ts_ns: 0,
        }
    }

    /// Run the replay loop over the given events. Returns the result.
    pub fn run(&mut self, events: &[(u64, ReplayEvent)]) -> ReplayResult {
        for (ts, event) in events {
            match event {
                ReplayEvent::Book(update) => {
                    // 1. Apply snapshot to order book.
                    self.book.apply_snapshot(&update.bids, &update.asks, *ts);

                    // 2. Update queue positions from book changes (min-cap).
                    if self.sim_config.queue_model {
                        self.update_queues_from_book();
                    }

                    // 3. Activate latent orders (check latency).
                    self.activate_pending_orders(*ts);

                    // 4. Generate fills.
                    let fills = if self.sim_config.queue_model {
                        // Drain pending fills from trade events.
                        std::mem::take(&mut self.pending_fills)
                    } else {
                        // Legacy: book-based matching.
                        self.match_orders_legacy()
                    };

                    // 5. Process fills: update position tracking.
                    for fill in &fills {
                        self.process_fill(fill);
                    }

                    // 6. Call strategy.on_book_update() with current position context.
                    let ctx_position = self.build_position_vec();
                    let mut ctx = TradingContext::new(ctx_position, vec![], Timestamp(*ts));
                    self.strategy.on_book_update(&mut ctx, &self.book);

                    // 7. Process strategy order actions.
                    self.process_actions(ctx.drain_actions(), *ts);

                    // 8. Deliver fills to strategy.
                    for fill in &fills {
                        let mut fill_ctx = TradingContext::new(vec![], vec![], Timestamp(*ts));
                        self.strategy.on_fill(&mut fill_ctx, fill);
                    }

                    // 9. Track mark-to-market PnL.
                    let mark_price = self.book.mid_price().map(|p| p.to_f64()).unwrap_or(0.0);
                    let unrealized = self.net_position * (mark_price - self.avg_entry);
                    self.pnl_series.push(self.realized_pnl + unrealized);
                }
                ReplayEvent::Trade(trade) => {
                    // Update event timestamp for latency tracking.
                    if trade.timestamp.0 > 0 {
                        self.last_event_ts_ns = trade.timestamp.0;
                    }

                    // Activate latent orders.
                    self.activate_pending_orders(*ts);

                    // Trade-based fill matching (queue model).
                    if self.sim_config.queue_model {
                        self.process_trade_for_fills(trade, *ts);
                    }

                    // Deliver trade to strategy.
                    let mut ctx = TradingContext::new(vec![], vec![], Timestamp(*ts));
                    self.strategy.on_trade(&mut ctx, trade);

                    // Process any actions from on_trade.
                    self.process_actions(ctx.drain_actions(), *ts);
                }
            }
        }

        ReplayResult {
            total_pnl: self.realized_pnl,
            fill_count: self.fill_count,
            partial_fill_count: self.partial_fill_count,
            queue_volume_missed: self.queue_volume_missed,
            order_count: self.order_count,
            max_position: self.max_position,
            peak_notional: self.peak_notional,
            peak_margin_1x: self.peak_notional, // 1x leverage = full notional
            pnl_series: self.pnl_series.clone(),
            fee_total: self.fee_total,
        }
    }

    /// Process a single fill: update position, PnL, fees, and stats.
    fn process_fill(&mut self, fill: &Fill) {
        let fill_price = fill.price.to_f64();
        let fill_qty = fill.quantity.to_f64();
        let signed_qty = match fill.side {
            Side::Buy => fill_qty,
            Side::Sell => -fill_qty,
        };

        let fill_pnl =
            calculate_fill_pnl(self.net_position, self.avg_entry, fill_price, signed_qty);
        let (new_pos, new_avg) =
            update_position(self.net_position, self.avg_entry, fill_price, signed_qty);

        self.net_position = new_pos;
        self.avg_entry = new_avg;
        self.realized_pnl += fill_pnl;

        // Fee tracking.
        let notional = fill_price * fill_qty;
        let fee_impact = -notional * self.sim_config.maker_fee_bps / 10_000.0;
        self.fee_total += fee_impact;
        self.realized_pnl += fee_impact;

        self.fill_count += 1;

        if self.net_position.abs() > self.max_position {
            self.max_position = self.net_position.abs();
        }

        // Track peak notional exposure.
        let mark = self
            .book
            .mid_price()
            .map(|p| p.to_f64())
            .unwrap_or(fill_price);
        let peak_notional = self.net_position.abs() * mark;
        if peak_notional > self.peak_notional {
            self.peak_notional = peak_notional;
        }
    }

    /// Build the position vec for strategy context.
    fn build_position_vec(&self) -> Vec<cm_oms::Position> {
        if self.net_position.abs() > 1e-12 {
            vec![cm_oms::Position {
                exchange: self.exchange,
                symbol: self.symbol.clone(),
                net_quantity: Quantity::from(self.net_position),
                avg_entry_price: Price::from(self.avg_entry),
                realized_pnl: Price::from(self.realized_pnl),
                fill_count: self.fill_count as u64,
            }]
        } else {
            vec![]
        }
    }

    /// Process order actions from strategy.
    fn process_actions(&mut self, actions: Vec<OrderAction>, ts: u64) {
        for action in actions {
            match action {
                OrderAction::Submit {
                    side,
                    price,
                    quantity,
                    ..
                } => {
                    self.order_count += 1;
                    let id = OrderId(self.next_order_id);
                    self.next_order_id += 1;
                    let qty_f64 = quantity.to_f64();
                    let queue_ahead = if self.sim_config.queue_model {
                        estimate_queue_ahead(&self.book, side, price)
                    } else {
                        0.0
                    };
                    let active = self.sim_config.latency_ns == 0;
                    let order = SimOrder {
                        id,
                        side,
                        price,
                        original_qty: qty_f64,
                        remaining: qty_f64,
                        queue_ahead,
                        placed_ts: ts,
                        active,
                    };
                    match side {
                        Side::Buy => self.sim_bids.push(order),
                        Side::Sell => self.sim_asks.push(order),
                    }
                }
                OrderAction::Cancel { order_id } => {
                    self.sim_bids.retain(|o| o.id != order_id);
                    self.sim_asks.retain(|o| o.id != order_id);
                }
                OrderAction::CancelAll { .. } => {
                    self.sim_bids.clear();
                    self.sim_asks.clear();
                }
            }
        }
    }

    /// Activate orders whose latency period has elapsed.
    fn activate_pending_orders(&mut self, seq_ts: u64) {
        if self.sim_config.latency_ns == 0 {
            return;
        }
        let latency = self.sim_config.latency_ns;
        for order in self.sim_bids.iter_mut().chain(self.sim_asks.iter_mut()) {
            if !order.active {
                // Use last_event_ts_ns for real nanosecond latency if available,
                // otherwise fall back to sequence number (always activates instantly).
                let elapsed = if self.last_event_ts_ns > 0 && order.placed_ts > 1_000_000 {
                    // Both look like real nanosecond timestamps.
                    self.last_event_ts_ns.saturating_sub(order.placed_ts)
                } else {
                    // Sequence numbers — can't model real latency, activate after 1 seq step.
                    let seq_diff = seq_ts.saturating_sub(order.placed_ts);
                    if seq_diff > 0 {
                        latency
                    } else {
                        0
                    }
                };
                if elapsed >= latency {
                    order.active = true;
                    // Re-estimate queue position from current book state.
                    if self.sim_config.queue_model {
                        order.queue_ahead =
                            estimate_queue_ahead(&self.book, order.side, order.price);
                    }
                }
            }
        }
    }

    /// Update queue positions from book changes (min-cap: queue can only shrink).
    fn update_queues_from_book(&mut self) {
        for order in self.sim_bids.iter_mut().chain(self.sim_asks.iter_mut()) {
            if !order.active {
                continue;
            }
            let current_vol = volume_at_price(&self.book, order.side, order.price);
            // Queue can only shrink (cancellations ahead), never grow (new participants behind us).
            if current_vol < order.queue_ahead {
                order.queue_ahead = current_vol;
            }
        }
    }

    /// Trade-based fill matching with queue depletion.
    fn process_trade_for_fills(&mut self, trade: &Trade, ts: u64) {
        let trade_price = trade.price.to_f64();
        let mut trade_remaining = trade.quantity.to_f64();

        // Sell trade → fills our BUY orders. Buy trade → fills our SELL orders.
        let orders = match trade.side {
            Side::Sell => &mut self.sim_bids,
            Side::Buy => &mut self.sim_asks,
        };

        let fill_prob = self.sim_config.fill_probability;
        let partial_fills = self.sim_config.partial_fills;
        let exchange = self.exchange;
        let symbol = self.symbol.clone();

        let mut fills = Vec::new();
        let mut to_remove = Vec::new();

        // Sort by price priority: bids descending, asks ascending (already maintained by push order,
        // but sort for correctness with multiple price levels).
        // Actually, orders are at specific prices from strategy; process in FIFO insertion order
        // which is natural Vec order.

        for (i, order) in orders.iter_mut().enumerate() {
            if !order.active || trade_remaining <= 1e-15 {
                continue;
            }

            let order_price = order.price.to_f64();

            // Determine if trade matches our order.
            let is_sweep;
            let is_at_level;
            match trade.side {
                Side::Sell => {
                    // Sell trade: fills our bids.
                    // Sweep: trade price < our bid (sold through our level).
                    // At level: trade price == our bid.
                    is_sweep = trade_price < order_price;
                    is_at_level = (trade_price - order_price).abs() < 1e-12;
                }
                Side::Buy => {
                    // Buy trade: fills our asks.
                    // Sweep: trade price > our ask (bought through our level).
                    // At level: trade price == our ask.
                    is_sweep = trade_price > order_price;
                    is_at_level = (trade_price - order_price).abs() < 1e-12;
                }
            }

            if is_sweep {
                // Level was consumed — fill immediately regardless of queue.
                let fill_qty = order.remaining;
                fills.push(Fill {
                    order_id: order.id,
                    exchange,
                    symbol: symbol.clone(),
                    side: order.side,
                    price: order.price,
                    quantity: Quantity::from(fill_qty),
                    timestamp: Timestamp(ts),
                    is_maker: true,
                });
                to_remove.push(i);
                // Don't reduce trade_remaining for sweeps — the trade consumed the whole level.
            } else if is_at_level {
                // Trade at our level — deplete queue first.
                if order.queue_ahead > 1e-15 {
                    let depleted = trade_remaining.min(order.queue_ahead);
                    order.queue_ahead -= depleted;
                    trade_remaining -= depleted;
                    // Track missed volume.
                    self.queue_volume_missed += depleted;
                }

                if order.queue_ahead <= 1e-15 && trade_remaining > 1e-15 {
                    // Queue exhausted — we can get filled.
                    let available = trade_remaining * fill_prob;
                    let fill_qty = if partial_fills {
                        available.min(order.remaining)
                    } else {
                        // All-or-nothing.
                        if available >= order.remaining {
                            order.remaining
                        } else {
                            continue;
                        }
                    };

                    if fill_qty > 1e-15 {
                        let is_partial = (fill_qty - order.remaining).abs() > 1e-15;
                        if is_partial {
                            self.partial_fill_count += 1;
                        }
                        order.remaining -= fill_qty;
                        trade_remaining -= fill_qty;

                        fills.push(Fill {
                            order_id: order.id,
                            exchange,
                            symbol: symbol.clone(),
                            side: order.side,
                            price: order.price,
                            quantity: Quantity::from(fill_qty),
                            timestamp: Timestamp(ts),
                            is_maker: true,
                        });

                        if order.remaining <= 1e-15 {
                            to_remove.push(i);
                        }
                    }
                }
            }
            // else: no match (trade price doesn't reach our level).
        }

        // Remove fully filled orders (reverse order to preserve indices).
        for i in to_remove.into_iter().rev() {
            orders.remove(i);
        }

        // Accumulate fills for delivery on next book event.
        self.pending_fills.extend(fills);
    }

    /// Legacy: Match resting orders against the current book state (book-based matching).
    fn match_orders_legacy(&mut self) -> Vec<Fill> {
        let mut fills = Vec::new();
        let best_ask = self.book.best_ask();
        let best_bid = self.book.best_bid();

        // Match resting bids against best ask.
        if let Some(ask_level) = best_ask {
            let mut to_remove = Vec::new();
            for (i, order) in self.sim_bids.iter_mut().enumerate() {
                if !order.active {
                    continue;
                }
                let bid_fills = if self.sim_config.strict_crossing {
                    order.price > ask_level.price
                } else {
                    order.price >= ask_level.price
                };
                if bid_fills {
                    fills.push(Fill {
                        order_id: order.id,
                        exchange: self.exchange,
                        symbol: self.symbol.clone(),
                        side: Side::Buy,
                        price: order.price,
                        quantity: Quantity::from(order.remaining),
                        timestamp: Timestamp(0),
                        is_maker: true,
                    });
                    to_remove.push(i);
                }
            }
            for i in to_remove.into_iter().rev() {
                self.sim_bids.remove(i);
            }
        }

        // Match resting asks against best bid.
        if let Some(bid_level) = best_bid {
            let mut to_remove = Vec::new();
            for (i, order) in self.sim_asks.iter_mut().enumerate() {
                if !order.active {
                    continue;
                }
                let ask_fills = if self.sim_config.strict_crossing {
                    order.price < bid_level.price
                } else {
                    order.price <= bid_level.price
                };
                if ask_fills {
                    fills.push(Fill {
                        order_id: order.id,
                        exchange: self.exchange,
                        symbol: self.symbol.clone(),
                        side: Side::Sell,
                        price: order.price,
                        quantity: Quantity::from(order.remaining),
                        timestamp: Timestamp(0),
                        is_maker: true,
                    });
                    to_remove.push(i);
                }
            }
            for i in to_remove.into_iter().rev() {
                self.sim_asks.remove(i);
            }
        }

        fills
    }
}

/// Estimate the L2 volume ahead of us at a given price level.
fn estimate_queue_ahead(book: &OrderBook, side: Side, price: Price) -> f64 {
    let levels = match side {
        Side::Buy => book.bid_depth(50),
        Side::Sell => book.ask_depth(50),
    };
    for level in levels {
        if level.price == price {
            return level.quantity.to_f64();
        }
    }
    0.0 // Level doesn't exist — we're first.
}

/// Get current visible volume at a specific price level.
fn volume_at_price(book: &OrderBook, side: Side, price: Price) -> f64 {
    let levels = match side {
        Side::Buy => book.bid_depth(50),
        Side::Sell => book.ask_depth(50),
    };
    for level in levels {
        if level.price == price {
            return level.quantity.to_f64();
        }
    }
    0.0 // Level gone — queue exhausted.
}

// Position tracking functions (same as pybridge/lib.rs).

fn calculate_fill_pnl(net_position: f64, avg_entry: f64, fill_price: f64, signed_qty: f64) -> f64 {
    if net_position.abs() < 1e-12 {
        return 0.0;
    }
    let is_reducing =
        (net_position > 0.0 && signed_qty < 0.0) || (net_position < 0.0 && signed_qty > 0.0);
    if !is_reducing {
        return 0.0;
    }
    let reduce_qty = signed_qty.abs().min(net_position.abs());
    if net_position > 0.0 {
        (fill_price - avg_entry) * reduce_qty
    } else {
        (avg_entry - fill_price) * reduce_qty
    }
}

fn update_position(
    net_position: f64,
    avg_entry: f64,
    fill_price: f64,
    signed_qty: f64,
) -> (f64, f64) {
    let new_position = net_position + signed_qty;
    if net_position.abs() < 1e-12 {
        return (new_position, fill_price);
    }
    let same_direction =
        (net_position > 0.0 && signed_qty > 0.0) || (net_position < 0.0 && signed_qty < 0.0);
    if same_direction {
        let old_cost = avg_entry * net_position.abs();
        let new_cost = fill_price * signed_qty.abs();
        let total_qty = net_position.abs() + signed_qty.abs();
        let new_avg = if total_qty.abs() > 1e-12 {
            (old_cost + new_cost) / total_qty
        } else {
            fill_price
        };
        (new_position, new_avg)
    } else if new_position.abs() < 1e-12 {
        (0.0, 0.0)
    } else if (net_position > 0.0 && new_position < 0.0)
        || (net_position < 0.0 && new_position > 0.0)
    {
        (new_position, fill_price)
    } else {
        (new_position, avg_entry)
    }
}

/// Helper to get default strategy params for testing.
pub fn default_params() -> StrategyParams {
    StrategyParams {
        params: serde_json::json!({}),
    }
}

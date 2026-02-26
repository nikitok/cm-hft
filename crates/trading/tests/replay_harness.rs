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

/// Load events from a gzipped JSONL file, sorted by timestamp.
pub fn load_events(path: &str) -> Result<Vec<(u64, ReplayEvent)>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open {}", path))?;
    let decoder = GzDecoder::new(file);
    let reader = BufReader::new(decoder);

    let mut events = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let recorded: RecordedEvent = serde_json::from_str(&line)
            .with_context(|| format!("failed to parse JSONL line"))?;

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

        events.push((recorded.ts_ns, event));
    }

    events.sort_by_key(|(ts, _)| *ts);
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
}

#[derive(Debug, Clone)]
struct SimOrder {
    id: OrderId,
    #[allow(dead_code)]
    side: Side,
    price: Price,
    remaining: f64,
}

/// Simulation configuration for realistic backtesting.
#[derive(Debug, Clone)]
pub struct SimConfig {
    /// Maker fee in basis points. Negative = rebate (e.g., -0.25 for Bybit -0.025%).
    pub maker_fee_bps: f64,
    /// If true, fills only when price crosses THROUGH our level (not just touches).
    pub strict_crossing: bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            maker_fee_bps: -0.25,
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
        Self::with_config(strategy_name, params, exchange, symbol, SimConfig::default())
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
            order_count: 0,
            max_position: 0.0,
            peak_notional: 0.0,
            pnl_series: Vec::new(),
            sim_bids: Vec::new(),
            sim_asks: Vec::new(),
            next_order_id: 1,
            sim_config,
            fee_total: 0.0,
        }
    }

    /// Run the replay loop over the given events. Returns the result.
    pub fn run(&mut self, events: &[(u64, ReplayEvent)]) -> ReplayResult {
        for (ts, event) in events {
            match event {
                ReplayEvent::Book(update) => {
                    // 1. Apply snapshot to order book.
                    self.book.apply_snapshot(
                        &update.bids,
                        &update.asks,
                        *ts,
                    );

                    // 2. Match resting orders against book (simplified sim exchange).
                    let fills = self.match_orders();

                    // 3. Process fills: update position tracking.
                    for fill in &fills {
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

                        // Fee tracking
                        let notional = fill_price * fill_qty;
                        let fee_impact = -notional * self.sim_config.maker_fee_bps / 10_000.0;
                        self.fee_total += fee_impact;
                        self.realized_pnl += fee_impact;

                        self.fill_count += 1;

                        if self.net_position.abs() > self.max_position {
                            self.max_position = self.net_position.abs();
                        }

                        // Track peak notional exposure.
                        let mark = self.book.mid_price().map(|p| p.to_f64()).unwrap_or(fill_price);
                        let notional = self.net_position.abs() * mark;
                        if notional > self.peak_notional {
                            self.peak_notional = notional;
                        }
                    }

                    // 4. Call strategy.on_book_update() with current position context.
                    let ctx_position = if self.net_position.abs() > 1e-12 {
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
                    };

                    let mut ctx = TradingContext::new(ctx_position, vec![], Timestamp(*ts));
                    self.strategy.on_book_update(&mut ctx, &self.book);

                    // 5. Process strategy order actions.
                    let actions = ctx.drain_actions();
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
                                let order = SimOrder {
                                    id,
                                    side,
                                    price,
                                    remaining: quantity.to_f64(),
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

                    // 6. Deliver fills to strategy.
                    for fill in &fills {
                        let mut fill_ctx = TradingContext::new(vec![], vec![], Timestamp(*ts));
                        self.strategy.on_fill(&mut fill_ctx, fill);
                    }

                    // 7. Track mark-to-market PnL.
                    let mark_price = self.book.mid_price().map(|p| p.to_f64()).unwrap_or(0.0);
                    let unrealized = self.net_position * (mark_price - self.avg_entry);
                    self.pnl_series.push(self.realized_pnl + unrealized);
                }
                ReplayEvent::Trade(trade) => {
                    let mut ctx = TradingContext::new(vec![], vec![], Timestamp(*ts));
                    self.strategy.on_trade(&mut ctx, trade);
                }
            }
        }

        ReplayResult {
            total_pnl: self.realized_pnl,
            fill_count: self.fill_count,
            order_count: self.order_count,
            max_position: self.max_position,
            peak_notional: self.peak_notional,
            peak_margin_1x: self.peak_notional, // 1x leverage = full notional
            pnl_series: self.pnl_series.clone(),
            fee_total: self.fee_total,
        }
    }

    /// Match resting orders against the current book state.
    fn match_orders(&mut self) -> Vec<Fill> {
        let mut fills = Vec::new();
        let best_ask = self.book.best_ask();
        let best_bid = self.book.best_bid();

        // Match resting bids against best ask.
        if let Some(ask_level) = best_ask {
            let mut to_remove = Vec::new();
            for (i, order) in self.sim_bids.iter_mut().enumerate() {
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

fn update_position(net_position: f64, avg_entry: f64, fill_price: f64, signed_qty: f64) -> (f64, f64) {
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
    } else if (net_position > 0.0 && new_position < 0.0) || (net_position < 0.0 && new_position > 0.0)
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

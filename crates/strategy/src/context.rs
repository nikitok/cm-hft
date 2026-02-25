//! Trading context provided to strategy callbacks.
//!
//! [`TradingContext`] buffers order actions during a callback. Actions are
//! flushed after the strategy callback returns, ensuring the strategy never
//! touches the OMS directly.

use cm_core::types::*;
use cm_oms::{Order, Position};

/// Actions that a strategy can request.
#[derive(Debug, Clone)]
pub enum OrderAction {
    /// Submit a new order.
    Submit {
        /// Target exchange.
        exchange: Exchange,
        /// Trading pair.
        symbol: Symbol,
        /// Order side.
        side: Side,
        /// Order type (limit, market, post-only).
        order_type: OrderType,
        /// Limit price.
        price: Price,
        /// Order quantity.
        quantity: Quantity,
    },
    /// Cancel an existing order.
    Cancel {
        /// Internal order identifier to cancel.
        order_id: OrderId,
    },
    /// Cancel all orders, optionally filtered by exchange and/or symbol.
    CancelAll {
        /// If set, only cancel orders on this exchange.
        exchange: Option<Exchange>,
        /// If set, only cancel orders for this symbol.
        symbol: Option<Symbol>,
    },
}

/// Context provided to strategy callbacks.
///
/// Buffers order actions during a callback; actions are flushed after the
/// strategy callback returns.
pub struct TradingContext {
    /// Buffered order actions (flushed after strategy callback returns).
    actions: Vec<OrderAction>,
    /// Current positions (read-only snapshot).
    positions: Vec<Position>,
    /// Current open orders (read-only snapshot).
    open_orders: Vec<Order>,
    /// Current timestamp.
    pub timestamp: Timestamp,
}

impl TradingContext {
    /// Create a new trading context with position/order snapshots.
    pub fn new(positions: Vec<Position>, open_orders: Vec<Order>, timestamp: Timestamp) -> Self {
        Self {
            actions: Vec::new(),
            positions,
            open_orders,
            timestamp,
        }
    }

    /// Buffer a new order submission.
    pub fn submit_order(
        &mut self,
        exchange: Exchange,
        symbol: Symbol,
        side: Side,
        order_type: OrderType,
        price: Price,
        quantity: Quantity,
    ) {
        self.actions.push(OrderAction::Submit {
            exchange,
            symbol,
            side,
            order_type,
            price,
            quantity,
        });
    }

    /// Buffer a cancel request for a specific order.
    pub fn cancel_order(&mut self, order_id: OrderId) {
        self.actions.push(OrderAction::Cancel { order_id });
    }

    /// Buffer a cancel-all request, optionally filtered by exchange and/or symbol.
    pub fn cancel_all(&mut self, exchange: Option<Exchange>, symbol: Option<Symbol>) {
        self.actions.push(OrderAction::CancelAll { exchange, symbol });
    }

    /// Look up a position by exchange and symbol.
    pub fn get_position(&self, exchange: &Exchange, symbol: &Symbol) -> Option<&Position> {
        self.positions
            .iter()
            .find(|p| p.exchange == *exchange && p.symbol == *symbol)
    }

    /// Get all current open orders.
    pub fn get_open_orders(&self) -> &[Order] {
        &self.open_orders
    }

    /// Convenience: net position quantity as `f64` for strategy math.
    ///
    /// Returns `0.0` if no position exists for the given exchange/symbol.
    pub fn net_position(&self, exchange: &Exchange, symbol: &Symbol) -> f64 {
        self.get_position(exchange, symbol)
            .map(|p| p.net_quantity.to_f64())
            .unwrap_or(0.0)
    }

    /// Drain all buffered actions, returning them and clearing the buffer.
    pub fn drain_actions(&mut self) -> Vec<OrderAction> {
        std::mem::take(&mut self.actions)
    }

    /// Number of buffered actions.
    pub fn action_count(&self) -> usize {
        self.actions.len()
    }
}

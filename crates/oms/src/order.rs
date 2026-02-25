//! Order state machine and lifecycle management.
//!
//! Tracks every order from creation through terminal states (Filled, Cancelled,
//! Rejected) using a deterministic state machine. All transitions emit
//! [`OrderEvent`]s for downstream consumption and journaling.

use cm_core::types::*;

/// Order states in the lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum OrderStatus {
    /// Order created locally, not yet sent to exchange.
    New,
    /// Order sent to exchange, awaiting acknowledgement.
    Sent,
    /// Exchange has acknowledged the order.
    Acked,
    /// Partially filled; some quantity remains open.
    PartialFill,
    /// Fully filled; terminal state.
    Filled,
    /// Cancelled by user or exchange; terminal state.
    Cancelled,
    /// Rejected by exchange; terminal state.
    Rejected,
}

impl OrderStatus {
    /// Returns `true` if this status is terminal (no further transitions).
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Filled | Self::Cancelled | Self::Rejected)
    }
}

/// An order event recording a state transition.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrderEvent {
    /// When the event occurred.
    pub timestamp: Timestamp,
    /// Internal order identifier.
    pub order_id: OrderId,
    /// Client-assigned order identifier.
    pub client_order_id: String,
    /// Type of event.
    pub event_type: OrderEventType,
    /// Exchange the order is on.
    pub exchange: Exchange,
    /// Trading pair.
    pub symbol: Symbol,
    /// Order side.
    pub side: Side,
    /// Price (if applicable).
    pub price: Option<Price>,
    /// Quantity (if applicable).
    pub quantity: Option<Quantity>,
    /// Cumulative filled quantity (if applicable).
    pub filled_quantity: Option<Quantity>,
    /// New status after this event.
    pub status: OrderStatus,
    /// Optional metadata (e.g., rejection reason).
    pub metadata: Option<String>,
}

/// Type of order event.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum OrderEventType {
    /// Order created locally.
    Created,
    /// Order sent to exchange.
    Sent,
    /// Exchange acknowledged the order.
    Acknowledged,
    /// Partial fill received.
    PartiallyFilled,
    /// Full fill received.
    Filled,
    /// Cancel request sent.
    CancelRequested,
    /// Cancel confirmed.
    Cancelled,
    /// Order rejected by exchange.
    Rejected,
    /// Error during processing.
    Error,
}

/// Internal order representation.
#[derive(Debug, Clone)]
pub struct Order {
    /// Internal order identifier.
    pub id: OrderId,
    /// Client-assigned order identifier.
    pub client_order_id: String,
    /// Exchange-assigned order identifier (set after ack).
    pub exchange_order_id: Option<ExchangeOrderId>,
    /// Exchange the order is on.
    pub exchange: Exchange,
    /// Trading pair.
    pub symbol: Symbol,
    /// Order side.
    pub side: Side,
    /// Order type (limit, market, post-only).
    pub order_type: OrderType,
    /// Limit price.
    pub price: Price,
    /// Total order quantity.
    pub quantity: Quantity,
    /// Cumulative filled quantity.
    pub filled_quantity: Quantity,
    /// Current order status.
    pub status: OrderStatus,
    /// When the order was created.
    pub created_at: Timestamp,
    /// When the order was last updated.
    pub updated_at: Timestamp,
}

/// Order state machine errors.
#[derive(Debug, thiserror::Error)]
pub enum OrderError {
    /// Attempted an invalid state transition.
    #[error("invalid transition from {from:?} to {to:?} for order {order_id}")]
    InvalidTransition {
        order_id: OrderId,
        from: OrderStatus,
        to: OrderStatus,
    },
    /// Order not found in the manager.
    #[error("order not found: {0}")]
    NotFound(OrderId),
    /// Duplicate order submission.
    #[error("duplicate order: {0}")]
    Duplicate(OrderId),
}

/// Manages all active orders and their lifecycle transitions.
///
/// Thread-safe: uses `DashMap` for concurrent order access and a `Mutex`
/// for ordered event buffering.
pub struct OrderManager {
    orders: dashmap::DashMap<OrderId, Order>,
    events: parking_lot::Mutex<Vec<OrderEvent>>,
}

impl OrderManager {
    /// Create a new, empty order manager.
    pub fn new() -> Self {
        Self {
            orders: dashmap::DashMap::new(),
            events: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// Submit a new order. The order must have status `New`.
    ///
    /// Returns a `Created` event on success.
    pub fn submit(&self, order: Order) -> Result<OrderEvent, OrderError> {
        if self.orders.contains_key(&order.id) {
            return Err(OrderError::Duplicate(order.id));
        }

        let event = OrderEvent {
            timestamp: Timestamp::now(),
            order_id: order.id,
            client_order_id: order.client_order_id.clone(),
            event_type: OrderEventType::Created,
            exchange: order.exchange,
            symbol: order.symbol.clone(),
            side: order.side,
            price: Some(order.price),
            quantity: Some(order.quantity),
            filled_quantity: Some(order.filled_quantity),
            status: OrderStatus::New,
            metadata: None,
        };

        self.orders.insert(order.id, order);
        self.events.lock().push(event.clone());
        Ok(event)
    }

    /// Record that the order was sent to the exchange. Transition: New -> Sent.
    pub fn on_sent(&self, order_id: OrderId) -> Result<OrderEvent, OrderError> {
        self.transition(order_id, OrderStatus::Sent, |order| {
            Self::validate_transition(order.id, order.status, OrderStatus::Sent)?;
            order.status = OrderStatus::Sent;
            order.updated_at = Timestamp::now();
            Ok(OrderEvent {
                timestamp: order.updated_at,
                order_id: order.id,
                client_order_id: order.client_order_id.clone(),
                event_type: OrderEventType::Sent,
                exchange: order.exchange,
                symbol: order.symbol.clone(),
                side: order.side,
                price: Some(order.price),
                quantity: Some(order.quantity),
                filled_quantity: Some(order.filled_quantity),
                status: OrderStatus::Sent,
                metadata: None,
            })
        })
    }

    /// Record exchange acknowledgement. Transition: Sent -> Acked.
    pub fn on_ack(
        &self,
        order_id: OrderId,
        exchange_order_id: ExchangeOrderId,
    ) -> Result<OrderEvent, OrderError> {
        let eid = exchange_order_id;
        self.transition(order_id, OrderStatus::Acked, |order| {
            Self::validate_transition(order.id, order.status, OrderStatus::Acked)?;
            order.exchange_order_id = Some(eid.clone());
            order.status = OrderStatus::Acked;
            order.updated_at = Timestamp::now();
            Ok(OrderEvent {
                timestamp: order.updated_at,
                order_id: order.id,
                client_order_id: order.client_order_id.clone(),
                event_type: OrderEventType::Acknowledged,
                exchange: order.exchange,
                symbol: order.symbol.clone(),
                side: order.side,
                price: Some(order.price),
                quantity: Some(order.quantity),
                filled_quantity: Some(order.filled_quantity),
                status: OrderStatus::Acked,
                metadata: None,
            })
        })
    }

    /// Record a fill (partial or full). Transition: Acked/PartialFill -> PartialFill/Filled.
    pub fn on_fill(
        &self,
        order_id: OrderId,
        fill_price: Price,
        fill_qty: Quantity,
        is_full: bool,
    ) -> Result<OrderEvent, OrderError> {
        let target = if is_full {
            OrderStatus::Filled
        } else {
            OrderStatus::PartialFill
        };

        self.transition(order_id, target, |order| {
            Self::validate_transition(order.id, order.status, target)?;
            order.filled_quantity = order.filled_quantity + fill_qty;
            order.status = target;
            order.updated_at = Timestamp::now();
            let event_type = if is_full {
                OrderEventType::Filled
            } else {
                OrderEventType::PartiallyFilled
            };
            Ok(OrderEvent {
                timestamp: order.updated_at,
                order_id: order.id,
                client_order_id: order.client_order_id.clone(),
                event_type,
                exchange: order.exchange,
                symbol: order.symbol.clone(),
                side: order.side,
                price: Some(fill_price),
                quantity: Some(fill_qty),
                filled_quantity: Some(order.filled_quantity),
                status: target,
                metadata: None,
            })
        })
    }

    /// Record a cancel acknowledgement. Transition: Acked/PartialFill -> Cancelled.
    pub fn on_cancel_ack(&self, order_id: OrderId) -> Result<OrderEvent, OrderError> {
        self.transition(order_id, OrderStatus::Cancelled, |order| {
            Self::validate_transition(order.id, order.status, OrderStatus::Cancelled)?;
            order.status = OrderStatus::Cancelled;
            order.updated_at = Timestamp::now();
            Ok(OrderEvent {
                timestamp: order.updated_at,
                order_id: order.id,
                client_order_id: order.client_order_id.clone(),
                event_type: OrderEventType::Cancelled,
                exchange: order.exchange,
                symbol: order.symbol.clone(),
                side: order.side,
                price: Some(order.price),
                quantity: Some(order.quantity),
                filled_quantity: Some(order.filled_quantity),
                status: OrderStatus::Cancelled,
                metadata: None,
            })
        })
    }

    /// Record a rejection. Transition: Sent -> Rejected.
    pub fn on_reject(
        &self,
        order_id: OrderId,
        reason: String,
    ) -> Result<OrderEvent, OrderError> {
        let r = reason;
        self.transition(order_id, OrderStatus::Rejected, |order| {
            Self::validate_transition(order.id, order.status, OrderStatus::Rejected)?;
            order.status = OrderStatus::Rejected;
            order.updated_at = Timestamp::now();
            Ok(OrderEvent {
                timestamp: order.updated_at,
                order_id: order.id,
                client_order_id: order.client_order_id.clone(),
                event_type: OrderEventType::Rejected,
                exchange: order.exchange,
                symbol: order.symbol.clone(),
                side: order.side,
                price: Some(order.price),
                quantity: Some(order.quantity),
                filled_quantity: Some(order.filled_quantity),
                status: OrderStatus::Rejected,
                metadata: Some(r.clone()),
            })
        })
    }

    /// Get a clone of an order by its ID.
    pub fn get_order(&self, order_id: &OrderId) -> Option<Order> {
        self.orders.get(order_id).map(|o| o.clone())
    }

    /// Get all non-terminal (open) orders.
    pub fn get_open_orders(&self) -> Vec<Order> {
        self.orders
            .iter()
            .filter(|entry| !entry.value().status.is_terminal())
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Drain the event buffer, returning all events in order.
    pub fn get_events(&self) -> Vec<OrderEvent> {
        let mut events = self.events.lock();
        std::mem::take(&mut *events)
    }

    /// Returns the total number of orders tracked.
    pub fn order_count(&self) -> usize {
        self.orders.len()
    }

    /// Validate that a transition from `from` to `to` is allowed.
    fn validate_transition(
        order_id: OrderId,
        from: OrderStatus,
        to: OrderStatus,
    ) -> Result<(), OrderError> {
        let valid = matches!(
            (from, to),
            (OrderStatus::New, OrderStatus::Sent)
                | (OrderStatus::Sent, OrderStatus::Acked)
                | (OrderStatus::Sent, OrderStatus::Rejected)
                | (OrderStatus::Acked, OrderStatus::PartialFill)
                | (OrderStatus::Acked, OrderStatus::Filled)
                | (OrderStatus::Acked, OrderStatus::Cancelled)
                | (OrderStatus::PartialFill, OrderStatus::PartialFill)
                | (OrderStatus::PartialFill, OrderStatus::Filled)
                | (OrderStatus::PartialFill, OrderStatus::Cancelled)
        );

        if valid {
            Ok(())
        } else {
            tracing::error!(
                order_id = %order_id,
                from = ?from,
                to = ?to,
                "invalid order state transition"
            );
            Err(OrderError::InvalidTransition {
                order_id,
                from,
                to,
            })
        }
    }

    /// Helper to look up an order, apply a mutation, and record the event.
    fn transition<F>(
        &self,
        order_id: OrderId,
        _target: OrderStatus,
        mutate: F,
    ) -> Result<OrderEvent, OrderError>
    where
        F: FnOnce(&mut Order) -> Result<OrderEvent, OrderError>,
    {
        let mut entry = self
            .orders
            .get_mut(&order_id)
            .ok_or(OrderError::NotFound(order_id))?;
        let event = mutate(entry.value_mut())?;
        self.events.lock().push(event.clone());
        Ok(event)
    }
}

impl Default for OrderManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a test order with the given ID.
    fn make_order(id: u64) -> Order {
        Order {
            id: OrderId(id),
            client_order_id: format!("cm_test_{:06}", id),
            exchange_order_id: None,
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            side: Side::Buy,
            order_type: OrderType::Limit,
            price: Price::new(5000000, 2),
            quantity: Quantity::new(100000000, 8), // 1.0 BTC
            filled_quantity: Quantity::zero(8),
            status: OrderStatus::New,
            created_at: Timestamp::now(),
            updated_at: Timestamp::now(),
        }
    }

    // ── Happy path: New → Sent → Acked → Filled ──

    #[test]
    fn test_happy_path_full_lifecycle() {
        let mgr = OrderManager::new();
        let order = make_order(1);

        let evt = mgr.submit(order).unwrap();
        assert_eq!(evt.status, OrderStatus::New);

        let evt = mgr.on_sent(OrderId(1)).unwrap();
        assert_eq!(evt.status, OrderStatus::Sent);

        let evt = mgr
            .on_ack(OrderId(1), ExchangeOrderId("EX-1".into()))
            .unwrap();
        assert_eq!(evt.status, OrderStatus::Acked);

        let fill_price = Price::new(5000000, 2);
        let fill_qty = Quantity::new(100000000, 8);
        let evt = mgr.on_fill(OrderId(1), fill_price, fill_qty, true).unwrap();
        assert_eq!(evt.status, OrderStatus::Filled);

        let order = mgr.get_order(&OrderId(1)).unwrap();
        assert_eq!(order.status, OrderStatus::Filled);
        assert_eq!(order.filled_quantity, Quantity::new(100000000, 8));
    }

    // ── Partial fills ──

    #[test]
    fn test_partial_fills_accumulate() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        mgr.on_sent(OrderId(1)).unwrap();
        mgr.on_ack(OrderId(1), ExchangeOrderId("EX-1".into()))
            .unwrap();

        let fill_price = Price::new(5000000, 2);
        let partial = Quantity::new(30000000, 8); // 0.3 BTC

        // First partial fill
        let evt = mgr.on_fill(OrderId(1), fill_price, partial, false).unwrap();
        assert_eq!(evt.status, OrderStatus::PartialFill);

        // Second partial fill
        let evt = mgr.on_fill(OrderId(1), fill_price, partial, false).unwrap();
        assert_eq!(evt.status, OrderStatus::PartialFill);

        // Final fill
        let remaining = Quantity::new(40000000, 8); // 0.4 BTC
        let evt = mgr
            .on_fill(OrderId(1), fill_price, remaining, true)
            .unwrap();
        assert_eq!(evt.status, OrderStatus::Filled);

        let order = mgr.get_order(&OrderId(1)).unwrap();
        assert_eq!(order.filled_quantity, Quantity::new(100000000, 8));
        assert_eq!(order.status, OrderStatus::Filled);
    }

    // ── Cancel path ──

    #[test]
    fn test_cancel_from_acked() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        mgr.on_sent(OrderId(1)).unwrap();
        mgr.on_ack(OrderId(1), ExchangeOrderId("EX-1".into()))
            .unwrap();

        let evt = mgr.on_cancel_ack(OrderId(1)).unwrap();
        assert_eq!(evt.status, OrderStatus::Cancelled);

        let order = mgr.get_order(&OrderId(1)).unwrap();
        assert_eq!(order.status, OrderStatus::Cancelled);
    }

    #[test]
    fn test_cancel_from_partial_fill() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        mgr.on_sent(OrderId(1)).unwrap();
        mgr.on_ack(OrderId(1), ExchangeOrderId("EX-1".into()))
            .unwrap();
        mgr.on_fill(
            OrderId(1),
            Price::new(5000000, 2),
            Quantity::new(30000000, 8),
            false,
        )
        .unwrap();

        let evt = mgr.on_cancel_ack(OrderId(1)).unwrap();
        assert_eq!(evt.status, OrderStatus::Cancelled);
    }

    // ── Reject path ──

    #[test]
    fn test_reject_from_sent() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        mgr.on_sent(OrderId(1)).unwrap();

        let evt = mgr
            .on_reject(OrderId(1), "insufficient margin".into())
            .unwrap();
        assert_eq!(evt.status, OrderStatus::Rejected);
        assert_eq!(evt.metadata, Some("insufficient margin".into()));
    }

    // ── Invalid transitions ──

    #[test]
    fn test_invalid_new_to_acked() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        let err = mgr
            .on_ack(OrderId(1), ExchangeOrderId("EX-1".into()))
            .unwrap_err();
        assert!(matches!(
            err,
            OrderError::InvalidTransition {
                from: OrderStatus::New,
                to: OrderStatus::Acked,
                ..
            }
        ));
    }

    #[test]
    fn test_invalid_new_to_filled() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        let err = mgr
            .on_fill(
                OrderId(1),
                Price::new(5000000, 2),
                Quantity::new(100000000, 8),
                true,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            OrderError::InvalidTransition {
                from: OrderStatus::New,
                to: OrderStatus::Filled,
                ..
            }
        ));
    }

    #[test]
    fn test_invalid_new_to_cancelled() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        let err = mgr.on_cancel_ack(OrderId(1)).unwrap_err();
        assert!(matches!(
            err,
            OrderError::InvalidTransition {
                from: OrderStatus::New,
                to: OrderStatus::Cancelled,
                ..
            }
        ));
    }

    #[test]
    fn test_invalid_sent_to_filled() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        mgr.on_sent(OrderId(1)).unwrap();
        let err = mgr
            .on_fill(
                OrderId(1),
                Price::new(5000000, 2),
                Quantity::new(100000000, 8),
                true,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            OrderError::InvalidTransition {
                from: OrderStatus::Sent,
                to: OrderStatus::Filled,
                ..
            }
        ));
    }

    #[test]
    fn test_invalid_sent_to_cancelled() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        mgr.on_sent(OrderId(1)).unwrap();
        let err = mgr.on_cancel_ack(OrderId(1)).unwrap_err();
        assert!(matches!(
            err,
            OrderError::InvalidTransition {
                from: OrderStatus::Sent,
                to: OrderStatus::Cancelled,
                ..
            }
        ));
    }

    #[test]
    fn test_invalid_filled_to_cancelled() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        mgr.on_sent(OrderId(1)).unwrap();
        mgr.on_ack(OrderId(1), ExchangeOrderId("EX-1".into()))
            .unwrap();
        mgr.on_fill(
            OrderId(1),
            Price::new(5000000, 2),
            Quantity::new(100000000, 8),
            true,
        )
        .unwrap();

        let err = mgr.on_cancel_ack(OrderId(1)).unwrap_err();
        assert!(matches!(
            err,
            OrderError::InvalidTransition {
                from: OrderStatus::Filled,
                to: OrderStatus::Cancelled,
                ..
            }
        ));
    }

    #[test]
    fn test_invalid_filled_to_partial_fill() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        mgr.on_sent(OrderId(1)).unwrap();
        mgr.on_ack(OrderId(1), ExchangeOrderId("EX-1".into()))
            .unwrap();
        mgr.on_fill(
            OrderId(1),
            Price::new(5000000, 2),
            Quantity::new(100000000, 8),
            true,
        )
        .unwrap();

        let err = mgr
            .on_fill(
                OrderId(1),
                Price::new(5000000, 2),
                Quantity::new(10000000, 8),
                false,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            OrderError::InvalidTransition {
                from: OrderStatus::Filled,
                to: OrderStatus::PartialFill,
                ..
            }
        ));
    }

    #[test]
    fn test_invalid_cancelled_to_filled() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        mgr.on_sent(OrderId(1)).unwrap();
        mgr.on_ack(OrderId(1), ExchangeOrderId("EX-1".into()))
            .unwrap();
        mgr.on_cancel_ack(OrderId(1)).unwrap();

        let err = mgr
            .on_fill(
                OrderId(1),
                Price::new(5000000, 2),
                Quantity::new(100000000, 8),
                true,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            OrderError::InvalidTransition {
                from: OrderStatus::Cancelled,
                to: OrderStatus::Filled,
                ..
            }
        ));
    }

    #[test]
    fn test_invalid_rejected_to_acked() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        mgr.on_sent(OrderId(1)).unwrap();
        mgr.on_reject(OrderId(1), "bad".into()).unwrap();

        let err = mgr
            .on_ack(OrderId(1), ExchangeOrderId("EX-1".into()))
            .unwrap_err();
        assert!(matches!(
            err,
            OrderError::InvalidTransition {
                from: OrderStatus::Rejected,
                to: OrderStatus::Acked,
                ..
            }
        ));
    }

    #[test]
    fn test_invalid_acked_to_rejected() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        mgr.on_sent(OrderId(1)).unwrap();
        mgr.on_ack(OrderId(1), ExchangeOrderId("EX-1".into()))
            .unwrap();

        let err = mgr.on_reject(OrderId(1), "too late".into()).unwrap_err();
        assert!(matches!(
            err,
            OrderError::InvalidTransition {
                from: OrderStatus::Acked,
                to: OrderStatus::Rejected,
                ..
            }
        ));
    }

    // ── Duplicate submission ──

    #[test]
    fn test_duplicate_order_submission() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        let err = mgr.submit(make_order(1)).unwrap_err();
        assert!(matches!(err, OrderError::Duplicate(OrderId(1))));
    }

    // ── Order not found ──

    #[test]
    fn test_order_not_found_on_sent() {
        let mgr = OrderManager::new();
        let err = mgr.on_sent(OrderId(999)).unwrap_err();
        assert!(matches!(err, OrderError::NotFound(OrderId(999))));
    }

    #[test]
    fn test_order_not_found_on_ack() {
        let mgr = OrderManager::new();
        let err = mgr
            .on_ack(OrderId(999), ExchangeOrderId("EX-1".into()))
            .unwrap_err();
        assert!(matches!(err, OrderError::NotFound(OrderId(999))));
    }

    #[test]
    fn test_order_not_found_on_fill() {
        let mgr = OrderManager::new();
        let err = mgr
            .on_fill(
                OrderId(999),
                Price::new(5000000, 2),
                Quantity::new(100000000, 8),
                true,
            )
            .unwrap_err();
        assert!(matches!(err, OrderError::NotFound(OrderId(999))));
    }

    #[test]
    fn test_order_not_found_on_cancel() {
        let mgr = OrderManager::new();
        let err = mgr.on_cancel_ack(OrderId(999)).unwrap_err();
        assert!(matches!(err, OrderError::NotFound(OrderId(999))));
    }

    #[test]
    fn test_order_not_found_on_reject() {
        let mgr = OrderManager::new();
        let err = mgr.on_reject(OrderId(999), "reason".into()).unwrap_err();
        assert!(matches!(err, OrderError::NotFound(OrderId(999))));
    }

    // ── Event emission ──

    #[test]
    fn test_events_emitted_on_each_transition() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        mgr.on_sent(OrderId(1)).unwrap();
        mgr.on_ack(OrderId(1), ExchangeOrderId("EX-1".into()))
            .unwrap();
        mgr.on_fill(
            OrderId(1),
            Price::new(5000000, 2),
            Quantity::new(100000000, 8),
            true,
        )
        .unwrap();

        let events = mgr.get_events();
        assert_eq!(events.len(), 4);
        assert!(matches!(events[0].event_type, OrderEventType::Created));
        assert!(matches!(events[1].event_type, OrderEventType::Sent));
        assert!(matches!(events[2].event_type, OrderEventType::Acknowledged));
        assert!(matches!(events[3].event_type, OrderEventType::Filled));
    }

    #[test]
    fn test_get_events_drains_buffer() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();

        let events = mgr.get_events();
        assert_eq!(events.len(), 1);

        let events = mgr.get_events();
        assert!(events.is_empty());
    }

    // ── Open orders ──

    #[test]
    fn test_get_open_orders() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        mgr.submit(make_order(2)).unwrap();
        mgr.submit(make_order(3)).unwrap();

        // Move order 1 to filled (terminal)
        mgr.on_sent(OrderId(1)).unwrap();
        mgr.on_ack(OrderId(1), ExchangeOrderId("EX-1".into()))
            .unwrap();
        mgr.on_fill(
            OrderId(1),
            Price::new(5000000, 2),
            Quantity::new(100000000, 8),
            true,
        )
        .unwrap();

        let open = mgr.get_open_orders();
        assert_eq!(open.len(), 2);
        assert!(open.iter().all(|o| o.id != OrderId(1)));
    }

    #[test]
    fn test_order_count() {
        let mgr = OrderManager::new();
        assert_eq!(mgr.order_count(), 0);
        mgr.submit(make_order(1)).unwrap();
        assert_eq!(mgr.order_count(), 1);
        mgr.submit(make_order(2)).unwrap();
        assert_eq!(mgr.order_count(), 2);
    }

    #[test]
    fn test_get_order_returns_none_for_missing() {
        let mgr = OrderManager::new();
        assert!(mgr.get_order(&OrderId(42)).is_none());
    }

    #[test]
    fn test_exchange_order_id_set_on_ack() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        mgr.on_sent(OrderId(1)).unwrap();
        mgr.on_ack(OrderId(1), ExchangeOrderId("EXCHANGE-123".into()))
            .unwrap();

        let order = mgr.get_order(&OrderId(1)).unwrap();
        assert_eq!(
            order.exchange_order_id,
            Some(ExchangeOrderId("EXCHANGE-123".into()))
        );
    }

    #[test]
    fn test_order_status_is_terminal() {
        assert!(!OrderStatus::New.is_terminal());
        assert!(!OrderStatus::Sent.is_terminal());
        assert!(!OrderStatus::Acked.is_terminal());
        assert!(!OrderStatus::PartialFill.is_terminal());
        assert!(OrderStatus::Filled.is_terminal());
        assert!(OrderStatus::Cancelled.is_terminal());
        assert!(OrderStatus::Rejected.is_terminal());
    }

    // ── Concurrent access ──

    #[test]
    fn test_concurrent_submit() {
        use std::sync::Arc;
        use std::thread;

        let mgr = Arc::new(OrderManager::new());
        let mut handles = vec![];

        for i in 0..100 {
            let mgr = mgr.clone();
            handles.push(thread::spawn(move || {
                mgr.submit(make_order(i)).unwrap();
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(mgr.order_count(), 100);
        let events = mgr.get_events();
        assert_eq!(events.len(), 100);
    }

    #[test]
    fn test_concurrent_transitions() {
        use std::sync::Arc;
        use std::thread;

        let mgr = Arc::new(OrderManager::new());

        // Submit orders sequentially
        for i in 0..50 {
            mgr.submit(make_order(i)).unwrap();
        }

        // Transition all to Sent concurrently
        let mut handles = vec![];
        for i in 0..50 {
            let mgr = mgr.clone();
            handles.push(thread::spawn(move || {
                mgr.on_sent(OrderId(i)).unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        // Verify all are Sent
        for i in 0..50 {
            let order = mgr.get_order(&OrderId(i)).unwrap();
            assert_eq!(order.status, OrderStatus::Sent);
        }
    }

    #[test]
    fn test_partial_fill_from_acked() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        mgr.on_sent(OrderId(1)).unwrap();
        mgr.on_ack(OrderId(1), ExchangeOrderId("EX-1".into()))
            .unwrap();

        let evt = mgr
            .on_fill(
                OrderId(1),
                Price::new(5000000, 2),
                Quantity::new(50000000, 8),
                false,
            )
            .unwrap();
        assert_eq!(evt.status, OrderStatus::PartialFill);
        assert!(matches!(evt.event_type, OrderEventType::PartiallyFilled));
    }

    #[test]
    fn test_fill_from_acked_directly() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        mgr.on_sent(OrderId(1)).unwrap();
        mgr.on_ack(OrderId(1), ExchangeOrderId("EX-1".into()))
            .unwrap();

        let evt = mgr
            .on_fill(
                OrderId(1),
                Price::new(5000000, 2),
                Quantity::new(100000000, 8),
                true,
            )
            .unwrap();
        assert_eq!(evt.status, OrderStatus::Filled);
        assert!(matches!(evt.event_type, OrderEventType::Filled));
    }

    #[test]
    fn test_invalid_new_to_rejected() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        let err = mgr.on_reject(OrderId(1), "bad".into()).unwrap_err();
        assert!(matches!(
            err,
            OrderError::InvalidTransition {
                from: OrderStatus::New,
                to: OrderStatus::Rejected,
                ..
            }
        ));
    }

    #[test]
    fn test_invalid_new_to_partial_fill() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        let err = mgr
            .on_fill(
                OrderId(1),
                Price::new(5000000, 2),
                Quantity::new(50000000, 8),
                false,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            OrderError::InvalidTransition {
                from: OrderStatus::New,
                to: OrderStatus::PartialFill,
                ..
            }
        ));
    }

    #[test]
    fn test_multiple_orders_independent() {
        let mgr = OrderManager::new();
        mgr.submit(make_order(1)).unwrap();
        mgr.submit(make_order(2)).unwrap();

        mgr.on_sent(OrderId(1)).unwrap();

        // Order 2 should still be New
        let o2 = mgr.get_order(&OrderId(2)).unwrap();
        assert_eq!(o2.status, OrderStatus::New);

        // Order 1 should be Sent
        let o1 = mgr.get_order(&OrderId(1)).unwrap();
        assert_eq!(o1.status, OrderStatus::Sent);
    }
}

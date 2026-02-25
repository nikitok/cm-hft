//! Unified exchange gateway trait for order management.
//!
//! The [`ExchangeGateway`] trait provides a common interface across exchanges,
//! allowing the OMS and strategy layers to submit, cancel, and amend orders
//! without coupling to exchange-specific API details.

use anyhow::Result;
use async_trait::async_trait;

use cm_core::types::{Exchange, OrderType, Price, Quantity, Side, Symbol};

/// A new order to submit to an exchange.
#[derive(Debug)]
pub struct NewOrder {
    /// Target exchange.
    pub exchange: Exchange,
    /// Trading pair symbol.
    pub symbol: Symbol,
    /// Order side (buy or sell).
    pub side: Side,
    /// Order type (limit, market, post-only).
    pub order_type: OrderType,
    /// Order price (ignored for market orders).
    pub price: Price,
    /// Order quantity.
    pub quantity: Quantity,
    /// Client-assigned order ID for tracking.
    pub client_order_id: String,
}

/// Acknowledgment returned after a successful order placement.
#[derive(Debug)]
pub struct OrderAck {
    /// Exchange-assigned order ID.
    pub exchange_order_id: String,
    /// Client-assigned order ID echoed back.
    pub client_order_id: String,
}

/// Acknowledgment returned after a successful order cancellation.
#[derive(Debug)]
pub struct CancelAck {
    /// The cancelled order ID.
    pub order_id: String,
}

/// Acknowledgment returned after a successful order amendment.
#[derive(Debug)]
pub struct AmendAck {
    /// The amended order ID.
    pub order_id: String,
}

/// Unified trait for exchange order operations.
///
/// Implementors handle exchange-specific serialization, signing, rate limiting,
/// and error mapping. All methods are async to support non-blocking I/O.
#[async_trait]
pub trait ExchangeGateway: Send + Sync {
    /// Place a new order on the exchange.
    async fn place_order(&self, order: &NewOrder) -> Result<OrderAck>;

    /// Cancel an existing order.
    async fn cancel_order(
        &self,
        exchange: Exchange,
        symbol: &str,
        order_id: &str,
    ) -> Result<CancelAck>;

    /// Amend an existing order's price and/or quantity.
    async fn amend_order(
        &self,
        exchange: Exchange,
        symbol: &str,
        order_id: &str,
        new_price: Option<Price>,
        new_qty: Option<Quantity>,
    ) -> Result<AmendAck>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that ExchangeGateway can be used as a trait object.
    #[test]
    fn test_gateway_is_object_safe() {
        fn _assert_object_safe(_g: &dyn ExchangeGateway) {}
    }

    #[test]
    fn test_new_order_debug() {
        let order = NewOrder {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            side: Side::Buy,
            order_type: OrderType::Limit,
            price: Price::new(5000000, 2),
            quantity: Quantity::new(100000, 8),
            client_order_id: "test-001".to_string(),
        };
        let debug = format!("{:?}", order);
        assert!(debug.contains("BTCUSDT"));
        assert!(debug.contains("test-001"));
    }

    #[test]
    fn test_order_ack_debug() {
        let ack = OrderAck {
            exchange_order_id: "EX-12345".to_string(),
            client_order_id: "CL-001".to_string(),
        };
        let debug = format!("{:?}", ack);
        assert!(debug.contains("EX-12345"));
    }

    #[test]
    fn test_cancel_ack_debug() {
        let ack = CancelAck {
            order_id: "EX-12345".to_string(),
        };
        assert_eq!(ack.order_id, "EX-12345");
    }

    #[test]
    fn test_amend_ack_debug() {
        let ack = AmendAck {
            order_id: "EX-12345".to_string(),
        };
        assert_eq!(ack.order_id, "EX-12345");
    }
}

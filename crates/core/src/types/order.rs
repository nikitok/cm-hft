//! Order-related types: exchange identifiers, side, order type, and order IDs.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported exchanges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Exchange {
    /// Binance spot and futures.
    Binance,
    /// Bybit unified trading.
    Bybit,
}

impl fmt::Display for Exchange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Exchange::Binance => write!(f, "Binance"),
            Exchange::Bybit => write!(f, "Bybit"),
        }
    }
}

/// Order side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    /// Buy / long.
    Buy,
    /// Sell / short.
    Sell,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Buy => write!(f, "Buy"),
            Side::Sell => write!(f, "Sell"),
        }
    }
}

/// Order type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderType {
    /// Limit order with specified price.
    Limit,
    /// Market order, fills at best available price.
    Market,
    /// Post-only limit order (rejected if it would take liquidity).
    PostOnly,
}

impl fmt::Display for OrderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderType::Limit => write!(f, "Limit"),
            OrderType::Market => write!(f, "Market"),
            OrderType::PostOnly => write!(f, "PostOnly"),
        }
    }
}

/// Internal order identifier (monotonic counter).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrderId(pub u64);

impl fmt::Display for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OID-{}", self.0)
    }
}

/// Exchange-assigned order identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExchangeOrderId(pub String);

impl fmt::Display for ExchangeOrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Trading pair symbol (e.g., "BTCUSDT").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Symbol(pub String);

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Symbol {
    /// Create a new symbol.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exchange_display() {
        assert_eq!(format!("{}", Exchange::Binance), "Binance");
        assert_eq!(format!("{}", Exchange::Bybit), "Bybit");
    }

    #[test]
    fn test_side_display() {
        assert_eq!(format!("{}", Side::Buy), "Buy");
        assert_eq!(format!("{}", Side::Sell), "Sell");
    }

    #[test]
    fn test_order_type_display() {
        assert_eq!(format!("{}", OrderType::Limit), "Limit");
        assert_eq!(format!("{}", OrderType::Market), "Market");
        assert_eq!(format!("{}", OrderType::PostOnly), "PostOnly");
    }

    #[test]
    fn test_order_id_display() {
        assert_eq!(format!("{}", OrderId(42)), "OID-42");
    }

    #[test]
    fn test_exchange_order_id() {
        let eid = ExchangeOrderId("ABC123".to_string());
        assert_eq!(format!("{}", eid), "ABC123");
    }

    #[test]
    fn test_symbol() {
        let s = Symbol::new("BTCUSDT");
        assert_eq!(format!("{}", s), "BTCUSDT");
        assert_eq!(s, Symbol("BTCUSDT".to_string()));
    }

    #[test]
    fn test_exchange_eq_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Exchange::Binance);
        assert!(set.contains(&Exchange::Binance));
        assert!(!set.contains(&Exchange::Bybit));
    }

    #[test]
    fn test_side_eq() {
        assert_eq!(Side::Buy, Side::Buy);
        assert_ne!(Side::Buy, Side::Sell);
    }
}

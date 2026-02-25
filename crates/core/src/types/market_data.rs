//! Normalized market data types shared across exchange implementations.
//!
//! These types represent the canonical internal format for market data. Exchange
//! specific parsers convert wire-format messages into these structures.

use serde::{Deserialize, Serialize};

use super::order::{Exchange, Side, Symbol};
use super::price::Price;
use super::quantity::Quantity;
use super::timestamp::Timestamp;

/// Top-of-book tick with best bid and ask.
///
/// This is the primary input to strategies on each market data update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedTick {
    /// Source exchange.
    pub exchange: Exchange,
    /// Trading pair.
    pub symbol: Symbol,
    /// Best bid price.
    pub bid: Price,
    /// Best ask price.
    pub ask: Price,
    /// Quantity available at the best bid.
    pub bid_qty: Quantity,
    /// Quantity available at the best ask.
    pub ask_qty: Quantity,
    /// Nanosecond timestamp of the event.
    pub timestamp_ns: Timestamp,
}

impl NormalizedTick {
    /// Calculate the mid-price as `(bid + ask) / 2`.
    ///
    /// Uses integer arithmetic with truncation.
    pub fn mid_price(&self) -> Price {
        (self.bid + self.ask) / 2
    }

    /// Calculate the spread as `ask - bid`.
    pub fn spread(&self) -> Price {
        self.ask - self.bid
    }
}

/// L2 order book update (snapshot or incremental delta).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookUpdate {
    /// Source exchange.
    pub exchange: Exchange,
    /// Trading pair.
    pub symbol: Symbol,
    /// Nanosecond timestamp of the event.
    pub timestamp: Timestamp,
    /// Bid levels (price, quantity). Quantity of zero means level removal.
    pub bids: Vec<(Price, Quantity)>,
    /// Ask levels (price, quantity). Quantity of zero means level removal.
    pub asks: Vec<(Price, Quantity)>,
    /// `true` if this is a full snapshot, `false` for an incremental delta.
    pub is_snapshot: bool,
}

/// Individual trade event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Source exchange.
    pub exchange: Exchange,
    /// Trading pair.
    pub symbol: Symbol,
    /// Nanosecond timestamp of the trade.
    pub timestamp: Timestamp,
    /// Trade price.
    pub price: Price,
    /// Trade quantity.
    pub quantity: Quantity,
    /// Taker side (the aggressor).
    pub side: Side,
    /// Exchange-assigned trade identifier.
    pub trade_id: String,
}

/// A single price level in the order book.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BookLevel {
    /// Price at this level.
    pub price: Price,
    /// Aggregate quantity at this level.
    pub quantity: Quantity,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tick() -> NormalizedTick {
        NormalizedTick {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            bid: Price::new(5000000, 2),
            ask: Price::new(5000100, 2),
            bid_qty: Quantity::new(100000, 8),
            ask_qty: Quantity::new(200000, 8),
            timestamp_ns: Timestamp::from_millis(1706000000000),
        }
    }

    #[test]
    fn test_mid_price() {
        let tick = sample_tick();
        let mid = tick.mid_price();
        assert_eq!(mid, Price::new(5000050, 2));
    }

    #[test]
    fn test_spread() {
        let tick = sample_tick();
        let spread = tick.spread();
        assert_eq!(spread, Price::new(100, 2));
    }

    #[test]
    fn test_book_update_snapshot() {
        let update = BookUpdate {
            exchange: Exchange::Bybit,
            symbol: Symbol::new("BTCUSDT"),
            timestamp: Timestamp::from_millis(1706000000000),
            bids: vec![(Price::new(5000000, 2), Quantity::new(100000, 8))],
            asks: vec![(Price::new(5000100, 2), Quantity::new(200000, 8))],
            is_snapshot: true,
        };
        assert!(update.is_snapshot);
        assert_eq!(update.bids.len(), 1);
        assert_eq!(update.asks.len(), 1);
    }

    #[test]
    fn test_book_update_delta_with_removal() {
        let update = BookUpdate {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            timestamp: Timestamp::from_millis(1706000000000),
            bids: vec![(Price::new(4999900, 2), Quantity::zero(8))],
            asks: vec![],
            is_snapshot: false,
        };
        assert!(!update.is_snapshot);
        assert!(update.bids[0].1.is_zero());
    }

    #[test]
    fn test_trade() {
        let trade = Trade {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            timestamp: Timestamp::from_millis(1706000000000),
            price: Price::new(5000050, 2),
            quantity: Quantity::new(10000, 8),
            side: Side::Buy,
            trade_id: "12345".to_string(),
        };
        assert_eq!(trade.side, Side::Buy);
        assert_eq!(trade.trade_id, "12345");
    }

    #[test]
    fn test_book_level() {
        let level = BookLevel {
            price: Price::new(5000000, 2),
            quantity: Quantity::new(100000, 8),
        };
        assert_eq!(level.price, Price::new(5000000, 2));
        assert_eq!(level.quantity, Quantity::new(100000, 8));
    }
}

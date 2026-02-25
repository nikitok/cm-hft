//! Binance-specific wire types for WebSocket and REST API responses.
//!
//! These types match the JSON format emitted by Binance's API and are
//! deserialized directly from the wire.  The [`From`] implementations
//! convert them into the normalized [`cm_core::types`] equivalents.

use cm_core::types::{BookUpdate, Exchange, Price, Quantity, Side, Symbol, Timestamp, Trade};
use serde::Deserialize;

/// Raw Binance depth update from the `@depth@100ms` WebSocket stream.
#[derive(Debug, Deserialize)]
pub struct BinanceDepthUpdate {
    /// Event type (always `"depthUpdate"`).
    #[serde(rename = "e")]
    pub event_type: String,
    /// Event time (milliseconds since epoch).
    #[serde(rename = "E")]
    pub event_time: u64,
    /// Symbol (uppercase, e.g., `"BTCUSDT"`).
    #[serde(rename = "s")]
    pub symbol: String,
    /// First update ID in event.
    #[serde(rename = "U")]
    pub first_update_id: u64,
    /// Final update ID in event.
    #[serde(rename = "u")]
    pub last_update_id: u64,
    /// Bid levels as `[price, quantity]` string pairs.
    #[serde(rename = "b")]
    pub bids: Vec<[String; 2]>,
    /// Ask levels as `[price, quantity]` string pairs.
    #[serde(rename = "a")]
    pub asks: Vec<[String; 2]>,
}

/// Raw Binance trade from the `@trade` WebSocket stream.
#[derive(Debug, Deserialize)]
pub struct BinanceTrade {
    /// Event type (always `"trade"`).
    #[serde(rename = "e")]
    pub event_type: String,
    /// Event time (milliseconds since epoch).
    #[serde(rename = "E")]
    pub event_time: u64,
    /// Symbol (uppercase).
    #[serde(rename = "s")]
    pub symbol: String,
    /// Trade ID.
    #[serde(rename = "t")]
    pub trade_id: u64,
    /// Price as a decimal string.
    #[serde(rename = "p")]
    pub price: String,
    /// Quantity as a decimal string.
    #[serde(rename = "q")]
    pub quantity: String,
    /// `true` if the buyer is the market maker (i.e., the trade was a sell).
    #[serde(rename = "m")]
    pub is_buyer_maker: bool,
    /// Trade time (milliseconds since epoch).
    #[serde(rename = "T")]
    pub trade_time: u64,
}

/// REST API response for `/api/v3/depth`.
#[derive(Debug, Deserialize)]
pub struct BinanceDepthSnapshot {
    /// Last update ID included in the snapshot.
    #[serde(rename = "lastUpdateId")]
    pub last_update_id: u64,
    /// Bid levels as `[price, quantity]` string pairs.
    pub bids: Vec<[String; 2]>,
    /// Ask levels as `[price, quantity]` string pairs.
    pub asks: Vec<[String; 2]>,
}

/// Parse a decimal string into a [`Price`] with scale 8.
///
/// Uses `f64` conversion internally — acceptable for deserialization but
/// **not** for the hot path.
fn parse_price(s: &str) -> Price {
    let v: f64 = s.parse().unwrap_or(0.0);
    Price::from(v)
}

/// Parse a decimal string into a [`Quantity`] with scale 8.
fn parse_quantity(s: &str) -> Quantity {
    let v: f64 = s.parse().unwrap_or(0.0);
    Quantity::from(v)
}

/// Parse a `[price, quantity]` string pair into `(Price, Quantity)`.
fn parse_level(pair: &[String; 2]) -> (Price, Quantity) {
    (parse_price(&pair[0]), parse_quantity(&pair[1]))
}

impl From<BinanceDepthUpdate> for BookUpdate {
    fn from(raw: BinanceDepthUpdate) -> Self {
        BookUpdate {
            exchange: Exchange::Binance,
            symbol: Symbol::new(raw.symbol),
            timestamp: Timestamp::from_millis(raw.event_time),
            bids: raw.bids.iter().map(parse_level).collect(),
            asks: raw.asks.iter().map(parse_level).collect(),
            is_snapshot: false,
        }
    }
}

impl From<BinanceTrade> for Trade {
    fn from(raw: BinanceTrade) -> Self {
        // `is_buyer_maker == true` means the buyer placed the resting order
        // and the seller was the aggressor (taker side = Sell).
        let side = if raw.is_buyer_maker {
            Side::Sell
        } else {
            Side::Buy
        };

        Trade {
            exchange: Exchange::Binance,
            symbol: Symbol::new(raw.symbol),
            timestamp: Timestamp::from_millis(raw.trade_time),
            price: parse_price(&raw.price),
            quantity: parse_quantity(&raw.quantity),
            side,
            trade_id: raw.trade_id.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sample Binance depth update JSON taken from the Binance API docs.
    const DEPTH_UPDATE_JSON: &str = r#"{
        "e": "depthUpdate",
        "E": 1706000000000,
        "s": "BTCUSDT",
        "U": 100,
        "u": 105,
        "b": [
            ["50000.50", "1.500"],
            ["49999.00", "0.200"]
        ],
        "a": [
            ["50001.00", "0.800"],
            ["50002.50", "2.000"]
        ]
    }"#;

    /// Sample Binance trade JSON.
    const TRADE_JSON: &str = r#"{
        "e": "trade",
        "E": 1706000000000,
        "s": "BTCUSDT",
        "t": 123456789,
        "p": "50000.50",
        "q": "0.001",
        "m": false,
        "T": 1706000000001
    }"#;

    /// Sample REST depth snapshot JSON.
    const SNAPSHOT_JSON: &str = r#"{
        "lastUpdateId": 200,
        "bids": [
            ["50000.50", "1.500"],
            ["49999.00", "0.200"]
        ],
        "asks": [
            ["50001.00", "0.800"],
            ["50002.50", "2.000"]
        ]
    }"#;

    #[test]
    fn test_deserialize_depth_update() {
        let update: BinanceDepthUpdate =
            serde_json::from_str(DEPTH_UPDATE_JSON).expect("deserialize depth update");

        assert_eq!(update.event_type, "depthUpdate");
        assert_eq!(update.event_time, 1706000000000);
        assert_eq!(update.symbol, "BTCUSDT");
        assert_eq!(update.first_update_id, 100);
        assert_eq!(update.last_update_id, 105);
        assert_eq!(update.bids.len(), 2);
        assert_eq!(update.asks.len(), 2);
        assert_eq!(update.bids[0][0], "50000.50");
        assert_eq!(update.bids[0][1], "1.500");
    }

    #[test]
    fn test_deserialize_trade() {
        let trade: BinanceTrade =
            serde_json::from_str(TRADE_JSON).expect("deserialize trade");

        assert_eq!(trade.event_type, "trade");
        assert_eq!(trade.event_time, 1706000000000);
        assert_eq!(trade.symbol, "BTCUSDT");
        assert_eq!(trade.trade_id, 123456789);
        assert_eq!(trade.price, "50000.50");
        assert_eq!(trade.quantity, "0.001");
        assert!(!trade.is_buyer_maker);
        assert_eq!(trade.trade_time, 1706000000001);
    }

    #[test]
    fn test_deserialize_snapshot() {
        let snapshot: BinanceDepthSnapshot =
            serde_json::from_str(SNAPSHOT_JSON).expect("deserialize snapshot");

        assert_eq!(snapshot.last_update_id, 200);
        assert_eq!(snapshot.bids.len(), 2);
        assert_eq!(snapshot.asks.len(), 2);
        assert_eq!(snapshot.bids[0][0], "50000.50");
        assert_eq!(snapshot.asks[1][1], "2.000");
    }

    #[test]
    fn test_depth_update_to_book_update() {
        let raw: BinanceDepthUpdate =
            serde_json::from_str(DEPTH_UPDATE_JSON).expect("deserialize");
        let book: BookUpdate = raw.into();

        assert_eq!(book.exchange, Exchange::Binance);
        assert_eq!(book.symbol, Symbol::new("BTCUSDT"));
        assert_eq!(book.timestamp, Timestamp::from_millis(1706000000000));
        assert!(!book.is_snapshot);
        assert_eq!(book.bids.len(), 2);
        assert_eq!(book.asks.len(), 2);

        // Check first bid: 50000.50 @ 1.500
        let (bid_price, bid_qty) = &book.bids[0];
        assert!((bid_price.to_f64() - 50000.50).abs() < 1e-6);
        assert!((bid_qty.to_f64() - 1.5).abs() < 1e-6);

        // Check first ask: 50001.00 @ 0.800
        let (ask_price, ask_qty) = &book.asks[0];
        assert!((ask_price.to_f64() - 50001.0).abs() < 1e-6);
        assert!((ask_qty.to_f64() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_trade_to_normalized_trade() {
        let raw: BinanceTrade =
            serde_json::from_str(TRADE_JSON).expect("deserialize");
        let trade: Trade = raw.into();

        assert_eq!(trade.exchange, Exchange::Binance);
        assert_eq!(trade.symbol, Symbol::new("BTCUSDT"));
        assert_eq!(trade.timestamp, Timestamp::from_millis(1706000000001));
        assert!((trade.price.to_f64() - 50000.50).abs() < 1e-6);
        assert!((trade.quantity.to_f64() - 0.001).abs() < 1e-6);
        // is_buyer_maker = false means buyer is taker => side = Buy
        assert_eq!(trade.side, Side::Buy);
        assert_eq!(trade.trade_id, "123456789");
    }

    #[test]
    fn test_trade_seller_is_taker() {
        let json = r#"{
            "e": "trade", "E": 1706000000000, "s": "BTCUSDT",
            "t": 999, "p": "50000.00", "q": "0.5",
            "m": true, "T": 1706000000000
        }"#;
        let raw: BinanceTrade = serde_json::from_str(json).expect("deserialize");
        let trade: Trade = raw.into();
        // is_buyer_maker = true => taker side = Sell
        assert_eq!(trade.side, Side::Sell);
    }

    #[test]
    fn test_parse_price_and_quantity() {
        let p = parse_price("50000.50");
        assert!((p.to_f64() - 50000.50).abs() < 1e-6);

        let q = parse_quantity("1.23456789");
        assert!((q.to_f64() - 1.23456789).abs() < 1e-6);
    }

    #[test]
    fn test_parse_price_zero() {
        let p = parse_price("0.00000000");
        assert!(p.is_zero());
    }
}

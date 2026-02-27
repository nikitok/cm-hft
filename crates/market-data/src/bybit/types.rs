//! Bybit wire-format types for WebSocket message deserialization.
//!
//! These types map directly to the JSON structures sent by Bybit's v5 public
//! linear perpetual WebSocket API. They are converted into the normalized
//! internal types before being passed to the rest of the system.

use serde::{Deserialize, Serialize};

use cm_core::types::{BookUpdate, Exchange, Price, Quantity, Side, Symbol, Timestamp, Trade};

/// Top-level WebSocket response envelope from Bybit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitWsResponse {
    /// Topic name (e.g., "orderbook.200.BTCUSDT").
    pub topic: Option<String>,
    /// Message type: "snapshot" or "delta".
    #[serde(rename = "type")]
    pub msg_type: Option<String>,
    /// Operation field for subscribe/pong responses.
    pub op: Option<String>,
    /// Whether a subscribe/unsubscribe operation succeeded.
    pub success: Option<bool>,
    /// Message payload (varies by topic).
    pub data: Option<serde_json::Value>,
    /// Server timestamp in milliseconds.
    pub ts: Option<u64>,
}

/// Bybit orderbook data from a snapshot or delta message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitOrderbook {
    /// Symbol name (e.g., "BTCUSDT").
    pub s: String,
    /// Bid levels as `[price_string, qty_string]` pairs.
    pub b: Vec<[String; 2]>,
    /// Ask levels as `[price_string, qty_string]` pairs.
    pub a: Vec<[String; 2]>,
    /// Update ID.
    pub u: u64,
    /// Sequence number for gap detection.
    pub seq: u64,
}

impl BybitOrderbook {
    /// Convert to a normalized [`BookUpdate`].
    pub fn to_book_update(&self, is_snapshot: bool) -> BookUpdate {
        let parse_levels = |levels: &[[String; 2]]| -> Vec<(Price, Quantity)> {
            levels
                .iter()
                .filter_map(|[price_str, qty_str]| {
                    let price: f64 = price_str.parse().ok()?;
                    let qty: f64 = qty_str.parse().ok()?;
                    Some((Price::from(price), Quantity::from(qty)))
                })
                .collect()
        };

        BookUpdate {
            exchange: Exchange::Bybit,
            symbol: Symbol::new(&self.s),
            timestamp: Timestamp::now(),
            bids: parse_levels(&self.b),
            asks: parse_levels(&self.a),
            is_snapshot,
        }
    }
}

/// Bybit public trade entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitTrade {
    /// Trade timestamp in milliseconds.
    #[serde(rename = "T")]
    pub timestamp: u64,
    /// Symbol name.
    pub s: String,
    /// Side: "Buy" or "Sell".
    #[serde(rename = "S")]
    pub side: String,
    /// Quantity as a string.
    pub v: String,
    /// Price as a string.
    pub p: String,
    /// Trade ID.
    pub i: String,
}

impl From<&BybitTrade> for Trade {
    fn from(bt: &BybitTrade) -> Self {
        let price: f64 = bt.p.parse().unwrap_or(0.0);
        let quantity: f64 = bt.v.parse().unwrap_or(0.0);
        let side = if bt.side == "Buy" {
            Side::Buy
        } else {
            Side::Sell
        };

        Trade {
            exchange: Exchange::Bybit,
            symbol: Symbol::new(&bt.s),
            timestamp: Timestamp::from_millis(bt.timestamp),
            price: Price::from(price),
            quantity: Quantity::from(quantity),
            side,
            trade_id: bt.i.clone(),
        }
    }
}

/// Subscribe request sent to Bybit WebSocket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitSubscribeRequest {
    /// Operation: "subscribe" or "unsubscribe".
    pub op: String,
    /// List of topic strings to subscribe to.
    pub args: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_orderbook_snapshot() {
        let json = r#"{
            "s": "BTCUSDT",
            "b": [
                ["65000.50", "1.234"],
                ["64999.00", "0.500"]
            ],
            "a": [
                ["65001.00", "2.100"],
                ["65002.50", "0.800"]
            ],
            "u": 1234567,
            "seq": 100
        }"#;

        let ob: BybitOrderbook = serde_json::from_str(json).unwrap();
        assert_eq!(ob.s, "BTCUSDT");
        assert_eq!(ob.b.len(), 2);
        assert_eq!(ob.a.len(), 2);
        assert_eq!(ob.b[0][0], "65000.50");
        assert_eq!(ob.b[0][1], "1.234");
        assert_eq!(ob.u, 1234567);
        assert_eq!(ob.seq, 100);
    }

    #[test]
    fn test_deserialize_orderbook_delta() {
        let json = r#"{
            "s": "BTCUSDT",
            "b": [
                ["64999.00", "0"]
            ],
            "a": [
                ["65001.00", "3.500"]
            ],
            "u": 1234568,
            "seq": 101
        }"#;

        let ob: BybitOrderbook = serde_json::from_str(json).unwrap();
        assert_eq!(ob.b.len(), 1);
        assert_eq!(ob.b[0][1], "0"); // removal
        assert_eq!(ob.seq, 101);
    }

    #[test]
    fn test_deserialize_trade() {
        let json = r#"{
            "T": 1706000000000,
            "s": "BTCUSDT",
            "S": "Buy",
            "v": "0.001",
            "p": "65000.50",
            "i": "trade-123"
        }"#;

        let trade: BybitTrade = serde_json::from_str(json).unwrap();
        assert_eq!(trade.timestamp, 1706000000000);
        assert_eq!(trade.s, "BTCUSDT");
        assert_eq!(trade.side, "Buy");
        assert_eq!(trade.v, "0.001");
        assert_eq!(trade.p, "65000.50");
        assert_eq!(trade.i, "trade-123");
    }

    #[test]
    fn test_deserialize_ws_response_data() {
        let json = r#"{
            "topic": "orderbook.200.BTCUSDT",
            "type": "snapshot",
            "data": {
                "s": "BTCUSDT",
                "b": [["65000.50", "1.234"]],
                "a": [["65001.00", "2.100"]],
                "u": 1234567,
                "seq": 100
            },
            "ts": 1706000000000
        }"#;

        let resp: BybitWsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.topic.as_deref(), Some("orderbook.200.BTCUSDT"));
        assert_eq!(resp.msg_type.as_deref(), Some("snapshot"));
        assert!(resp.data.is_some());
        assert_eq!(resp.ts, Some(1706000000000));
        assert!(resp.success.is_none());
    }

    #[test]
    fn test_deserialize_ws_response_subscribe_confirm() {
        let json = r#"{
            "success": true,
            "op": "subscribe",
            "conn_id": "abc123"
        }"#;

        let resp: BybitWsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.success, Some(true));
        assert_eq!(resp.op.as_deref(), Some("subscribe"));
        assert!(resp.topic.is_none());
        assert!(resp.data.is_none());
    }

    #[test]
    fn test_orderbook_to_book_update_snapshot() {
        let ob = BybitOrderbook {
            s: "BTCUSDT".to_string(),
            b: vec![
                ["65000.50".to_string(), "1.234".to_string()],
                ["64999.00".to_string(), "0.500".to_string()],
            ],
            a: vec![["65001.00".to_string(), "2.100".to_string()]],
            u: 100,
            seq: 50,
        };

        let update = ob.to_book_update(true);
        assert_eq!(update.exchange, Exchange::Bybit);
        assert_eq!(update.symbol, Symbol::new("BTCUSDT"));
        assert!(update.is_snapshot);
        assert_eq!(update.bids.len(), 2);
        assert_eq!(update.asks.len(), 1);

        let (bid_price, bid_qty) = &update.bids[0];
        assert!((bid_price.to_f64() - 65000.50).abs() < 1e-4);
        assert!((bid_qty.to_f64() - 1.234).abs() < 1e-4);
    }

    #[test]
    fn test_orderbook_to_book_update_delta() {
        let ob = BybitOrderbook {
            s: "ETHUSDT".to_string(),
            b: vec![["3500.00".to_string(), "0".to_string()]],
            a: vec![],
            u: 200,
            seq: 51,
        };

        let update = ob.to_book_update(false);
        assert!(!update.is_snapshot);
        assert_eq!(update.bids.len(), 1);
        assert!(update.bids[0].1.is_zero());
        assert!(update.asks.is_empty());
    }

    #[test]
    fn test_trade_conversion() {
        let bybit_trade = BybitTrade {
            timestamp: 1706000000000,
            s: "BTCUSDT".to_string(),
            side: "Buy".to_string(),
            v: "0.001".to_string(),
            p: "65000.50".to_string(),
            i: "trade-456".to_string(),
        };

        let trade = Trade::from(&bybit_trade);
        assert_eq!(trade.exchange, Exchange::Bybit);
        assert_eq!(trade.symbol, Symbol::new("BTCUSDT"));
        assert_eq!(trade.timestamp, Timestamp::from_millis(1706000000000));
        assert!((trade.price.to_f64() - 65000.50).abs() < 1e-4);
        assert!((trade.quantity.to_f64() - 0.001).abs() < 1e-8);
        assert_eq!(trade.side, Side::Buy);
        assert_eq!(trade.trade_id, "trade-456");
    }

    #[test]
    fn test_trade_conversion_sell_side() {
        let bybit_trade = BybitTrade {
            timestamp: 1706000000000,
            s: "BTCUSDT".to_string(),
            side: "Sell".to_string(),
            v: "0.500".to_string(),
            p: "64999.00".to_string(),
            i: "trade-789".to_string(),
        };

        let trade = Trade::from(&bybit_trade);
        assert_eq!(trade.side, Side::Sell);
    }

    #[test]
    fn test_subscribe_request_serialization() {
        let req = BybitSubscribeRequest {
            op: "subscribe".to_string(),
            args: vec![
                "orderbook.200.BTCUSDT".to_string(),
                "publicTrade.BTCUSDT".to_string(),
            ],
        };

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["op"], "subscribe");
        let args = json["args"].as_array().unwrap();
        assert_eq!(args.len(), 2);
        assert_eq!(args[0], "orderbook.200.BTCUSDT");
        assert_eq!(args[1], "publicTrade.BTCUSDT");
    }

    #[test]
    fn test_deserialize_trade_array() {
        let json = r#"[
            {
                "T": 1706000000000,
                "s": "BTCUSDT",
                "S": "Buy",
                "v": "0.001",
                "p": "65000.50",
                "i": "t1"
            },
            {
                "T": 1706000000001,
                "s": "BTCUSDT",
                "S": "Sell",
                "v": "0.002",
                "p": "65001.00",
                "i": "t2"
            }
        ]"#;

        let trades: Vec<BybitTrade> = serde_json::from_str(json).unwrap();
        assert_eq!(trades.len(), 2);
        assert_eq!(trades[0].side, "Buy");
        assert_eq!(trades[1].side, "Sell");
    }
}

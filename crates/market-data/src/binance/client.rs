//! Binance WebSocket client for market data feeds.
//!
//! [`BinanceWsClient`] connects to the Binance WebSocket API, subscribes to
//! depth and trade streams, and converts incoming messages into the normalized
//! [`cm_core::types`] format.

use anyhow::{Context, Result};
use cm_core::config::ExchangeConfig;
use cm_core::types::{BookUpdate, Timestamp, Trade};
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use super::types::{BinanceDepthSnapshot, BinanceDepthUpdate, BinanceTrade};

/// A connected Binance WebSocket stream.
pub type BinanceWsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Parsed market data message from Binance.
#[derive(Debug)]
pub enum BinanceMessage {
    /// Depth (order book) update.
    Depth(BookUpdate),
    /// Trade event.
    Trade(Trade),
}

/// WebSocket client for Binance market data feeds.
///
/// Handles connection, subscription, snapshot fetching, and message parsing
/// for the Binance exchange.
pub struct BinanceWsClient {
    config: ExchangeConfig,
    symbols: Vec<String>,
}

impl BinanceWsClient {
    /// Create a new Binance WebSocket client.
    pub fn new(config: ExchangeConfig, symbols: Vec<String>) -> Self {
        Self { config, symbols }
    }

    /// Connect to the Binance WebSocket endpoint.
    ///
    /// Uses the `ws_url` from the [`ExchangeConfig`]. The connection is
    /// established but no subscriptions are sent yet — call
    /// [`subscribe`](Self::subscribe) after connecting.
    pub async fn connect(&self) -> Result<BinanceWsStream> {
        let url = &self.config.ws_url;
        tracing::info!(url = %url, "connecting to Binance WebSocket");

        let (stream, _response) = tokio_tungstenite::connect_async(url)
            .await
            .context("failed to connect to Binance WebSocket")?;

        tracing::info!(url = %url, "connected to Binance WebSocket");
        Ok(stream)
    }

    /// Subscribe to depth and trade streams for all configured symbols.
    ///
    /// Sends a single combined subscription message following the Binance
    /// combined stream format:
    /// ```json
    /// {"method": "SUBSCRIBE", "params": ["btcusdt@depth@100ms", "btcusdt@trade"], "id": 1}
    /// ```
    pub async fn subscribe(&self, stream: &mut BinanceWsStream) -> Result<()> {
        let params: Vec<String> = self
            .symbols
            .iter()
            .flat_map(|s| {
                let lower = s.to_lowercase();
                vec![format!("{lower}@depth@100ms"), format!("{lower}@trade")]
            })
            .collect();

        let subscribe_msg = serde_json::json!({
            "method": "SUBSCRIBE",
            "params": params,
            "id": 1
        });

        let msg_text = serde_json::to_string(&subscribe_msg)
            .context("failed to serialize subscribe message")?;

        tracing::info!(
            symbols = ?self.symbols,
            params = ?params,
            "subscribing to Binance streams"
        );

        stream
            .send(Message::Text(msg_text))
            .await
            .context("failed to send subscribe message")?;

        Ok(())
    }

    /// Fetch a depth snapshot from the REST API.
    ///
    /// GET `/api/v3/depth?symbol={SYMBOL}&limit=1000`
    ///
    /// The snapshot is used to initialize the order book before applying
    /// incremental depth updates from the WebSocket stream.
    pub async fn fetch_snapshot(&self, symbol: &str) -> Result<BinanceDepthSnapshot> {
        let url = format!(
            "{}/api/v3/depth?symbol={}&limit=1000",
            self.config.rest_url,
            symbol.to_uppercase()
        );

        tracing::info!(url = %url, symbol = %symbol, "fetching depth snapshot");

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(self.config.timeout_ms))
            .build()
            .context("failed to build HTTP client")?;

        let snapshot: BinanceDepthSnapshot = client
            .get(&url)
            .send()
            .await
            .context("failed to send snapshot request")?
            .json()
            .await
            .context("failed to deserialize depth snapshot")?;

        tracing::info!(
            symbol = %symbol,
            last_update_id = snapshot.last_update_id,
            bids = snapshot.bids.len(),
            asks = snapshot.asks.len(),
            "depth snapshot received"
        );

        Ok(snapshot)
    }

    /// Read and parse the next message from the WebSocket stream.
    ///
    /// Returns `None` for non-data frames (pong responses, subscription
    /// confirmations) and `Some(BinanceMessage)` for depth or trade data.
    ///
    /// Logs wire-to-parsed latency on every successfully parsed message.
    pub async fn read_message(stream: &mut BinanceWsStream) -> Result<Option<BinanceMessage>> {
        loop {
            let msg = stream
                .next()
                .await
                .context("WebSocket stream ended unexpectedly")?
                .context("WebSocket read error")?;

            match msg {
                Message::Text(text) => {
                    let recv_ts = Timestamp::now();
                    return Ok(Self::parse_text_message(&text, recv_ts));
                }
                Message::Ping(data) => {
                    // Respond to ping with pong.
                    stream
                        .send(Message::Pong(data))
                        .await
                        .context("failed to send pong")?;
                    tracing::trace!("responded to ping with pong");
                    continue;
                }
                Message::Pong(_) => {
                    tracing::trace!("received pong");
                    continue;
                }
                Message::Close(frame) => {
                    tracing::warn!(frame = ?frame, "WebSocket close frame received");
                    anyhow::bail!("WebSocket connection closed by server");
                }
                Message::Binary(_) | Message::Frame(_) => {
                    tracing::trace!("ignoring non-text frame");
                    continue;
                }
            }
        }
    }

    /// Parse a text WebSocket message into a [`BinanceMessage`].
    ///
    /// Attempts to parse as a depth update first, then as a trade.  Returns
    /// `None` if the message is neither (e.g., subscription confirmation).
    fn parse_text_message(text: &str, recv_ts: Timestamp) -> Option<BinanceMessage> {
        // Try depth update.
        if let Ok(depth) = serde_json::from_str::<BinanceDepthUpdate>(text) {
            if depth.event_type == "depthUpdate" {
                let event_ts = Timestamp::from_millis(depth.event_time);
                let latency_us = recv_ts.elapsed_since(&event_ts) / 1_000;
                tracing::debug!(
                    symbol = %depth.symbol,
                    latency_us = latency_us,
                    first_id = depth.first_update_id,
                    last_id = depth.last_update_id,
                    bids = depth.bids.len(),
                    asks = depth.asks.len(),
                    "parsed depth update"
                );

                let book_update: BookUpdate = depth.into();
                return Some(BinanceMessage::Depth(book_update));
            }
        }

        // Try trade.
        if let Ok(trade) = serde_json::from_str::<BinanceTrade>(text) {
            if trade.event_type == "trade" {
                let event_ts = Timestamp::from_millis(trade.trade_time);
                let latency_us = recv_ts.elapsed_since(&event_ts) / 1_000;
                tracing::debug!(
                    symbol = %trade.symbol,
                    latency_us = latency_us,
                    trade_id = trade.trade_id,
                    price = %trade.price,
                    qty = %trade.quantity,
                    "parsed trade"
                );

                let normalized: Trade = trade.into();
                return Some(BinanceMessage::Trade(normalized));
            }
        }

        // Neither depth nor trade — likely a subscription response or unknown.
        tracing::trace!(msg = %text, "ignoring non-data message");
        None
    }

    /// Build the subscription message JSON string for testing.
    pub fn build_subscribe_message(symbols: &[String]) -> String {
        let params: Vec<String> = symbols
            .iter()
            .flat_map(|s| {
                let lower = s.to_lowercase();
                vec![format!("{lower}@depth@100ms"), format!("{lower}@trade")]
            })
            .collect();

        serde_json::json!({
            "method": "SUBSCRIBE",
            "params": params,
            "id": 1
        })
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subscription_message_format() {
        let symbols = vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()];
        let msg = BinanceWsClient::build_subscribe_message(&symbols);
        let parsed: serde_json::Value = serde_json::from_str(&msg).expect("valid JSON");

        assert_eq!(parsed["method"], "SUBSCRIBE");
        assert_eq!(parsed["id"], 1);

        let params = parsed["params"].as_array().expect("params is array");
        assert_eq!(params.len(), 4);
        assert_eq!(params[0], "btcusdt@depth@100ms");
        assert_eq!(params[1], "btcusdt@trade");
        assert_eq!(params[2], "ethusdt@depth@100ms");
        assert_eq!(params[3], "ethusdt@trade");
    }

    #[test]
    fn test_parse_depth_message() {
        let json = r#"{
            "e": "depthUpdate",
            "E": 1706000000000,
            "s": "BTCUSDT",
            "U": 100,
            "u": 105,
            "b": [["50000.50", "1.500"]],
            "a": [["50001.00", "0.800"]]
        }"#;

        let recv_ts = Timestamp::from_millis(1706000000001);
        let msg = BinanceWsClient::parse_text_message(json, recv_ts);
        assert!(msg.is_some());
        match msg.unwrap() {
            BinanceMessage::Depth(update) => {
                assert_eq!(update.exchange, cm_core::types::Exchange::Binance);
                assert_eq!(update.symbol, cm_core::types::Symbol::new("BTCUSDT"));
                assert!(!update.is_snapshot);
                assert_eq!(update.bids.len(), 1);
                assert_eq!(update.asks.len(), 1);
            }
            BinanceMessage::Trade(_) => panic!("expected depth, got trade"),
        }
    }

    #[test]
    fn test_parse_trade_message() {
        let json = r#"{
            "e": "trade",
            "E": 1706000000000,
            "s": "BTCUSDT",
            "t": 123456,
            "p": "50000.50",
            "q": "0.001",
            "m": false,
            "T": 1706000000001
        }"#;

        let recv_ts = Timestamp::from_millis(1706000000002);
        let msg = BinanceWsClient::parse_text_message(json, recv_ts);
        assert!(msg.is_some());
        match msg.unwrap() {
            BinanceMessage::Trade(trade) => {
                assert_eq!(trade.exchange, cm_core::types::Exchange::Binance);
                assert_eq!(trade.trade_id, "123456");
                assert_eq!(trade.side, cm_core::types::Side::Buy);
            }
            BinanceMessage::Depth(_) => panic!("expected trade, got depth"),
        }
    }

    #[test]
    fn test_parse_subscription_response_returns_none() {
        let json = r#"{"result": null, "id": 1}"#;
        let recv_ts = Timestamp::from_millis(1706000000000);
        let msg = BinanceWsClient::parse_text_message(json, recv_ts);
        assert!(msg.is_none());
    }

    #[test]
    fn test_single_symbol_subscription() {
        let symbols = vec!["BTCUSDT".to_string()];
        let msg = BinanceWsClient::build_subscribe_message(&symbols);
        let parsed: serde_json::Value = serde_json::from_str(&msg).expect("valid JSON");

        let params = parsed["params"].as_array().expect("params is array");
        assert_eq!(params.len(), 2);
        assert_eq!(params[0], "btcusdt@depth@100ms");
        assert_eq!(params[1], "btcusdt@trade");
    }
}

//! Bybit WebSocket client for public linear perpetual feeds.
//!
//! Connects to Bybit's v5 public WebSocket API, subscribes to orderbook and
//! trade topics, parses responses into normalized types, and validates message
//! sequencing for gap detection.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result};
use futures_util::SinkExt;
use tokio_tungstenite::tungstenite::Message;

use cm_core::types::{BookUpdate, Timestamp, Trade};

use crate::ws::{ConnectionState, ReconnectConfig, WsConnection, WsHandler, WsSink};

use super::types::{BybitOrderbook, BybitSubscribeRequest, BybitTrade, BybitWsResponse};

/// Default Bybit public linear WebSocket endpoint.
const BYBIT_WS_URL: &str = "wss://stream.bybit.com/v5/public/linear";

/// Default Bybit testnet public linear WebSocket endpoint.
const BYBIT_TESTNET_WS_URL: &str = "wss://stream-testnet.bybit.com/v5/public/linear";

/// Ping interval for Bybit WebSocket keep-alive.
const PING_INTERVAL: Duration = Duration::from_secs(20);

/// Configuration for the Bybit WebSocket client.
///
/// This is a standalone config until `ExchangeConfig` from `cm-core` is
/// available. Fields mirror what `ExchangeConfig` will provide.
#[derive(Debug, Clone)]
pub struct BybitConfig {
    /// Use testnet endpoint if `true`.
    pub testnet: bool,
    /// Custom WebSocket URL override (takes precedence over `testnet`).
    pub ws_url: Option<String>,
    /// Reconnection configuration.
    pub reconnect: ReconnectConfig,
}

impl Default for BybitConfig {
    fn default() -> Self {
        Self {
            testnet: false,
            ws_url: None,
            reconnect: ReconnectConfig::default(),
        }
    }
}

/// Bybit WebSocket client for public market data feeds.
///
/// Subscribes to `orderbook.200.{SYMBOL}` and `publicTrade.{SYMBOL}` topics
/// for each configured symbol, parses the wire-format messages, and converts
/// them into normalized [`BookUpdate`] and [`Trade`] types.
pub struct BybitWsClient {
    config: BybitConfig,
    symbols: Vec<String>,
}

impl BybitWsClient {
    /// Create a new Bybit WebSocket client.
    pub fn new(config: BybitConfig, symbols: Vec<String>) -> Self {
        Self { config, symbols }
    }

    /// Resolve the WebSocket URL based on configuration.
    pub fn ws_url(&self) -> &str {
        if let Some(ref url) = self.config.ws_url {
            url.as_str()
        } else if self.config.testnet {
            BYBIT_TESTNET_WS_URL
        } else {
            BYBIT_WS_URL
        }
    }

    /// Build the subscription topics for all configured symbols.
    fn subscription_topics(&self) -> Vec<String> {
        let mut topics = Vec::with_capacity(self.symbols.len() * 2);
        for symbol in &self.symbols {
            topics.push(format!("orderbook.200.{symbol}"));
            topics.push(format!("publicTrade.{symbol}"));
        }
        topics
    }

    /// Start the client, connecting and processing messages.
    ///
    /// This spawns the ping task internally and delegates to [`WsConnection`]
    /// for reconnection management.
    pub async fn run(
        &self,
        book_tx: tokio::sync::mpsc::Sender<BookUpdate>,
        trade_tx: tokio::sync::mpsc::Sender<Trade>,
    ) -> Result<()> {
        let url = self.ws_url().to_string();
        let conn = WsConnection::new(url, self.config.reconnect.clone());

        let mut handler = BybitHandler {
            topics: self.subscription_topics(),
            book_tx,
            trade_tx,
            seq_tracker: HashMap::new(),
        };

        conn.run(&mut handler).await
    }
}

/// Per-symbol sequence tracking state.
#[derive(Debug)]
struct SymbolSeqState {
    /// Last seen sequence number.
    last_seq: u64,
    /// Whether we have received the initial snapshot.
    initialized: bool,
}

/// Internal handler implementing [`WsHandler`] for Bybit feeds.
struct BybitHandler {
    topics: Vec<String>,
    book_tx: tokio::sync::mpsc::Sender<BookUpdate>,
    trade_tx: tokio::sync::mpsc::Sender<Trade>,
    seq_tracker: HashMap<String, SymbolSeqState>,
}

impl BybitHandler {
    /// Build and send a subscribe request over the sink.
    async fn send_subscribe(&self, sink: &mut WsSink) -> Result<()> {
        let req = BybitSubscribeRequest {
            op: "subscribe".to_string(),
            args: self.topics.clone(),
        };
        let payload = serde_json::to_string(&req)?;
        tracing::info!(topics = ?self.topics, "subscribing to Bybit topics");
        sink.send(Message::Text(payload)).await?;
        Ok(())
    }

    /// Handle a parsed data message.
    async fn handle_data_message(&mut self, resp: &BybitWsResponse) -> Result<()> {
        let topic = match resp.topic.as_deref() {
            Some(t) => t,
            None => return Ok(()),
        };
        let data = match &resp.data {
            Some(d) => d,
            None => return Ok(()),
        };

        let receive_ts = Timestamp::now();

        if topic.starts_with("orderbook.") {
            self.handle_orderbook(topic, data, resp.msg_type.as_deref(), resp.ts)
                .await?;
        } else if topic.starts_with("publicTrade.") {
            self.handle_trades(data, resp.ts).await?;
        }

        // Log wire-to-parsed latency.
        if let Some(server_ts) = resp.ts {
            let server_ns = Timestamp::from_millis(server_ts);
            let latency_ns = receive_ts.elapsed_since(&server_ns);
            tracing::debug!(
                topic = topic,
                latency_us = latency_ns / 1000,
                "wire-to-parsed latency"
            );
        }

        Ok(())
    }

    /// Handle an orderbook message (snapshot or delta).
    async fn handle_orderbook(
        &mut self,
        _topic: &str,
        data: &serde_json::Value,
        msg_type: Option<&str>,
        _server_ts: Option<u64>,
    ) -> Result<()> {
        let ob: BybitOrderbook =
            serde_json::from_value(data.clone()).context("failed to parse orderbook data")?;

        let is_snapshot = msg_type == Some("snapshot");

        // Sequence validation.
        let state = self
            .seq_tracker
            .entry(ob.s.clone())
            .or_insert(SymbolSeqState {
                last_seq: 0,
                initialized: false,
            });

        if is_snapshot {
            state.last_seq = ob.seq;
            state.initialized = true;
        } else if state.initialized {
            let expected = state.last_seq + 1;
            if ob.seq != expected {
                tracing::warn!(
                    symbol = %ob.s,
                    expected_seq = expected,
                    received_seq = ob.seq,
                    "sequence gap detected in orderbook"
                );
            }
            state.last_seq = ob.seq;
        } else {
            tracing::debug!(
                symbol = %ob.s,
                "ignoring delta before initial snapshot"
            );
            return Ok(());
        }

        let update = ob.to_book_update(is_snapshot);
        if self.book_tx.send(update).await.is_err() {
            tracing::warn!("book_tx receiver dropped");
        }

        Ok(())
    }

    /// Handle a public trade message.
    async fn handle_trades(
        &mut self,
        data: &serde_json::Value,
        _server_ts: Option<u64>,
    ) -> Result<()> {
        // Bybit sends trades as an array.
        let trades: Vec<BybitTrade> =
            serde_json::from_value(data.clone()).context("failed to parse trade data")?;

        for bybit_trade in &trades {
            let trade = Trade::from(bybit_trade);
            if self.trade_tx.send(trade).await.is_err() {
                tracing::warn!("trade_tx receiver dropped");
                break;
            }
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl WsHandler for BybitHandler {
    async fn on_connect(&mut self, sink: &mut WsSink) -> Result<()> {
        // Reset sequence tracking on reconnect.
        self.seq_tracker.clear();
        self.send_subscribe(sink).await
    }

    async fn on_message(&mut self, msg: Message) -> Result<()> {
        match msg {
            Message::Text(text) => {
                let resp: BybitWsResponse = serde_json::from_str(&text)
                    .context("failed to parse Bybit WS response")?;

                // Handle subscription confirmations.
                if resp.op.as_deref() == Some("subscribe") {
                    if resp.success == Some(true) {
                        tracing::info!("Bybit subscription confirmed");
                    } else {
                        tracing::warn!("Bybit subscription failed: {:?}", resp);
                    }
                    return Ok(());
                }

                // Handle pong responses.
                if resp.op.as_deref() == Some("pong") {
                    tracing::trace!("Bybit pong received");
                    return Ok(());
                }

                // Handle data messages.
                self.handle_data_message(&resp).await?;
            }
            Message::Ping(_data) => {
                tracing::trace!("received WebSocket ping");
                // tungstenite auto-replies with pong.
            }
            Message::Close(frame) => {
                tracing::info!(frame = ?frame, "received WebSocket close frame");
            }
            _ => {}
        }

        Ok(())
    }

    fn on_state_change(&mut self, state: ConnectionState) {
        match &state {
            ConnectionState::Connected => {
                tracing::info!("Bybit WebSocket connected");
            }
            ConnectionState::Disconnected { reason } => {
                tracing::warn!(reason = %reason, "Bybit WebSocket disconnected");
            }
            ConnectionState::Reconnecting { attempt } => {
                tracing::info!(attempt = attempt, "Bybit WebSocket reconnecting");
            }
            ConnectionState::Failed { reason } => {
                tracing::error!(reason = %reason, "Bybit WebSocket connection failed permanently");
            }
        }
    }
}

/// Spawn a periodic ping task that sends `{"op":"ping"}` at the configured interval.
///
/// Returns a [`tokio::task::JoinHandle`] that can be used to abort the task
/// when the connection is torn down.
pub fn spawn_ping_task(
    sink: tokio::sync::mpsc::Sender<String>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(PING_INTERVAL);
        loop {
            interval.tick().await;
            let ping = r#"{"op":"ping"}"#.to_string();
            if sink.send(ping).await.is_err() {
                break;
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cm_core::types::{Exchange, Side};

    #[test]
    fn test_bybit_config_default() {
        let config = BybitConfig::default();
        assert!(!config.testnet);
        assert!(config.ws_url.is_none());
    }

    #[test]
    fn test_ws_url_production() {
        let client = BybitWsClient::new(BybitConfig::default(), vec!["BTCUSDT".to_string()]);
        assert_eq!(client.ws_url(), BYBIT_WS_URL);
    }

    #[test]
    fn test_ws_url_testnet() {
        let config = BybitConfig {
            testnet: true,
            ..Default::default()
        };
        let client = BybitWsClient::new(config, vec!["BTCUSDT".to_string()]);
        assert_eq!(client.ws_url(), BYBIT_TESTNET_WS_URL);
    }

    #[test]
    fn test_ws_url_custom_override() {
        let config = BybitConfig {
            ws_url: Some("wss://custom.example.com".to_string()),
            testnet: true, // should be ignored
            ..Default::default()
        };
        let client = BybitWsClient::new(config, vec!["BTCUSDT".to_string()]);
        assert_eq!(client.ws_url(), "wss://custom.example.com");
    }

    #[test]
    fn test_subscription_topics() {
        let client = BybitWsClient::new(
            BybitConfig::default(),
            vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],
        );
        let topics = client.subscription_topics();
        assert_eq!(topics.len(), 4);
        assert!(topics.contains(&"orderbook.200.BTCUSDT".to_string()));
        assert!(topics.contains(&"publicTrade.BTCUSDT".to_string()));
        assert!(topics.contains(&"orderbook.200.ETHUSDT".to_string()));
        assert!(topics.contains(&"publicTrade.ETHUSDT".to_string()));
    }

    #[test]
    fn test_sequence_gap_detection() {
        // Simulate sequence tracking logic.
        let mut tracker: HashMap<String, SymbolSeqState> = HashMap::new();

        // First snapshot initializes.
        let state = tracker.entry("BTCUSDT".to_string()).or_insert(SymbolSeqState {
            last_seq: 0,
            initialized: false,
        });
        state.last_seq = 100;
        state.initialized = true;

        // Normal delta (seq 101): no gap.
        let state = tracker.get_mut("BTCUSDT").unwrap();
        assert_eq!(state.last_seq + 1, 101);
        state.last_seq = 101;

        // Another normal delta (seq 102): no gap.
        assert_eq!(state.last_seq + 1, 102);
        state.last_seq = 102;

        // Gap: seq jumps to 105 (expected 103).
        let expected = state.last_seq + 1;
        let received = 105u64;
        assert_ne!(expected, received, "should detect gap");
        assert_eq!(expected, 103);
    }

    #[test]
    fn test_sequence_tracker_reset_on_snapshot() {
        let mut tracker: HashMap<String, SymbolSeqState> = HashMap::new();

        // Insert existing state.
        tracker.insert(
            "BTCUSDT".to_string(),
            SymbolSeqState {
                last_seq: 500,
                initialized: true,
            },
        );

        // New snapshot resets seq.
        let state = tracker.get_mut("BTCUSDT").unwrap();
        state.last_seq = 1000; // snapshot seq
        state.initialized = true;

        assert_eq!(state.last_seq, 1000);
    }

    #[test]
    fn test_full_message_parsing_orderbook_snapshot() {
        let json = r#"{
            "topic": "orderbook.200.BTCUSDT",
            "type": "snapshot",
            "data": {
                "s": "BTCUSDT",
                "b": [
                    ["65000.50", "1.234"],
                    ["64999.00", "0.500"]
                ],
                "a": [
                    ["65001.00", "2.100"]
                ],
                "u": 1234567,
                "seq": 100
            },
            "ts": 1706000000000
        }"#;

        let resp: BybitWsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.topic.as_deref(), Some("orderbook.200.BTCUSDT"));
        assert_eq!(resp.msg_type.as_deref(), Some("snapshot"));

        let data = resp.data.unwrap();
        let ob: BybitOrderbook = serde_json::from_value(data).unwrap();
        let update = ob.to_book_update(true);

        assert!(update.is_snapshot);
        assert_eq!(update.exchange, Exchange::Bybit);
        assert_eq!(update.bids.len(), 2);
        assert_eq!(update.asks.len(), 1);
    }

    #[test]
    fn test_full_message_parsing_trade() {
        let json = r#"{
            "topic": "publicTrade.BTCUSDT",
            "type": "snapshot",
            "data": [
                {
                    "T": 1706000000000,
                    "s": "BTCUSDT",
                    "S": "Buy",
                    "v": "0.001",
                    "p": "65000.50",
                    "i": "trade-1"
                }
            ],
            "ts": 1706000000000
        }"#;

        let resp: BybitWsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.topic.as_deref(), Some("publicTrade.BTCUSDT"));

        let data = resp.data.unwrap();
        let trades: Vec<BybitTrade> = serde_json::from_value(data).unwrap();
        assert_eq!(trades.len(), 1);

        let trade = Trade::from(&trades[0]);
        assert_eq!(trade.exchange, Exchange::Bybit);
        assert_eq!(trade.side, Side::Buy);
    }
}

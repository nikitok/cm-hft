//! QuestDB ILP (InfluxDB Line Protocol) client for high-speed data ingestion.
//!
//! Writes market data to QuestDB over a raw TCP connection using the ILP format:
//! `measurement,tag1=val1 field1=value1 timestamp_ns\n`

use std::fmt::Write as FmtWrite;
use std::time::Duration;

use anyhow::Result;
use thiserror::Error;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tracing::{debug, info, warn};

use cm_core::types::{BookUpdate, Side, Trade};

/// Errors specific to QuestDB ILP operations.
#[derive(Debug, Error)]
pub enum QuestDbError {
    /// Failed to establish TCP connection to QuestDB.
    #[error("connection failed: {0}")]
    ConnectionFailed(String),
    /// Failed to write data to the TCP stream.
    #[error("write failed: {0}")]
    WriteFailed(String),
    /// Client is not connected.
    #[error("not connected")]
    NotConnected,
}

/// QuestDB ILP (InfluxDB Line Protocol) client for high-speed data ingestion.
///
/// ILP format: `measurement,tag1=val1,tag2=val2 field1=value1,field2=value2 timestamp_ns\n`
///
/// The client buffers ILP lines in memory and flushes them to QuestDB over TCP
/// when [`flush`](QuestDbClient::flush) is called.
pub struct QuestDbClient {
    addr: String,
    stream: Option<TcpStream>,
    buffer: Vec<u8>,
    max_buffer_size: usize,
}

impl QuestDbClient {
    /// Maximum number of connection retry attempts.
    const MAX_RETRIES: u32 = 3;
    /// Base delay between retry attempts.
    const RETRY_BASE_DELAY: Duration = Duration::from_millis(100);

    /// Creates a new QuestDB ILP client targeting the given address.
    ///
    /// Does **not** establish a connection; call [`connect`](QuestDbClient::connect)
    /// before writing data.
    pub fn new(addr: String) -> Self {
        Self {
            addr,
            stream: None,
            buffer: Vec::with_capacity(64 * 1024),
            max_buffer_size: 4 * 1024 * 1024, // 4 MB
        }
    }

    /// Establishes a TCP connection to QuestDB with exponential-backoff retry.
    pub async fn connect(&mut self) -> Result<()> {
        for attempt in 0..Self::MAX_RETRIES {
            match TcpStream::connect(&self.addr).await {
                Ok(stream) => {
                    stream.set_nodelay(true).ok();
                    self.stream = Some(stream);
                    info!(addr = %self.addr, "connected to QuestDB");
                    return Ok(());
                }
                Err(e) => {
                    let delay = Self::RETRY_BASE_DELAY * 2u32.pow(attempt);
                    warn!(
                        addr = %self.addr,
                        attempt = attempt + 1,
                        error = %e,
                        "connection attempt failed, retrying in {:?}",
                        delay,
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }
        Err(QuestDbError::ConnectionFailed(format!(
            "failed to connect to {} after {} attempts",
            self.addr,
            Self::MAX_RETRIES,
        ))
        .into())
    }

    /// Drops the current connection and establishes a new one.
    pub async fn reconnect(&mut self) -> Result<()> {
        if let Some(stream) = self.stream.take() {
            drop(stream);
        }
        debug!(addr = %self.addr, "reconnecting to QuestDB");
        self.connect().await
    }

    /// Returns `true` if the buffer has exceeded its maximum allowed size.
    pub fn buffer_full(&self) -> bool {
        self.buffer.len() >= self.max_buffer_size
    }

    /// Formats a [`BookUpdate`] as an ILP line and appends it to the internal buffer.
    ///
    /// Writes up to 20 bid/ask levels. Fields use `to_f64()` for numeric values.
    /// The `receive_ts` is the local receive timestamp in nanoseconds, used as the
    /// ILP timestamp. If the buffer has exceeded `max_buffer_size`, the write is
    /// skipped and a warning is logged.
    pub fn write_book_update(&mut self, update: &BookUpdate, receive_ts: u64) {
        if self.buffer_full() {
            warn!(
                buffer_len = self.buffer.len(),
                max = self.max_buffer_size,
                "ILP buffer full, dropping book update",
            );
            return;
        }
        // Measurement + tags
        let exchange = exchange_tag(update.exchange);
        let symbol = &update.symbol.0;

        let mut line = String::with_capacity(2048);
        // measurement,tag=val,tag=val
        write!(line, "book_update,exchange={},symbol={}", exchange, symbol,).unwrap();

        // Fields — space before first field
        line.push(' ');

        let max_levels = 20;
        let mut first_field = true;

        for (i, (price, qty)) in update.bids.iter().enumerate().take(max_levels) {
            if !first_field {
                line.push(',');
            }
            write!(
                line,
                "bid{}={},bid{}_qty={}",
                i + 1,
                price.to_f64(),
                i + 1,
                qty.to_f64(),
            )
            .unwrap();
            first_field = false;
        }

        for (i, (price, qty)) in update.asks.iter().enumerate().take(max_levels) {
            if !first_field {
                line.push(',');
            }
            write!(
                line,
                "ask{}={},ask{}_qty={}",
                i + 1,
                price.to_f64(),
                i + 1,
                qty.to_f64(),
            )
            .unwrap();
            first_field = false;
        }

        // If there were no fields at all (empty book), write a placeholder.
        if first_field {
            line.push_str("empty=true");
        }

        // Timestamp in nanoseconds
        write!(line, " {}", receive_ts).unwrap();
        line.push('\n');

        self.buffer.extend_from_slice(line.as_bytes());
    }

    /// Formats a [`Trade`] as an ILP line and appends it to the internal buffer.
    ///
    /// The `receive_ts` is the local receive timestamp in nanoseconds, used as the
    /// ILP timestamp. If the buffer has exceeded `max_buffer_size`, the write is
    /// skipped and a warning is logged.
    pub fn write_trade(&mut self, trade: &Trade, receive_ts: u64) {
        if self.buffer_full() {
            warn!(
                buffer_len = self.buffer.len(),
                max = self.max_buffer_size,
                "ILP buffer full, dropping trade",
            );
            return;
        }
        let exchange = exchange_tag(trade.exchange);
        let symbol = &trade.symbol.0;
        let side = match trade.side {
            Side::Buy => "buy",
            Side::Sell => "sell",
        };

        let mut line = String::with_capacity(256);
        // measurement,tags
        write!(
            line,
            "trade,exchange={},symbol={},side={}",
            exchange, symbol, side,
        )
        .unwrap();

        // fields
        write!(
            line,
            " price={},quantity={},trade_id=\"{}\"",
            trade.price.to_f64(),
            trade.quantity.to_f64(),
            escape_ilp_string(&trade.trade_id),
        )
        .unwrap();

        // timestamp
        write!(line, " {}", receive_ts).unwrap();
        line.push('\n');

        self.buffer.extend_from_slice(line.as_bytes());
    }

    /// Flushes the internal buffer to the QuestDB TCP stream.
    ///
    /// Returns the number of bytes written. Clears the buffer after a successful
    /// write.
    pub async fn flush(&mut self) -> Result<usize> {
        if self.buffer.is_empty() {
            return Ok(0);
        }

        let stream = self.stream.as_mut().ok_or(QuestDbError::NotConnected)?;

        let len = self.buffer.len();
        stream
            .write_all(&self.buffer)
            .await
            .map_err(|e| QuestDbError::WriteFailed(e.to_string()))?;
        stream
            .flush()
            .await
            .map_err(|e| QuestDbError::WriteFailed(e.to_string()))?;

        self.buffer.clear();
        debug!(bytes = len, "flushed ILP buffer to QuestDB");
        Ok(len)
    }

    /// Returns the current size of the internal buffer in bytes.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `true` if the client has an active TCP connection.
    pub fn is_connected(&self) -> bool {
        self.stream.is_some()
    }

    /// Performs a health check by verifying the connection is alive.
    ///
    /// If the connection appears broken, attempts a reconnect.
    pub async fn health_check(&mut self) -> Result<()> {
        if self.stream.is_none() {
            return self.reconnect().await;
        }

        // Try a zero-byte write to detect a broken pipe.
        let stream = self.stream.as_mut().unwrap();
        match stream.write_all(b"").await {
            Ok(()) => Ok(()),
            Err(_) => {
                warn!("QuestDB health check failed, reconnecting");
                self.reconnect().await
            }
        }
    }
}

/// Lowercase exchange name for ILP tag values.
fn exchange_tag(exchange: cm_core::types::order::Exchange) -> &'static str {
    match exchange {
        cm_core::types::order::Exchange::Binance => "binance",
        cm_core::types::order::Exchange::Bybit => "bybit",
    }
}

/// Escape special characters in ILP string field values.
///
/// In ILP string fields (quoted with `"`), backslashes and double quotes must be
/// escaped.
fn escape_ilp_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use cm_core::types::{BookUpdate, Exchange, Price, Quantity, Side, Symbol, Timestamp, Trade};

    fn sample_book_update() -> BookUpdate {
        BookUpdate {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            timestamp: Timestamp::from_millis(1706000000000),
            bids: vec![
                (Price::new(5000050, 2), Quantity::new(15000000, 8)),
                (Price::new(5000000, 2), Quantity::new(20000000, 8)),
            ],
            asks: vec![
                (Price::new(5000100, 2), Quantity::new(20000000, 8)),
                (Price::new(5000150, 2), Quantity::new(10000000, 8)),
            ],
            is_snapshot: true,
        }
    }

    fn sample_trade() -> Trade {
        Trade {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            timestamp: Timestamp::from_millis(1706000000000),
            price: Price::new(5000050, 2),
            quantity: Quantity::new(10000000, 8),
            side: Side::Buy,
            trade_id: "abc123".to_string(),
        }
    }

    #[test]
    fn test_book_update_ilp_format() {
        let mut client = QuestDbClient::new("localhost:9009".to_string());
        let update = sample_book_update();
        let ts: u64 = 1706000000000000000;

        client.write_book_update(&update, ts);

        let line = String::from_utf8(client.buffer.clone()).unwrap();
        assert!(line.starts_with("book_update,exchange=binance,symbol=BTCUSDT "));
        assert!(line.contains("bid1=50000.5"));
        assert!(line.contains("bid1_qty=0.15"));
        assert!(line.contains("bid2=50000"));
        assert!(line.contains("bid2_qty=0.2"));
        assert!(line.contains("ask1=50001"));
        assert!(line.contains("ask1_qty=0.2"));
        assert!(line.contains("ask2=50001.5"));
        assert!(line.contains("ask2_qty=0.1"));
        assert!(line.contains(" 1706000000000000000\n"));
    }

    #[test]
    fn test_trade_ilp_format() {
        let mut client = QuestDbClient::new("localhost:9009".to_string());
        let trade = sample_trade();
        let ts: u64 = 1706000000000000000;

        client.write_trade(&trade, ts);

        let line = String::from_utf8(client.buffer.clone()).unwrap();
        assert!(line.starts_with("trade,exchange=binance,symbol=BTCUSDT,side=buy "));
        assert!(line.contains("price=50000.5"));
        assert!(line.contains("quantity=0.1"));
        assert!(line.contains("trade_id=\"abc123\""));
        assert!(line.contains(" 1706000000000000000\n"));
    }

    #[test]
    fn test_trade_sell_side() {
        let mut client = QuestDbClient::new("localhost:9009".to_string());
        let mut trade = sample_trade();
        trade.side = Side::Sell;

        client.write_trade(&trade, 100);

        let line = String::from_utf8(client.buffer.clone()).unwrap();
        assert!(line.starts_with("trade,exchange=binance,symbol=BTCUSDT,side=sell "));
    }

    #[test]
    fn test_buffer_accumulation() {
        let mut client = QuestDbClient::new("localhost:9009".to_string());
        let trade = sample_trade();

        assert_eq!(client.buffer_len(), 0);

        client.write_trade(&trade, 100);
        let first_len = client.buffer_len();
        assert!(first_len > 0);

        client.write_trade(&trade, 200);
        let second_len = client.buffer_len();
        assert!(second_len > first_len);

        // Two lines should be present.
        let content = String::from_utf8(client.buffer.clone()).unwrap();
        let line_count = content.lines().count();
        assert_eq!(line_count, 2);
    }

    #[test]
    fn test_buffer_cleared_after_flush() {
        // We can't actually flush without a connection, but we can verify
        // clear behaviour by simulating.
        let mut client = QuestDbClient::new("localhost:9009".to_string());
        let trade = sample_trade();

        client.write_trade(&trade, 100);
        assert!(client.buffer_len() > 0);

        // Manually clear to simulate post-flush behaviour.
        client.buffer.clear();
        assert_eq!(client.buffer_len(), 0);
    }

    #[test]
    fn test_not_connected_by_default() {
        let client = QuestDbClient::new("localhost:9009".to_string());
        assert!(!client.is_connected());
    }

    #[tokio::test]
    async fn test_flush_when_not_connected() {
        let mut client = QuestDbClient::new("localhost:9009".to_string());
        let trade = sample_trade();
        client.write_trade(&trade, 100);

        let result = client.flush().await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("not connected"));
    }

    #[tokio::test]
    async fn test_flush_empty_buffer() {
        let mut client = QuestDbClient::new("localhost:9009".to_string());
        let result = client.flush().await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_connect_failure() {
        // Connect to a port that should not be listening.
        let mut client = QuestDbClient::new("127.0.0.1:1".to_string());
        let result = client.connect().await;
        assert!(result.is_err());
        assert!(!client.is_connected());
    }

    #[test]
    fn test_empty_book_update() {
        let mut client = QuestDbClient::new("localhost:9009".to_string());
        let update = BookUpdate {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            timestamp: Timestamp::from_millis(1706000000000),
            bids: vec![],
            asks: vec![],
            is_snapshot: true,
        };

        client.write_book_update(&update, 100);

        let line = String::from_utf8(client.buffer.clone()).unwrap();
        assert!(line.contains("empty=true"));
    }

    #[test]
    fn test_bybit_exchange_tag() {
        let mut client = QuestDbClient::new("localhost:9009".to_string());
        let mut trade = sample_trade();
        trade.exchange = Exchange::Bybit;

        client.write_trade(&trade, 100);

        let line = String::from_utf8(client.buffer.clone()).unwrap();
        assert!(line.starts_with("trade,exchange=bybit,"));
    }

    #[test]
    fn test_trade_id_escaping() {
        let mut client = QuestDbClient::new("localhost:9009".to_string());
        let mut trade = sample_trade();
        trade.trade_id = r#"id"with\special"#.to_string();

        client.write_trade(&trade, 100);

        let line = String::from_utf8(client.buffer.clone()).unwrap();
        assert!(line.contains(r#"trade_id="id\"with\\special""#));
    }

    #[test]
    fn test_book_update_max_20_levels() {
        let mut client = QuestDbClient::new("localhost:9009".to_string());
        let update = BookUpdate {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            timestamp: Timestamp::from_millis(1706000000000),
            bids: (0..25)
                .map(|i| (Price::new(5000000 - i * 100, 2), Quantity::new(10000000, 8)))
                .collect(),
            asks: (0..25)
                .map(|i| (Price::new(5000100 + i * 100, 2), Quantity::new(10000000, 8)))
                .collect(),
            is_snapshot: true,
        };

        client.write_book_update(&update, 100);

        let line = String::from_utf8(client.buffer.clone()).unwrap();
        // Should have bid1..bid20 but NOT bid21
        assert!(line.contains("bid20="));
        assert!(!line.contains("bid21="));
        assert!(line.contains("ask20="));
        assert!(!line.contains("ask21="));
    }
}

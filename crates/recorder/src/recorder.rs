//! Market data recorder that batches events and flushes to QuestDB.
//!
//! The [`Recorder`] runs an async event loop that receives market data events
//! through an mpsc channel, buffers them via the QuestDB ILP client, and
//! periodically flushes to QuestDB over TCP.

use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use cm_core::types::{BookUpdate, Trade};

use crate::questdb::QuestDbClient;

/// Configuration for the market data recorder.
#[derive(Debug, Clone)]
pub struct RecorderConfig {
    /// QuestDB ILP address (`host:port`).
    pub questdb_addr: String,
    /// Maximum events to buffer before forcing a flush.
    pub batch_size: usize,
    /// Flush interval in milliseconds.
    pub flush_interval_ms: u64,
    /// Channel capacity for incoming events.
    pub channel_capacity: usize,
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self {
            questdb_addr: "localhost:9009".to_string(),
            batch_size: 1000,
            flush_interval_ms: 100,
            channel_capacity: 10_000,
        }
    }
}

/// Events that can be recorded.
#[derive(Debug, Clone)]
pub enum RecordEvent {
    /// An L2 order book update.
    BookUpdate {
        /// The book update payload.
        update: BookUpdate,
        /// Local receive timestamp in nanoseconds.
        receive_timestamp_ns: u64,
    },
    /// A trade event.
    Trade {
        /// The trade payload.
        trade: Trade,
        /// Local receive timestamp in nanoseconds.
        receive_timestamp_ns: u64,
    },
}

/// Metrics tracked by the recorder.
#[derive(Debug, Clone, Default)]
pub struct RecorderMetrics {
    /// Total events received from the channel.
    pub events_received: u64,
    /// Total events successfully flushed to QuestDB.
    pub events_flushed: u64,
    /// Total events dropped (channel full or flush failure).
    pub events_dropped: u64,
    /// Number of flush operations performed.
    pub flush_count: u64,
    /// Duration of the last flush in microseconds.
    pub last_flush_duration_us: u64,
}

/// Market data recorder that batches events and flushes to QuestDB.
///
/// Create via [`Recorder::new`], which returns both the recorder and a
/// clonable [`RecorderHandle`] for sending events from other tasks.
pub struct Recorder {
    config: RecorderConfig,
    rx: mpsc::Receiver<RecordEvent>,
    client: QuestDbClient,
    metrics: RecorderMetrics,
    pending_events: usize,
}

/// Clonable handle for sending events to the [`Recorder`].
///
/// Obtained from [`Recorder::new`]. Dropping all handles causes the recorder's
/// event loop to drain remaining events and shut down.
#[derive(Clone)]
pub struct RecorderHandle {
    tx: mpsc::Sender<RecordEvent>,
}

impl Recorder {
    /// Creates a new recorder and its associated [`RecorderHandle`].
    pub fn new(config: RecorderConfig) -> (Self, RecorderHandle) {
        let (tx, rx) = mpsc::channel(config.channel_capacity);
        let client = QuestDbClient::new(config.questdb_addr.clone());

        let recorder = Self {
            config,
            rx,
            client,
            metrics: RecorderMetrics::default(),
            pending_events: 0,
        };

        let handle = RecorderHandle { tx };
        (recorder, handle)
    }

    /// Returns a snapshot of the current recorder metrics.
    pub fn metrics(&self) -> &RecorderMetrics {
        &self.metrics
    }

    /// Runs the recorder event loop.
    ///
    /// 1. Connects to QuestDB.
    /// 2. Receives events from the channel, buffering them in the ILP client.
    /// 3. Flushes on a timer tick or when the batch size is reached.
    /// 4. On channel close, flushes remaining data and returns.
    pub async fn run(&mut self) -> Result<()> {
        // Attempt initial connection — log and continue if QuestDB is not up.
        if let Err(e) = self.client.connect().await {
            warn!(error = %e, "initial QuestDB connection failed, will retry on flush");
        }

        let flush_interval = Duration::from_millis(self.config.flush_interval_ms);
        let mut interval = tokio::time::interval(flush_interval);
        // Don't try to "catch up" if we fall behind.
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        info!(
            addr = %self.config.questdb_addr,
            batch_size = self.config.batch_size,
            flush_interval_ms = self.config.flush_interval_ms,
            "recorder started",
        );

        loop {
            tokio::select! {
                // Prefer draining the channel over ticking.
                biased;

                event = self.rx.recv() => {
                    match event {
                        Some(ev) => {
                            self.handle_event(ev);

                            // Flush immediately if batch size is reached.
                            if self.pending_events >= self.config.batch_size {
                                self.do_flush().await;
                            }
                        }
                        None => {
                            // Channel closed — all handles dropped.
                            info!("recorder channel closed, flushing remaining data");
                            self.do_flush().await;
                            return Ok(());
                        }
                    }
                }

                _ = interval.tick() => {
                    if self.pending_events > 0 {
                        self.do_flush().await;
                    }
                }
            }
        }
    }

    /// Buffers a single event in the QuestDB ILP client.
    fn handle_event(&mut self, event: RecordEvent) {
        self.metrics.events_received += 1;

        match event {
            RecordEvent::BookUpdate {
                update,
                receive_timestamp_ns,
            } => {
                self.client.write_book_update(&update, receive_timestamp_ns);
            }
            RecordEvent::Trade {
                trade,
                receive_timestamp_ns,
            } => {
                self.client.write_trade(&trade, receive_timestamp_ns);
            }
        }

        self.pending_events += 1;
    }

    /// Flushes the buffer to QuestDB, handling errors gracefully.
    async fn do_flush(&mut self) {
        let events_in_batch = self.pending_events;
        let start = Instant::now();

        match self.client.flush().await {
            Ok(bytes) => {
                let elapsed = start.elapsed();
                self.metrics.events_flushed += events_in_batch as u64;
                self.metrics.flush_count += 1;
                self.metrics.last_flush_duration_us = elapsed.as_micros() as u64;
                self.pending_events = 0;

                debug!(
                    events = events_in_batch,
                    bytes,
                    duration_us = self.metrics.last_flush_duration_us,
                    "flush complete",
                );
            }
            Err(e) => {
                warn!(error = %e, events = events_in_batch, "flush failed, attempting reconnect");
                self.metrics.events_dropped += events_in_batch as u64;
                self.pending_events = 0;

                // Attempt reconnect. If it fails too, we just log and keep going.
                if let Err(re) = self.client.reconnect().await {
                    warn!(error = %re, "reconnect failed, events will be dropped until recovery");
                }
            }
        }
    }
}

impl RecorderHandle {
    /// Sends an event to the recorder without blocking.
    ///
    /// Uses `try_send` to avoid backpressure — if the channel is full the event
    /// is dropped and a warning is logged. This ensures the hot path is never
    /// blocked by slow QuestDB writes.
    pub fn record(&self, event: RecordEvent) -> Result<()> {
        match self.tx.try_send(event) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(_)) => {
                warn!("recorder channel full, dropping event");
                Ok(())
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                Err(anyhow::anyhow!("recorder channel closed"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cm_core::types::{Exchange, Price, Quantity, Side, Symbol, Timestamp};

    fn sample_trade_event() -> RecordEvent {
        RecordEvent::Trade {
            trade: Trade {
                exchange: Exchange::Binance,
                symbol: Symbol::new("BTCUSDT"),
                timestamp: Timestamp::from_millis(1706000000000),
                price: Price::new(5000050, 2),
                quantity: Quantity::new(10000000, 8),
                side: Side::Buy,
                trade_id: "t1".to_string(),
            },
            receive_timestamp_ns: 1706000000000000000,
        }
    }

    fn sample_book_event() -> RecordEvent {
        RecordEvent::BookUpdate {
            update: BookUpdate {
                exchange: Exchange::Binance,
                symbol: Symbol::new("BTCUSDT"),
                timestamp: Timestamp::from_millis(1706000000000),
                bids: vec![(Price::new(5000050, 2), Quantity::new(15000000, 8))],
                asks: vec![(Price::new(5000100, 2), Quantity::new(20000000, 8))],
                is_snapshot: true,
            },
            receive_timestamp_ns: 1706000000000000000,
        }
    }

    #[tokio::test]
    async fn test_handle_sends_events_to_channel() {
        let config = RecorderConfig {
            channel_capacity: 10,
            ..Default::default()
        };
        let (_recorder, handle) = Recorder::new(config);

        // Should successfully send events.
        assert!(handle.record(sample_trade_event()).is_ok());
        assert!(handle.record(sample_book_event()).is_ok());
    }

    #[tokio::test]
    async fn test_backpressure_drops_events() {
        let config = RecorderConfig {
            channel_capacity: 2,
            ..Default::default()
        };
        let (_recorder, handle) = Recorder::new(config);

        // Fill the channel.
        handle.record(sample_trade_event()).unwrap();
        handle.record(sample_trade_event()).unwrap();

        // Third event should be dropped (not error), channel full.
        let result = handle.record(sample_trade_event());
        assert!(result.is_ok()); // Dropped gracefully, not an error.
    }

    #[tokio::test]
    async fn test_handle_event_increments_received() {
        let config = RecorderConfig::default();
        let (mut recorder, _handle) = Recorder::new(config);

        recorder.handle_event(sample_trade_event());
        recorder.handle_event(sample_book_event());

        assert_eq!(recorder.metrics.events_received, 2);
        assert_eq!(recorder.pending_events, 2);
    }

    #[tokio::test]
    async fn test_handle_event_buffers_in_client() {
        let config = RecorderConfig::default();
        let (mut recorder, _handle) = Recorder::new(config);

        assert_eq!(recorder.client.buffer_len(), 0);

        recorder.handle_event(sample_trade_event());
        assert!(recorder.client.buffer_len() > 0);
    }

    #[tokio::test]
    async fn test_flush_failure_increments_dropped() {
        let config = RecorderConfig::default();
        let (mut recorder, _handle) = Recorder::new(config);
        // Client is not connected, so flush will fail.

        recorder.handle_event(sample_trade_event());
        recorder.handle_event(sample_trade_event());
        recorder.handle_event(sample_trade_event());

        assert_eq!(recorder.pending_events, 3);

        recorder.do_flush().await;

        assert_eq!(recorder.metrics.events_dropped, 3);
        assert_eq!(recorder.metrics.events_flushed, 0);
        assert_eq!(recorder.pending_events, 0);
    }

    #[tokio::test]
    async fn test_run_exits_on_channel_close() {
        let config = RecorderConfig {
            questdb_addr: "127.0.0.1:1".to_string(), // won't connect
            channel_capacity: 10,
            flush_interval_ms: 10,
            ..Default::default()
        };
        let (mut recorder, handle) = Recorder::new(config);

        // Send some events then drop the handle to close the channel.
        handle.record(sample_trade_event()).unwrap();
        handle.record(sample_book_event()).unwrap();
        drop(handle);

        // run() should return once the channel is drained.
        let result = recorder.run().await;
        assert!(result.is_ok());
        assert_eq!(recorder.metrics.events_received, 2);
    }

    #[tokio::test]
    async fn test_batch_size_triggers_flush() {
        let config = RecorderConfig {
            questdb_addr: "127.0.0.1:1".to_string(),
            batch_size: 2,
            channel_capacity: 10,
            flush_interval_ms: 10000, // Very long so timer won't fire.
        };
        let (mut recorder, handle) = Recorder::new(config);

        // Send batch_size events and then close.
        handle.record(sample_trade_event()).unwrap();
        handle.record(sample_trade_event()).unwrap();
        drop(handle);

        let _ = recorder.run().await;

        // Events should have been attempted (dropped since no connection).
        assert_eq!(recorder.metrics.events_received, 2);
        // Flush was attempted at least once (batch trigger + final drain).
        assert!(recorder.metrics.events_dropped > 0 || recorder.metrics.events_flushed > 0);
    }

    #[tokio::test]
    async fn test_metrics_initial_state() {
        let config = RecorderConfig::default();
        let (recorder, _handle) = Recorder::new(config);
        let m = recorder.metrics();

        assert_eq!(m.events_received, 0);
        assert_eq!(m.events_flushed, 0);
        assert_eq!(m.events_dropped, 0);
        assert_eq!(m.flush_count, 0);
        assert_eq!(m.last_flush_duration_us, 0);
    }

    #[tokio::test]
    async fn test_channel_closed_error() {
        let config = RecorderConfig {
            channel_capacity: 10,
            ..Default::default()
        };
        let (recorder, handle) = Recorder::new(config);
        drop(recorder); // Drop the receiver side.

        let result = handle.record(sample_trade_event());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("closed"));
    }
}

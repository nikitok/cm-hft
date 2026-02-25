//! Event replay engine for backtesting.
//!
//! Reads Parquet files containing historical market data and replays events
//! in chronological order. Supports both book updates and trade events.

use std::path::PathBuf;

use anyhow::{Context, Result};
use arrow::array::{Array, Float64Array, StringArray, UInt64Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use cm_core::types::*;

/// Events that can be replayed from historical data.
#[derive(Debug, Clone)]
pub enum ReplayEvent {
    /// An L2 order book update.
    BookUpdate(BookUpdate),
    /// An individual trade event.
    Trade(Trade),
}

/// Configuration for the replay engine.
#[derive(Debug, Clone)]
pub struct ReplayConfig {
    /// Parquet data files to replay.
    pub data_paths: Vec<PathBuf>,
    /// Start timestamp filter (nanos since epoch, 0 = no filter).
    pub start_ns: u64,
    /// End timestamp filter (0 = no filter).
    pub end_ns: u64,
    /// Replay speed multiplier (1.0 = real-time, 0.0 = max speed).
    pub speed: f64,
}

/// Reads Parquet files and replays market data events in chronological order.
pub struct ReplayEngine {
    config: ReplayConfig,
    events: Vec<(u64, ReplayEvent)>,
}

impl ReplayEngine {
    /// Create a new replay engine with the given configuration.
    pub fn new(config: ReplayConfig) -> Self {
        Self {
            config,
            events: Vec::new(),
        }
    }

    /// Construct a replay engine directly from a pre-built event vector.
    ///
    /// Events are sorted by timestamp upon construction. Useful for testing
    /// and synthetic data generation.
    pub fn from_events(mut events: Vec<(u64, ReplayEvent)>) -> Self {
        events.sort_by_key(|(ts, _)| *ts);
        Self {
            config: ReplayConfig {
                data_paths: Vec::new(),
                start_ns: 0,
                end_ns: 0,
                speed: 0.0,
            },
            events,
        }
    }

    /// Load events from all configured Parquet files.
    ///
    /// Reads each file, detects whether it contains book updates or trades
    /// based on column names, parses rows into events, applies timestamp
    /// filters, and sorts chronologically.
    ///
    /// Returns the total number of events loaded.
    pub fn load(&mut self) -> Result<usize> {
        self.events.clear();

        let paths = self.config.data_paths.clone();
        for path in &paths {
            let file = std::fs::File::open(path)
                .with_context(|| format!("failed to open parquet file: {}", path.display()))?;

            let builder = ParquetRecordBatchReaderBuilder::try_new(file)
                .with_context(|| format!("failed to read parquet metadata: {}", path.display()))?;

            let reader = builder
                .build()
                .with_context(|| format!("failed to build parquet reader: {}", path.display()))?;

            for batch_result in reader {
                let batch = batch_result
                    .with_context(|| format!("failed to read batch from {}", path.display()))?;

                let schema = batch.schema();
                let field_names: Vec<&str> =
                    schema.fields().iter().map(|f| f.name().as_str()).collect();

                if field_names.contains(&"bid1") || field_names.contains(&"bid_price") {
                    self.parse_book_updates(&batch, &field_names)?;
                } else if field_names.contains(&"trade_id") {
                    self.parse_trades(&batch, &field_names)?;
                } else {
                    tracing::warn!(
                        path = %path.display(),
                        columns = ?field_names,
                        "unknown parquet schema, skipping file"
                    );
                }
            }
        }

        // Apply timestamp filters
        if self.config.start_ns > 0 || self.config.end_ns > 0 {
            self.events.retain(|(ts, _)| {
                let after_start = self.config.start_ns == 0 || *ts >= self.config.start_ns;
                let before_end = self.config.end_ns == 0 || *ts <= self.config.end_ns;
                after_start && before_end
            });
        }

        // Sort by timestamp
        self.events.sort_by_key(|(ts, _)| *ts);

        Ok(self.events.len())
    }

    /// Parse book update rows from a record batch.
    fn parse_book_updates(
        &mut self,
        batch: &arrow::record_batch::RecordBatch,
        field_names: &[&str],
    ) -> Result<()> {
        let ts_col = self.get_u64_column(batch, field_names, "timestamp_ns")
            .or_else(|_| self.get_u64_column(batch, field_names, "timestamp"))?;

        let bid1_col = self.get_f64_column(batch, field_names, "bid1")
            .or_else(|_| self.get_f64_column(batch, field_names, "bid_price"))?;
        let ask1_col = self.get_f64_column(batch, field_names, "ask1")
            .or_else(|_| self.get_f64_column(batch, field_names, "ask_price"))?;

        let bid1_qty_col = self.get_f64_column(batch, field_names, "bid1_qty")
            .or_else(|_| self.get_f64_column(batch, field_names, "bid_qty"))
            .ok();
        let ask1_qty_col = self.get_f64_column(batch, field_names, "ask1_qty")
            .or_else(|_| self.get_f64_column(batch, field_names, "ask_qty"))
            .ok();

        for i in 0..batch.num_rows() {
            let ts = ts_col.value(i);
            let bid_price = bid1_col.value(i);
            let ask_price = ask1_col.value(i);

            let bid_qty = bid1_qty_col.as_ref().map(|c| c.value(i)).unwrap_or(1.0);
            let ask_qty = ask1_qty_col.as_ref().map(|c| c.value(i)).unwrap_or(1.0);

            let update = BookUpdate {
                exchange: Exchange::Binance,
                symbol: Symbol::new("BTCUSDT"),
                timestamp: Timestamp(ts),
                bids: vec![(Price::from(bid_price), Quantity::from(bid_qty))],
                asks: vec![(Price::from(ask_price), Quantity::from(ask_qty))],
                is_snapshot: true,
            };

            self.events.push((ts, ReplayEvent::BookUpdate(update)));
        }

        Ok(())
    }

    /// Parse trade rows from a record batch.
    fn parse_trades(
        &mut self,
        batch: &arrow::record_batch::RecordBatch,
        field_names: &[&str],
    ) -> Result<()> {
        let ts_col = self.get_u64_column(batch, field_names, "timestamp_ns")
            .or_else(|_| self.get_u64_column(batch, field_names, "timestamp"))?;

        let price_col = self.get_f64_column(batch, field_names, "price")?;
        let qty_col = self.get_f64_column(batch, field_names, "quantity")
            .or_else(|_| self.get_f64_column(batch, field_names, "qty"))?;

        let side_col = self.get_string_column(batch, field_names, "side").ok();
        let trade_id_col = self.get_string_column(batch, field_names, "trade_id").ok();

        for i in 0..batch.num_rows() {
            let ts = ts_col.value(i);
            let price = price_col.value(i);
            let qty = qty_col.value(i);

            let side = side_col
                .as_ref()
                .and_then(|c| {
                    let s = c.value(i);
                    match s.to_lowercase().as_str() {
                        "buy" | "bid" => Some(Side::Buy),
                        "sell" | "ask" => Some(Side::Sell),
                        _ => None,
                    }
                })
                .unwrap_or(Side::Buy);

            let trade_id = trade_id_col
                .as_ref()
                .map(|c| c.value(i).to_string())
                .unwrap_or_else(|| i.to_string());

            let trade = Trade {
                exchange: Exchange::Binance,
                symbol: Symbol::new("BTCUSDT"),
                timestamp: Timestamp(ts),
                price: Price::from(price),
                quantity: Quantity::from(qty),
                side,
                trade_id,
            };

            self.events.push((ts, ReplayEvent::Trade(trade)));
        }

        Ok(())
    }

    /// Helper to extract a UInt64 column by name.
    fn get_u64_column<'a>(
        &self,
        batch: &'a arrow::record_batch::RecordBatch,
        field_names: &[&str],
        name: &str,
    ) -> Result<&'a UInt64Array> {
        let idx = field_names
            .iter()
            .position(|n| *n == name)
            .with_context(|| format!("column '{}' not found", name))?;
        batch
            .column(idx)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .with_context(|| format!("column '{}' is not UInt64", name))
    }

    /// Helper to extract a Float64 column by name.
    fn get_f64_column<'a>(
        &self,
        batch: &'a arrow::record_batch::RecordBatch,
        field_names: &[&str],
        name: &str,
    ) -> Result<&'a Float64Array> {
        let idx = field_names
            .iter()
            .position(|n| *n == name)
            .with_context(|| format!("column '{}' not found", name))?;
        batch
            .column(idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .with_context(|| format!("column '{}' is not Float64", name))
    }

    /// Helper to extract a String column by name.
    fn get_string_column<'a>(
        &self,
        batch: &'a arrow::record_batch::RecordBatch,
        field_names: &[&str],
        name: &str,
    ) -> Result<&'a StringArray> {
        let idx = field_names
            .iter()
            .position(|n| *n == name)
            .with_context(|| format!("column '{}' not found", name))?;
        batch
            .column(idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .with_context(|| format!("column '{}' is not String", name))
    }

    /// Returns a reference to all loaded events in chronological order.
    pub fn events(&self) -> &[(u64, ReplayEvent)] {
        &self.events
    }

    /// Returns the number of loaded events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Returns the time range `(first_ns, last_ns)` of loaded events, or `None` if empty.
    pub fn time_range(&self) -> Option<(u64, u64)> {
        if self.events.is_empty() {
            return None;
        }
        let first = self.events.first().unwrap().0;
        let last = self.events.last().unwrap().0;
        Some((first, last))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_book_event(ts: u64, bid: f64, ask: f64) -> (u64, ReplayEvent) {
        let update = BookUpdate {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            timestamp: Timestamp(ts),
            bids: vec![(Price::from(bid), Quantity::from(1.0))],
            asks: vec![(Price::from(ask), Quantity::from(1.0))],
            is_snapshot: true,
        };
        (ts, ReplayEvent::BookUpdate(update))
    }

    fn make_trade_event(ts: u64, price: f64) -> (u64, ReplayEvent) {
        let trade = Trade {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            timestamp: Timestamp(ts),
            price: Price::from(price),
            quantity: Quantity::from(0.1),
            side: Side::Buy,
            trade_id: format!("t{}", ts),
        };
        (ts, ReplayEvent::Trade(trade))
    }

    // -- 1. from_events sorts by timestamp --
    #[test]
    fn test_from_events_sorts_by_timestamp() {
        let events = vec![
            make_book_event(300, 50000.0, 50001.0),
            make_book_event(100, 49999.0, 50000.0),
            make_book_event(200, 50001.0, 50002.0),
        ];
        let engine = ReplayEngine::from_events(events);
        let timestamps: Vec<u64> = engine.events().iter().map(|(ts, _)| *ts).collect();
        assert_eq!(timestamps, vec![100, 200, 300]);
    }

    // -- 2. event_count returns correct count --
    #[test]
    fn test_event_count() {
        let events = vec![
            make_book_event(100, 50000.0, 50001.0),
            make_trade_event(200, 50000.5),
            make_book_event(300, 50001.0, 50002.0),
        ];
        let engine = ReplayEngine::from_events(events);
        assert_eq!(engine.event_count(), 3);
    }

    // -- 3. time_range returns first and last timestamps --
    #[test]
    fn test_time_range() {
        let events = vec![
            make_book_event(500, 50000.0, 50001.0),
            make_book_event(100, 49999.0, 50000.0),
            make_trade_event(900, 50000.5),
        ];
        let engine = ReplayEngine::from_events(events);
        let (first, last) = engine.time_range().unwrap();
        assert_eq!(first, 100);
        assert_eq!(last, 900);
    }

    // -- 4. empty engine --
    #[test]
    fn test_empty_engine() {
        let engine = ReplayEngine::from_events(vec![]);
        assert_eq!(engine.event_count(), 0);
        assert!(engine.time_range().is_none());
        assert!(engine.events().is_empty());
    }

    // -- 5. single event --
    #[test]
    fn test_single_event() {
        let events = vec![make_book_event(42, 50000.0, 50001.0)];
        let engine = ReplayEngine::from_events(events);
        assert_eq!(engine.event_count(), 1);
        let (first, last) = engine.time_range().unwrap();
        assert_eq!(first, 42);
        assert_eq!(last, 42);
    }

    // -- 6. mixed event types --
    #[test]
    fn test_mixed_event_types() {
        let events = vec![
            make_book_event(100, 50000.0, 50001.0),
            make_trade_event(150, 50000.5),
            make_book_event(200, 50001.0, 50002.0),
            make_trade_event(250, 50001.5),
        ];
        let engine = ReplayEngine::from_events(events);
        assert_eq!(engine.event_count(), 4);

        // Verify ordering preserved
        let timestamps: Vec<u64> = engine.events().iter().map(|(ts, _)| *ts).collect();
        assert_eq!(timestamps, vec![100, 150, 200, 250]);
    }

    // -- 7. events with same timestamp maintain stable order --
    #[test]
    fn test_same_timestamp_stable() {
        let events = vec![
            make_book_event(100, 50000.0, 50001.0),
            make_trade_event(100, 50000.5),
            make_book_event(100, 50001.0, 50002.0),
        ];
        let engine = ReplayEngine::from_events(events);
        assert_eq!(engine.event_count(), 3);
        // All have same timestamp
        for (ts, _) in engine.events() {
            assert_eq!(*ts, 100);
        }
    }

    // -- 8. new creates empty engine --
    #[test]
    fn test_new_creates_empty() {
        let config = ReplayConfig {
            data_paths: vec![],
            start_ns: 0,
            end_ns: 0,
            speed: 1.0,
        };
        let engine = ReplayEngine::new(config);
        assert_eq!(engine.event_count(), 0);
        assert!(engine.time_range().is_none());
    }

    // -- 9. load with no files returns 0 --
    #[test]
    fn test_load_no_files() {
        let config = ReplayConfig {
            data_paths: vec![],
            start_ns: 0,
            end_ns: 0,
            speed: 0.0,
        };
        let mut engine = ReplayEngine::new(config);
        let count = engine.load().unwrap();
        assert_eq!(count, 0);
    }

    // -- 10. load with nonexistent file returns error --
    #[test]
    fn test_load_nonexistent_file_errors() {
        let config = ReplayConfig {
            data_paths: vec![PathBuf::from("/nonexistent/file.parquet")],
            start_ns: 0,
            end_ns: 0,
            speed: 0.0,
        };
        let mut engine = ReplayEngine::new(config);
        assert!(engine.load().is_err());
    }

    // -- 11. large event set sorting --
    #[test]
    fn test_large_event_set_sorting() {
        let mut events = Vec::new();
        for i in (0..1000).rev() {
            events.push(make_book_event(i, 50000.0 + i as f64, 50001.0 + i as f64));
        }
        let engine = ReplayEngine::from_events(events);
        assert_eq!(engine.event_count(), 1000);

        let timestamps: Vec<u64> = engine.events().iter().map(|(ts, _)| *ts).collect();
        for i in 1..timestamps.len() {
            assert!(timestamps[i] >= timestamps[i - 1]);
        }
    }

    // -- 12. event content is preserved --
    #[test]
    fn test_event_content_preserved() {
        let events = vec![make_book_event(100, 50123.45, 50124.56)];
        let engine = ReplayEngine::from_events(events);

        match &engine.events()[0].1 {
            ReplayEvent::BookUpdate(update) => {
                assert!((update.bids[0].0.to_f64() - 50123.45).abs() < 0.01);
                assert!((update.asks[0].0.to_f64() - 50124.56).abs() < 0.01);
            }
            _ => panic!("expected BookUpdate"),
        }
    }
}

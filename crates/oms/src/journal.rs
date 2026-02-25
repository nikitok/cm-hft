//! Append-only journal for order events.
//!
//! Each line is a JSON-serialized [`OrderEvent`]. The journal supports
//! replay for crash recovery, rotation for archival, and graceful handling
//! of corrupt lines.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::order::OrderEvent;

/// Append-only journal for order events.
///
/// Each line is a JSON-serialized `OrderEvent`. Writes are flushed
/// immediately for durability.
pub struct OrderJournal {
    /// Path to the journal file.
    path: PathBuf,
    /// Buffered writer (None if journal has been rotated and not reopened).
    writer: Option<BufWriter<File>>,
}

impl OrderJournal {
    /// Create or open a journal file at the given path.
    pub fn new(path: PathBuf) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("failed to open journal at {}", path.display()))?;

        Ok(Self {
            path,
            writer: Some(BufWriter::new(file)),
        })
    }

    /// Append an order event to the journal.
    ///
    /// Serializes the event to JSON, writes it as a single line, and flushes.
    pub fn append(&mut self, event: &OrderEvent) -> Result<()> {
        let writer = self
            .writer
            .as_mut()
            .context("journal writer not available (rotated?)")?;

        let json = serde_json::to_string(event).context("failed to serialize OrderEvent")?;
        writeln!(writer, "{}", json).context("failed to write to journal")?;
        writer.flush().context("failed to flush journal")?;
        Ok(())
    }

    /// Replay all events from the journal file.
    ///
    /// Corrupt lines are skipped with a warning log. Returns all
    /// successfully deserialized events in order.
    pub fn replay(&self) -> Result<Vec<OrderEvent>> {
        let file =
            File::open(&self.path).with_context(|| format!("failed to open journal for replay"))?;
        let reader = BufReader::new(file);
        let mut events = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.context("failed to read journal line")?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<OrderEvent>(&line) {
                Ok(event) => events.push(event),
                Err(e) => {
                    tracing::warn!(
                        line_num = line_num + 1,
                        error = %e,
                        "skipping corrupt journal line"
                    );
                }
            }
        }

        Ok(events)
    }

    /// Rotate the journal: rename the current file with a date suffix and open a new one.
    ///
    /// Returns the path to the rotated (old) file.
    pub fn rotate(&mut self) -> Result<PathBuf> {
        // Close current writer
        self.writer.take();

        // Generate rotated filename with timestamp
        let now = chrono::Utc::now();
        let suffix = now.format("%Y%m%d_%H%M%S");
        let stem = self
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "journal".into());
        let ext = self
            .path
            .extension()
            .map(|s| format!(".{}", s.to_string_lossy()))
            .unwrap_or_default();
        let rotated_name = format!("{}_{}{}", stem, suffix, ext);
        let rotated_path = self
            .path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join(rotated_name);

        std::fs::rename(&self.path, &rotated_path)
            .with_context(|| format!("failed to rotate journal to {}", rotated_path.display()))?;

        // Open new file
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("failed to open new journal at {}", self.path.display()))?;
        self.writer = Some(BufWriter::new(file));

        Ok(rotated_path)
    }

    /// Count the number of events (lines) in the journal.
    pub fn event_count(&self) -> Result<usize> {
        let file = File::open(&self.path)
            .with_context(|| format!("failed to open journal for counting"))?;
        let reader = BufReader::new(file);
        let count = reader
            .lines()
            .filter_map(|l| l.ok())
            .filter(|l| !l.trim().is_empty())
            .count();
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::order::{OrderEventType, OrderStatus};
    use cm_core::types::*;
    use std::io::Write as IoWrite;

    fn make_event(order_id: u64) -> OrderEvent {
        OrderEvent {
            timestamp: Timestamp::now(),
            order_id: OrderId(order_id),
            client_order_id: format!("cm_test_{:06}", order_id),
            event_type: OrderEventType::Created,
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            side: Side::Buy,
            price: Some(Price::new(5000000, 2)),
            quantity: Some(Quantity::new(100000000, 8)),
            filled_quantity: Some(Quantity::zero(8)),
            status: OrderStatus::New,
            metadata: None,
        }
    }

    fn temp_journal_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("cm_hft_test_journal");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(format!("{}.jsonl", name))
    }

    fn cleanup(path: &PathBuf) {
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_write_and_read_back() {
        let path = temp_journal_path("write_read");
        cleanup(&path);

        {
            let mut journal = OrderJournal::new(path.clone()).unwrap();
            journal.append(&make_event(1)).unwrap();
        }

        let journal = OrderJournal::new(path.clone()).unwrap();
        let events = journal.replay().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].order_id, OrderId(1));

        cleanup(&path);
    }

    #[test]
    fn test_multiple_events_maintain_order() {
        let path = temp_journal_path("multi_order");
        cleanup(&path);

        {
            let mut journal = OrderJournal::new(path.clone()).unwrap();
            for i in 1..=5 {
                journal.append(&make_event(i)).unwrap();
            }
        }

        let journal = OrderJournal::new(path.clone()).unwrap();
        let events = journal.replay().unwrap();
        assert_eq!(events.len(), 5);
        for (i, event) in events.iter().enumerate() {
            assert_eq!(event.order_id, OrderId((i + 1) as u64));
        }

        cleanup(&path);
    }

    #[test]
    fn test_replay_returns_all_events() {
        let path = temp_journal_path("replay_all");
        cleanup(&path);

        let mut journal = OrderJournal::new(path.clone()).unwrap();
        for i in 1..=10 {
            journal.append(&make_event(i)).unwrap();
        }

        let events = journal.replay().unwrap();
        assert_eq!(events.len(), 10);

        cleanup(&path);
    }

    #[test]
    fn test_event_count() {
        let path = temp_journal_path("event_count");
        cleanup(&path);

        let mut journal = OrderJournal::new(path.clone()).unwrap();
        assert_eq!(journal.event_count().unwrap(), 0);

        for i in 1..=3 {
            journal.append(&make_event(i)).unwrap();
        }
        assert_eq!(journal.event_count().unwrap(), 3);

        cleanup(&path);
    }

    #[test]
    fn test_rotate_creates_new_file() {
        let path = temp_journal_path("rotate");
        cleanup(&path);

        let mut journal = OrderJournal::new(path.clone()).unwrap();
        journal.append(&make_event(1)).unwrap();
        journal.append(&make_event(2)).unwrap();

        let rotated = journal.rotate().unwrap();

        // Rotated file should exist with old events
        assert!(rotated.exists());
        let old_file = File::open(&rotated).unwrap();
        let old_reader = BufReader::new(old_file);
        let old_lines: Vec<_> = old_reader.lines().filter_map(|l| l.ok()).collect();
        assert_eq!(old_lines.len(), 2);

        // New file should be empty
        assert_eq!(journal.event_count().unwrap(), 0);

        // Can write to new file
        journal.append(&make_event(3)).unwrap();
        assert_eq!(journal.event_count().unwrap(), 1);

        cleanup(&path);
        let _ = std::fs::remove_file(&rotated);
    }

    #[test]
    fn test_handle_corrupt_line_gracefully() {
        let path = temp_journal_path("corrupt");
        cleanup(&path);

        // Write a valid event, then a corrupt line, then another valid event
        {
            let mut file = File::create(&path).unwrap();
            let event = make_event(1);
            let json = serde_json::to_string(&event).unwrap();
            writeln!(file, "{}", json).unwrap();
            writeln!(file, "{{this is not valid json}}").unwrap();
            let event2 = make_event(2);
            let json2 = serde_json::to_string(&event2).unwrap();
            writeln!(file, "{}", json2).unwrap();
        }

        let journal = OrderJournal::new(path.clone()).unwrap();
        let events = journal.replay().unwrap();

        // Should have 2 events, skipping the corrupt line
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].order_id, OrderId(1));
        assert_eq!(events[1].order_id, OrderId(2));

        cleanup(&path);
    }

    #[test]
    fn test_empty_journal_replay() {
        let path = temp_journal_path("empty");
        cleanup(&path);

        let journal = OrderJournal::new(path.clone()).unwrap();
        let events = journal.replay().unwrap();
        assert!(events.is_empty());

        cleanup(&path);
    }

    #[test]
    fn test_journal_persists_across_reopen() {
        let path = temp_journal_path("persist");
        cleanup(&path);

        {
            let mut journal = OrderJournal::new(path.clone()).unwrap();
            journal.append(&make_event(1)).unwrap();
        }

        {
            let mut journal = OrderJournal::new(path.clone()).unwrap();
            journal.append(&make_event(2)).unwrap();
        }

        let journal = OrderJournal::new(path.clone()).unwrap();
        let events = journal.replay().unwrap();
        assert_eq!(events.len(), 2);

        cleanup(&path);
    }

    #[test]
    fn test_event_serialization_roundtrip() {
        let path = temp_journal_path("roundtrip");
        cleanup(&path);

        let original = OrderEvent {
            timestamp: Timestamp::from_millis(1706000000000),
            order_id: OrderId(42),
            client_order_id: "cm_test_000042".into(),
            event_type: OrderEventType::Filled,
            exchange: Exchange::Bybit,
            symbol: Symbol::new("ETHUSDT"),
            side: Side::Sell,
            price: Some(Price::new(300000, 2)),
            quantity: Some(Quantity::new(1000000000, 8)),
            filled_quantity: Some(Quantity::new(1000000000, 8)),
            status: OrderStatus::Filled,
            metadata: Some("test metadata".into()),
        };

        let mut journal = OrderJournal::new(path.clone()).unwrap();
        journal.append(&original).unwrap();

        let events = journal.replay().unwrap();
        assert_eq!(events.len(), 1);
        let replayed = &events[0];
        assert_eq!(replayed.order_id, OrderId(42));
        assert_eq!(replayed.client_order_id, "cm_test_000042");
        assert_eq!(replayed.status, OrderStatus::Filled);
        assert_eq!(replayed.metadata, Some("test metadata".into()));

        cleanup(&path);
    }
}

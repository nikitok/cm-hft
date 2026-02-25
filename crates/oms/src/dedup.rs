//! Fill deduplication and client order ID generation.
//!
//! [`FillDeduplicator`] ensures that duplicate fill messages from exchanges
//! (which can occur during reconnection or retransmission) are detected and
//! discarded. It also generates unique, monotonic client order IDs.

use std::collections::HashSet;
use std::time::{Duration, Instant};

/// Generates unique client order IDs and deduplicates exchange fill messages.
///
/// Thread-safe: internal state is protected by `parking_lot::Mutex`.
pub struct FillDeduplicator {
    /// Monotonic counter for client order IDs.
    counter: std::sync::atomic::AtomicU64,
    /// Instance ID for this process (used in client order ID prefix).
    instance_id: String,
    /// Set of processed fill IDs (exchange trade IDs).
    processed_fills: parking_lot::Mutex<HashSet<String>>,
    /// Timestamps for pruning old entries.
    fill_timestamps: parking_lot::Mutex<Vec<(String, Instant)>>,
}

impl FillDeduplicator {
    /// Create a new deduplicator with the given instance ID.
    ///
    /// The instance ID is embedded in generated client order IDs to ensure
    /// uniqueness across process restarts.
    pub fn new(instance_id: String) -> Self {
        Self {
            counter: std::sync::atomic::AtomicU64::new(0),
            instance_id,
            processed_fills: parking_lot::Mutex::new(HashSet::new()),
            fill_timestamps: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// Generate the next unique client order ID.
    ///
    /// Format: `cm_{instance_id}_{counter:06}` (e.g., `cm_01_000042`).
    /// The counter is monotonically increasing and atomic.
    pub fn next_client_order_id(&self) -> String {
        let n = self
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        format!("cm_{}_{:06}", self.instance_id, n)
    }

    /// Check whether a fill ID has already been processed.
    ///
    /// Returns `true` if the fill is a duplicate (already seen).
    pub fn is_duplicate(&self, fill_id: &str) -> bool {
        self.processed_fills.lock().contains(fill_id)
    }

    /// Record a fill ID as processed.
    pub fn record_fill(&self, fill_id: String) {
        self.processed_fills.lock().insert(fill_id.clone());
        self.fill_timestamps
            .lock()
            .push((fill_id, Instant::now()));
    }

    /// Remove entries older than `max_age` to bound memory usage.
    pub fn prune(&self, max_age: Duration) {
        let cutoff = Instant::now() - max_age;
        let mut timestamps = self.fill_timestamps.lock();
        let mut fills = self.processed_fills.lock();

        timestamps.retain(|(fill_id, ts)| {
            if *ts < cutoff {
                fills.remove(fill_id);
                false
            } else {
                true
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_client_order_id_format() {
        let dedup = FillDeduplicator::new("01".into());
        assert_eq!(dedup.next_client_order_id(), "cm_01_000000");
        assert_eq!(dedup.next_client_order_id(), "cm_01_000001");
        assert_eq!(dedup.next_client_order_id(), "cm_01_000002");
    }

    #[test]
    fn test_client_order_id_monotonic() {
        let dedup = FillDeduplicator::new("test".into());
        let mut ids = Vec::new();
        for _ in 0..100 {
            ids.push(dedup.next_client_order_id());
        }
        // All unique
        let set: HashSet<_> = ids.iter().collect();
        assert_eq!(set.len(), 100);
        // Monotonic (lexicographic ordering of the formatted counter)
        for i in 1..ids.len() {
            assert!(ids[i] > ids[i - 1]);
        }
    }

    #[test]
    fn test_duplicate_detection() {
        let dedup = FillDeduplicator::new("01".into());
        assert!(!dedup.is_duplicate("fill_001"));

        dedup.record_fill("fill_001".into());
        assert!(dedup.is_duplicate("fill_001"));
    }

    #[test]
    fn test_non_duplicate_passes() {
        let dedup = FillDeduplicator::new("01".into());
        dedup.record_fill("fill_001".into());

        assert!(!dedup.is_duplicate("fill_002"));
        assert!(!dedup.is_duplicate("fill_003"));
    }

    #[test]
    fn test_prune_removes_old_entries() {
        let dedup = FillDeduplicator::new("01".into());
        dedup.record_fill("old_fill".into());

        // Sleep briefly, then prune with a very short max_age
        thread::sleep(Duration::from_millis(50));
        dedup.prune(Duration::from_millis(10));

        assert!(!dedup.is_duplicate("old_fill"));
    }

    #[test]
    fn test_prune_keeps_recent_entries() {
        let dedup = FillDeduplicator::new("01".into());
        dedup.record_fill("recent_fill".into());

        // Prune with a long max_age should keep the entry
        dedup.prune(Duration::from_secs(60));

        assert!(dedup.is_duplicate("recent_fill"));
    }

    #[test]
    fn test_concurrent_client_order_ids() {
        use std::sync::Arc;

        let dedup = Arc::new(FillDeduplicator::new("concurrent".into()));
        let mut handles = vec![];

        for _ in 0..10 {
            let dedup = dedup.clone();
            handles.push(thread::spawn(move || {
                let mut ids = Vec::new();
                for _ in 0..100 {
                    ids.push(dedup.next_client_order_id());
                }
                ids
            }));
        }

        let mut all_ids = HashSet::new();
        for h in handles {
            for id in h.join().unwrap() {
                all_ids.insert(id);
            }
        }

        // All 1000 IDs should be unique
        assert_eq!(all_ids.len(), 1000);
    }

    #[test]
    fn test_concurrent_record_and_check() {
        use std::sync::Arc;

        let dedup = Arc::new(FillDeduplicator::new("01".into()));

        // Record from one thread
        let dedup_w = dedup.clone();
        let writer = thread::spawn(move || {
            for i in 0..100 {
                dedup_w.record_fill(format!("fill_{}", i));
            }
        });

        writer.join().unwrap();

        // Check from another thread
        let dedup_r = dedup.clone();
        let reader = thread::spawn(move || {
            for i in 0..100 {
                assert!(dedup_r.is_duplicate(&format!("fill_{}", i)));
            }
        });

        reader.join().unwrap();
    }

    #[test]
    fn test_multiple_records_same_fill() {
        let dedup = FillDeduplicator::new("01".into());
        dedup.record_fill("fill_001".into());
        dedup.record_fill("fill_001".into()); // recording again is safe

        assert!(dedup.is_duplicate("fill_001"));
    }

    #[test]
    fn test_instance_id_in_client_order_id() {
        let dedup = FillDeduplicator::new("mynode".into());
        let id = dedup.next_client_order_id();
        assert!(id.starts_with("cm_mynode_"));
    }
}

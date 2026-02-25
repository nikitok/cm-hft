//! L2 order book maintaining price levels for bids and asks.
//!
//! The order book receives snapshots and incremental updates from exchange
//! feeds and maintains a consistent view of market depth. Bids are stored
//! with [`std::cmp::Reverse`] keys so that iteration over the underlying
//! [`BTreeMap`] yields prices in descending order (highest bid first).
//! Asks use natural ordering (lowest ask first).

use std::cmp::Reverse;
use std::collections::BTreeMap;

use cm_core::types::{BookLevel, BookUpdate, Exchange, Price, Quantity, Symbol};

/// Errors that can occur during order book operations.
#[derive(Debug, thiserror::Error)]
pub enum OrderBookError {
    /// The book has not received an initial snapshot.
    #[error("book not initialized — apply snapshot first")]
    NotInitialized,
    /// An update arrived with a stale (already-seen) sequence id.
    #[error("stale update: received id {received}, last was {last}")]
    StaleUpdate { received: u64, last: u64 },
    /// Best bid >= best ask, indicating a data integrity issue.
    #[error("crossed book detected: best bid {bid} >= best ask {ask}")]
    CrossedBook { bid: Price, ask: Price },
}

/// L2 order book maintaining price levels for bids and asks.
///
/// Bids are stored with `Reverse<Price>` keys so that `BTreeMap` iteration
/// yields highest-price-first ordering. Asks use natural `Price` ordering
/// (lowest first).
pub struct OrderBook {
    /// Exchange this book belongs to.
    exchange: Exchange,
    /// Symbol this book represents.
    symbol: Symbol,
    /// Bid levels: Reverse(price) -> quantity. Highest bid comes first in iteration.
    bids: BTreeMap<Reverse<Price>, Quantity>,
    /// Ask levels: price -> quantity. Lowest ask comes first in iteration.
    asks: BTreeMap<Price, Quantity>,
    /// Last update ID for sequencing (exchange-specific).
    last_update_id: u64,
    /// Whether the book has received its initial snapshot.
    initialized: bool,
}

impl OrderBook {
    /// Create a new, empty, uninitialized order book.
    pub fn new(exchange: Exchange, symbol: Symbol) -> Self {
        Self {
            exchange,
            symbol,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_update_id: 0,
            initialized: false,
        }
    }

    /// Returns `true` if the book has received an initial snapshot.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Replace the entire book contents with the given snapshot.
    ///
    /// Clears all existing levels and rebuilds from the provided bid/ask
    /// slices. Sets the book as initialized.
    pub fn apply_snapshot(
        &mut self,
        bids: &[(Price, Quantity)],
        asks: &[(Price, Quantity)],
        update_id: u64,
    ) {
        self.bids.clear();
        self.asks.clear();

        for &(price, qty) in bids {
            if !qty.is_zero() {
                self.bids.insert(Reverse(price), qty);
            }
        }
        for &(price, qty) in asks {
            if !qty.is_zero() {
                self.asks.insert(price, qty);
            }
        }

        self.last_update_id = update_id;
        self.initialized = true;
    }

    /// Apply an incremental update to the book.
    ///
    /// - If quantity is zero for a price level, that level is removed.
    /// - Otherwise the level is inserted or updated.
    /// - Rejects stale updates (update_id <= last_update_id).
    /// - The book must be initialized (snapshot applied) before deltas.
    pub fn apply_update(
        &mut self,
        update: &BookUpdate,
        update_id: u64,
    ) -> Result<(), OrderBookError> {
        if !self.initialized {
            return Err(OrderBookError::NotInitialized);
        }

        if update_id <= self.last_update_id {
            tracing::debug!(
                received = update_id,
                last = self.last_update_id,
                "dropping stale book update"
            );
            return Err(OrderBookError::StaleUpdate {
                received: update_id,
                last: self.last_update_id,
            });
        }

        for &(price, qty) in &update.bids {
            if qty.is_zero() {
                self.bids.remove(&Reverse(price));
            } else {
                self.bids.insert(Reverse(price), qty);
            }
        }

        for &(price, qty) in &update.asks {
            if qty.is_zero() {
                self.asks.remove(&price);
            } else {
                self.asks.insert(price, qty);
            }
        }

        self.last_update_id = update_id;
        Ok(())
    }

    /// Returns the highest bid level, if any.
    pub fn best_bid(&self) -> Option<BookLevel> {
        self.bids.iter().next().map(|(Reverse(price), qty)| BookLevel {
            price: *price,
            quantity: *qty,
        })
    }

    /// Returns the lowest ask level, if any.
    pub fn best_ask(&self) -> Option<BookLevel> {
        self.asks.iter().next().map(|(price, qty)| BookLevel {
            price: *price,
            quantity: *qty,
        })
    }

    /// Returns the mid-price: (best_bid + best_ask) / 2.
    ///
    /// Returns `None` if either side of the book is empty.
    pub fn mid_price(&self) -> Option<Price> {
        let bid = self.best_bid()?.price;
        let ask = self.best_ask()?.price;
        Some((bid + ask) / 2)
    }

    /// Returns the spread: best_ask - best_bid.
    ///
    /// Returns `None` if either side of the book is empty.
    pub fn spread(&self) -> Option<Price> {
        let bid = self.best_bid()?.price;
        let ask = self.best_ask()?.price;
        Some(ask - bid)
    }

    /// Returns the spread in basis points relative to the mid-price.
    ///
    /// `spread_bps = (spread / mid_price) * 10_000`
    ///
    /// Returns `None` if either side is empty or mid-price is zero.
    pub fn spread_bps(&self) -> Option<f64> {
        let spread = self.spread()?;
        let mid = self.mid_price()?;
        if mid.is_zero() {
            return None;
        }
        Some(spread.to_f64() / mid.to_f64() * 10_000.0)
    }

    /// Returns the top N bid levels in descending price order (highest first).
    pub fn bid_depth(&self, levels: usize) -> Vec<BookLevel> {
        self.bids
            .iter()
            .take(levels)
            .map(|(Reverse(price), qty)| BookLevel {
                price: *price,
                quantity: *qty,
            })
            .collect()
    }

    /// Returns the top N ask levels in ascending price order (lowest first).
    pub fn ask_depth(&self, levels: usize) -> Vec<BookLevel> {
        self.asks
            .iter()
            .take(levels)
            .map(|(price, qty)| BookLevel {
                price: *price,
                quantity: *qty,
            })
            .collect()
    }

    /// Returns the total quantity across the top N bid levels.
    pub fn bid_volume(&self, levels: usize) -> Quantity {
        self.bids
            .iter()
            .take(levels)
            .fold(Quantity::zero(8), |acc, (_, qty)| acc + *qty)
    }

    /// Returns the total quantity across the top N ask levels.
    pub fn ask_volume(&self, levels: usize) -> Quantity {
        self.asks
            .iter()
            .take(levels)
            .fold(Quantity::zero(8), |acc, (_, qty)| acc + *qty)
    }

    /// Returns `true` if the book is crossed (best_bid >= best_ask).
    ///
    /// A crossed book indicates a data integrity issue and should never
    /// occur under normal operation.
    pub fn is_crossed(&self) -> bool {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => bid.price >= ask.price,
            _ => false,
        }
    }

    /// Returns `(bid_level_count, ask_level_count)`.
    pub fn level_count(&self) -> (usize, usize) {
        (self.bids.len(), self.asks.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cm_core::types::Timestamp;

    fn make_book() -> OrderBook {
        OrderBook::new(Exchange::Binance, Symbol::new("BTCUSDT"))
    }

    fn make_update(
        bids: Vec<(Price, Quantity)>,
        asks: Vec<(Price, Quantity)>,
    ) -> BookUpdate {
        BookUpdate {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            timestamp: Timestamp::from_millis(1706000000000),
            bids,
            asks,
            is_snapshot: false,
        }
    }

    // -- 1. test_empty_book --
    #[test]
    fn test_empty_book() {
        let book = make_book();
        assert!(!book.is_initialized());
        assert!(book.best_bid().is_none());
        assert!(book.best_ask().is_none());
        assert!(book.mid_price().is_none());
        assert!(book.spread().is_none());
        assert!(book.spread_bps().is_none());
        assert!(!book.is_crossed());
        assert_eq!(book.level_count(), (0, 0));
    }

    // -- 2. test_snapshot_initializes_book --
    #[test]
    fn test_snapshot_initializes_book() {
        let mut book = make_book();
        let bids = vec![
            (Price::new(5000000, 2), Quantity::new(100000, 8)),
            (Price::new(4999900, 2), Quantity::new(200000, 8)),
        ];
        let asks = vec![
            (Price::new(5000100, 2), Quantity::new(150000, 8)),
            (Price::new(5000200, 2), Quantity::new(250000, 8)),
        ];
        book.apply_snapshot(&bids, &asks, 100);

        assert!(book.is_initialized());
        let best_bid = book.best_bid().unwrap();
        assert_eq!(best_bid.price, Price::new(5000000, 2));
        assert_eq!(best_bid.quantity, Quantity::new(100000, 8));

        let best_ask = book.best_ask().unwrap();
        assert_eq!(best_ask.price, Price::new(5000100, 2));
        assert_eq!(best_ask.quantity, Quantity::new(150000, 8));
    }

    // -- 3. test_incremental_update_adds_level --
    #[test]
    fn test_incremental_update_adds_level() {
        let mut book = make_book();
        book.apply_snapshot(
            &[(Price::new(5000000, 2), Quantity::new(100000, 8))],
            &[(Price::new(5000100, 2), Quantity::new(150000, 8))],
            100,
        );

        let update = make_update(
            vec![(Price::new(4999800, 2), Quantity::new(300000, 8))],
            vec![],
        );
        book.apply_update(&update, 101).unwrap();

        assert_eq!(book.level_count(), (2, 1));
    }

    // -- 4. test_incremental_update_removes_level --
    #[test]
    fn test_incremental_update_removes_level() {
        let mut book = make_book();
        book.apply_snapshot(
            &[
                (Price::new(5000000, 2), Quantity::new(100000, 8)),
                (Price::new(4999900, 2), Quantity::new(200000, 8)),
            ],
            &[(Price::new(5000100, 2), Quantity::new(150000, 8))],
            100,
        );

        // Remove the 49999.00 bid level by sending qty=0
        let update = make_update(
            vec![(Price::new(4999900, 2), Quantity::zero(8))],
            vec![],
        );
        book.apply_update(&update, 101).unwrap();

        assert_eq!(book.level_count(), (1, 1));
        assert_eq!(book.best_bid().unwrap().price, Price::new(5000000, 2));
    }

    // -- 5. test_incremental_update_modifies_level --
    #[test]
    fn test_incremental_update_modifies_level() {
        let mut book = make_book();
        book.apply_snapshot(
            &[(Price::new(5000000, 2), Quantity::new(100000, 8))],
            &[(Price::new(5000100, 2), Quantity::new(150000, 8))],
            100,
        );

        let update = make_update(
            vec![(Price::new(5000000, 2), Quantity::new(500000, 8))],
            vec![],
        );
        book.apply_update(&update, 101).unwrap();

        assert_eq!(book.level_count(), (1, 1));
        assert_eq!(
            book.best_bid().unwrap().quantity,
            Quantity::new(500000, 8)
        );
    }

    // -- 6. test_stale_update_rejected --
    #[test]
    fn test_stale_update_rejected() {
        let mut book = make_book();
        book.apply_snapshot(
            &[(Price::new(5000000, 2), Quantity::new(100000, 8))],
            &[(Price::new(5000100, 2), Quantity::new(150000, 8))],
            100,
        );

        let update = make_update(
            vec![(Price::new(4999800, 2), Quantity::new(300000, 8))],
            vec![],
        );

        // Same update_id as snapshot
        let result = book.apply_update(&update, 100);
        assert!(result.is_err());
        match result.unwrap_err() {
            OrderBookError::StaleUpdate { received, last } => {
                assert_eq!(received, 100);
                assert_eq!(last, 100);
            }
            other => panic!("expected StaleUpdate, got {:?}", other),
        }

        // Older update_id
        let result = book.apply_update(&update, 50);
        assert!(result.is_err());

        // Book unchanged
        assert_eq!(book.level_count(), (1, 1));
    }

    // -- 7. test_best_bid_ask --
    #[test]
    fn test_best_bid_ask() {
        let mut book = make_book();
        book.apply_snapshot(
            &[
                (Price::new(5000000, 2), Quantity::new(100000, 8)), // 50000.00
                (Price::new(4999900, 2), Quantity::new(200000, 8)), // 49999.00
                (Price::new(5000050, 2), Quantity::new(50000, 8)),  // 50000.50
            ],
            &[
                (Price::new(5000200, 2), Quantity::new(150000, 8)), // 50002.00
                (Price::new(5000100, 2), Quantity::new(100000, 8)), // 50001.00
                (Price::new(5000300, 2), Quantity::new(250000, 8)), // 50003.00
            ],
            100,
        );

        // Best bid should be highest: 50000.50
        assert_eq!(book.best_bid().unwrap().price, Price::new(5000050, 2));
        // Best ask should be lowest: 50001.00
        assert_eq!(book.best_ask().unwrap().price, Price::new(5000100, 2));
    }

    // -- 8. test_mid_price --
    #[test]
    fn test_mid_price() {
        let mut book = make_book();
        // bid=50000.00, ask=50001.00 => mid=50000.50
        book.apply_snapshot(
            &[(Price::new(5000000, 2), Quantity::new(100000, 8))],
            &[(Price::new(5000100, 2), Quantity::new(150000, 8))],
            100,
        );

        let mid = book.mid_price().unwrap();
        assert_eq!(mid, Price::new(5000050, 2));
    }

    // -- 9. test_spread --
    #[test]
    fn test_spread() {
        let mut book = make_book();
        // bid=50000.00, ask=50001.00 => spread=1.00
        book.apply_snapshot(
            &[(Price::new(5000000, 2), Quantity::new(100000, 8))],
            &[(Price::new(5000100, 2), Quantity::new(150000, 8))],
            100,
        );

        let spread = book.spread().unwrap();
        assert_eq!(spread, Price::new(100, 2)); // 1.00
    }

    // -- 10. test_spread_bps --
    #[test]
    fn test_spread_bps() {
        let mut book = make_book();
        // bid=50000.00, ask=50001.00
        // spread = 1.00, mid = 50000.50
        // bps = (1.00 / 50000.50) * 10000 ≈ 0.19998
        book.apply_snapshot(
            &[(Price::new(5000000, 2), Quantity::new(100000, 8))],
            &[(Price::new(5000100, 2), Quantity::new(150000, 8))],
            100,
        );

        let bps = book.spread_bps().unwrap();
        let expected = 1.0 / 50000.50 * 10_000.0;
        assert!((bps - expected).abs() < 1e-6, "bps={bps}, expected={expected}");
    }

    // -- 11. test_bid_depth --
    #[test]
    fn test_bid_depth() {
        let mut book = make_book();
        book.apply_snapshot(
            &[
                (Price::new(5000000, 2), Quantity::new(100000, 8)), // 50000.00
                (Price::new(4999900, 2), Quantity::new(200000, 8)), // 49999.00
                (Price::new(5000050, 2), Quantity::new(50000, 8)),  // 50000.50
                (Price::new(4999800, 2), Quantity::new(300000, 8)), // 49998.00
            ],
            &[(Price::new(5000100, 2), Quantity::new(150000, 8))],
            100,
        );

        let depth = book.bid_depth(3);
        assert_eq!(depth.len(), 3);
        // Should be in descending price order
        assert_eq!(depth[0].price, Price::new(5000050, 2)); // 50000.50
        assert_eq!(depth[1].price, Price::new(5000000, 2)); // 50000.00
        assert_eq!(depth[2].price, Price::new(4999900, 2)); // 49999.00
    }

    // -- 12. test_ask_depth --
    #[test]
    fn test_ask_depth() {
        let mut book = make_book();
        book.apply_snapshot(
            &[(Price::new(5000000, 2), Quantity::new(100000, 8))],
            &[
                (Price::new(5000300, 2), Quantity::new(250000, 8)), // 50003.00
                (Price::new(5000100, 2), Quantity::new(150000, 8)), // 50001.00
                (Price::new(5000200, 2), Quantity::new(200000, 8)), // 50002.00
                (Price::new(5000400, 2), Quantity::new(100000, 8)), // 50004.00
            ],
            100,
        );

        let depth = book.ask_depth(3);
        assert_eq!(depth.len(), 3);
        // Should be in ascending price order
        assert_eq!(depth[0].price, Price::new(5000100, 2)); // 50001.00
        assert_eq!(depth[1].price, Price::new(5000200, 2)); // 50002.00
        assert_eq!(depth[2].price, Price::new(5000300, 2)); // 50003.00
    }

    // -- 13. test_bid_ask_volume --
    #[test]
    fn test_bid_ask_volume() {
        let mut book = make_book();
        book.apply_snapshot(
            &[
                (Price::new(5000000, 2), Quantity::new(100000, 8)),
                (Price::new(4999900, 2), Quantity::new(200000, 8)),
                (Price::new(4999800, 2), Quantity::new(300000, 8)),
            ],
            &[
                (Price::new(5000100, 2), Quantity::new(150000, 8)),
                (Price::new(5000200, 2), Quantity::new(250000, 8)),
                (Price::new(5000300, 2), Quantity::new(350000, 8)),
            ],
            100,
        );

        // Top 2 bid levels: 100000 + 200000 = 300000
        let bid_vol = book.bid_volume(2);
        assert_eq!(bid_vol, Quantity::new(300000, 8));

        // Top 2 ask levels: 150000 + 250000 = 400000
        let ask_vol = book.ask_volume(2);
        assert_eq!(ask_vol, Quantity::new(400000, 8));

        // All 3 bid levels
        let bid_vol_all = book.bid_volume(10);
        assert_eq!(bid_vol_all, Quantity::new(600000, 8));
    }

    // -- 14. test_crossed_book_detection --
    #[test]
    fn test_crossed_book_detection() {
        let mut book = make_book();
        // Normal book: not crossed
        book.apply_snapshot(
            &[(Price::new(5000000, 2), Quantity::new(100000, 8))],
            &[(Price::new(5000100, 2), Quantity::new(150000, 8))],
            100,
        );
        assert!(!book.is_crossed());

        // Create a crossed book: bid >= ask
        book.apply_snapshot(
            &[(Price::new(5000200, 2), Quantity::new(100000, 8))], // 50002.00 bid
            &[(Price::new(5000100, 2), Quantity::new(150000, 8))], // 50001.00 ask
            200,
        );
        assert!(book.is_crossed());

        // Equal bid and ask is also crossed
        book.apply_snapshot(
            &[(Price::new(5000100, 2), Quantity::new(100000, 8))],
            &[(Price::new(5000100, 2), Quantity::new(150000, 8))],
            300,
        );
        assert!(book.is_crossed());
    }

    // -- 15. test_multiple_updates --
    #[test]
    fn test_multiple_updates() {
        let mut book = make_book();
        book.apply_snapshot(
            &[(Price::new(5000000, 2), Quantity::new(100000, 8))],
            &[(Price::new(5000100, 2), Quantity::new(150000, 8))],
            100,
        );

        // Apply 25 incremental updates
        for i in 1..=25u64 {
            let bid_price = Price::new(5000000 - (i as i64) * 10, 2);
            let ask_price = Price::new(5000100 + (i as i64) * 10, 2);
            let qty = Quantity::new((i as i64) * 10000, 8);

            let update = make_update(vec![(bid_price, qty)], vec![(ask_price, qty)]);
            book.apply_update(&update, 100 + i).unwrap();
        }

        // Original levels + 25 new ones = 26 each
        assert_eq!(book.level_count(), (26, 26));

        // Best bid is still the original 50000.00
        assert_eq!(book.best_bid().unwrap().price, Price::new(5000000, 2));
        // Best ask is still the original 50001.00
        assert_eq!(book.best_ask().unwrap().price, Price::new(5000100, 2));
    }

    // -- 16. test_snapshot_resets_book --
    #[test]
    fn test_snapshot_resets_book() {
        let mut book = make_book();
        book.apply_snapshot(
            &[
                (Price::new(5000000, 2), Quantity::new(100000, 8)),
                (Price::new(4999900, 2), Quantity::new(200000, 8)),
            ],
            &[
                (Price::new(5000100, 2), Quantity::new(150000, 8)),
                (Price::new(5000200, 2), Quantity::new(250000, 8)),
            ],
            100,
        );
        assert_eq!(book.level_count(), (2, 2));

        // Apply some updates
        let update = make_update(
            vec![(Price::new(4999800, 2), Quantity::new(300000, 8))],
            vec![(Price::new(5000300, 2), Quantity::new(350000, 8))],
        );
        book.apply_update(&update, 101).unwrap();
        assert_eq!(book.level_count(), (3, 3));

        // New snapshot should reset
        book.apply_snapshot(
            &[(Price::new(5100000, 2), Quantity::new(50000, 8))],
            &[(Price::new(5100100, 2), Quantity::new(60000, 8))],
            200,
        );

        assert_eq!(book.level_count(), (1, 1));
        assert_eq!(book.best_bid().unwrap().price, Price::new(5100000, 2));
        assert_eq!(book.best_ask().unwrap().price, Price::new(5100100, 2));
    }

    // -- 17. test_update_before_snapshot_fails --
    #[test]
    fn test_update_before_snapshot_fails() {
        let mut book = make_book();
        let update = make_update(
            vec![(Price::new(5000000, 2), Quantity::new(100000, 8))],
            vec![],
        );

        let result = book.apply_update(&update, 1);
        assert!(result.is_err());
        match result.unwrap_err() {
            OrderBookError::NotInitialized => {}
            other => panic!("expected NotInitialized, got {:?}", other),
        }
    }

    // -- 18. test_level_count --
    #[test]
    fn test_level_count() {
        let mut book = make_book();
        assert_eq!(book.level_count(), (0, 0));

        book.apply_snapshot(
            &[
                (Price::new(5000000, 2), Quantity::new(100000, 8)),
                (Price::new(4999900, 2), Quantity::new(200000, 8)),
                (Price::new(4999800, 2), Quantity::new(300000, 8)),
            ],
            &[
                (Price::new(5000100, 2), Quantity::new(150000, 8)),
                (Price::new(5000200, 2), Quantity::new(250000, 8)),
            ],
            100,
        );
        assert_eq!(book.level_count(), (3, 2));

        // Remove one bid
        let update = make_update(
            vec![(Price::new(4999800, 2), Quantity::zero(8))],
            vec![],
        );
        book.apply_update(&update, 101).unwrap();
        assert_eq!(book.level_count(), (2, 2));

        // Add one ask
        let update = make_update(
            vec![],
            vec![(Price::new(5000300, 2), Quantity::new(100000, 8))],
        );
        book.apply_update(&update, 102).unwrap();
        assert_eq!(book.level_count(), (2, 3));
    }

    // -- Additional edge case tests --

    #[test]
    fn test_bid_depth_fewer_levels_than_requested() {
        let mut book = make_book();
        book.apply_snapshot(
            &[(Price::new(5000000, 2), Quantity::new(100000, 8))],
            &[(Price::new(5000100, 2), Quantity::new(150000, 8))],
            100,
        );

        let depth = book.bid_depth(10);
        assert_eq!(depth.len(), 1);
    }

    #[test]
    fn test_ask_depth_fewer_levels_than_requested() {
        let mut book = make_book();
        book.apply_snapshot(
            &[(Price::new(5000000, 2), Quantity::new(100000, 8))],
            &[(Price::new(5000100, 2), Quantity::new(150000, 8))],
            100,
        );

        let depth = book.ask_depth(10);
        assert_eq!(depth.len(), 1);
    }

    #[test]
    fn test_snapshot_ignores_zero_quantity_levels() {
        let mut book = make_book();
        book.apply_snapshot(
            &[
                (Price::new(5000000, 2), Quantity::new(100000, 8)),
                (Price::new(4999900, 2), Quantity::zero(8)), // zero qty
            ],
            &[
                (Price::new(5000100, 2), Quantity::zero(8)), // zero qty
                (Price::new(5000200, 2), Quantity::new(200000, 8)),
            ],
            100,
        );

        assert_eq!(book.level_count(), (1, 1));
    }

    #[test]
    fn test_volume_empty_book() {
        let book = make_book();
        assert_eq!(book.bid_volume(5), Quantity::zero(8));
        assert_eq!(book.ask_volume(5), Quantity::zero(8));
    }
}

#[cfg(test)]
mod prop_tests {
    use super::*;
    use cm_core::types::Timestamp;
    use proptest::prelude::*;

    fn arb_nonzero_quantity() -> impl Strategy<Value = Quantity> {
        (1i64..1000000i64).prop_map(|m| Quantity::new(m, 8))
    }

    fn arb_bid_ask_levels(
        max_levels: usize,
    ) -> impl Strategy<Value = (Vec<(Price, Quantity)>, Vec<(Price, Quantity)>)> {
        // Generate bid prices below 50000.00 and ask prices above 50000.00
        // to avoid crossed books.
        let bids = proptest::collection::vec(
            (
                (100000i64..5000000i64).prop_map(|m| Price::new(m, 2)),
                arb_nonzero_quantity(),
            ),
            0..max_levels,
        );
        let asks = proptest::collection::vec(
            (
                (5000001i64..9999999i64).prop_map(|m| Price::new(m, 2)),
                arb_nonzero_quantity(),
            ),
            0..max_levels,
        );
        (bids, asks)
    }

    // Strategy 1: Snapshot always produces a consistent book
    proptest! {
        #[test]
        fn snapshot_produces_consistent_book(
            (bids, asks) in arb_bid_ask_levels(20),
            update_id in 1u64..10000u64,
        ) {
            let mut book = OrderBook::new(Exchange::Binance, Symbol::new("BTCUSDT"));
            book.apply_snapshot(&bids, &asks, update_id);

            prop_assert!(book.is_initialized());

            let (bid_count, ask_count) = book.level_count();

            // Deduplicate prices to find expected unique count
            let unique_bids: std::collections::BTreeSet<_> =
                bids.iter().filter(|(_, q)| !q.is_zero()).map(|(p, _)| p).collect();
            let unique_asks: std::collections::BTreeSet<_> =
                asks.iter().filter(|(_, q)| !q.is_zero()).map(|(p, _)| p).collect();

            prop_assert_eq!(bid_count, unique_bids.len());
            prop_assert_eq!(ask_count, unique_asks.len());

            // If we have both sides, the book should not be crossed
            // (our generation ensures bids < 50000 and asks > 50000)
            if bid_count > 0 && ask_count > 0 {
                prop_assert!(!book.is_crossed());
            }
        }
    }

    // Strategy 2: Removing a level with qty=0 actually removes it
    proptest! {
        #[test]
        fn zero_quantity_removes_level(
            bid_price in (100000i64..5000000i64).prop_map(|m| Price::new(m, 2)),
            ask_price in (5000001i64..9999999i64).prop_map(|m| Price::new(m, 2)),
            qty in arb_nonzero_quantity(),
        ) {
            let mut book = OrderBook::new(Exchange::Binance, Symbol::new("BTCUSDT"));
            book.apply_snapshot(
                &[(bid_price, qty)],
                &[(ask_price, qty)],
                100,
            );

            prop_assert_eq!(book.level_count(), (1, 1));

            // Remove bid
            let update = BookUpdate {
                exchange: Exchange::Binance,
                symbol: Symbol::new("BTCUSDT"),
                timestamp: Timestamp::from_millis(1706000000000),
                bids: vec![(bid_price, Quantity::zero(8))],
                asks: vec![],
                is_snapshot: false,
            };
            book.apply_update(&update, 101).unwrap();
            prop_assert_eq!(book.level_count(), (0, 1));
            prop_assert!(book.best_bid().is_none());

            // Remove ask
            let update = BookUpdate {
                exchange: Exchange::Binance,
                symbol: Symbol::new("BTCUSDT"),
                timestamp: Timestamp::from_millis(1706000000000),
                bids: vec![],
                asks: vec![(ask_price, Quantity::zero(8))],
                is_snapshot: false,
            };
            book.apply_update(&update, 102).unwrap();
            prop_assert_eq!(book.level_count(), (0, 0));
            prop_assert!(book.best_ask().is_none());
        }
    }

    // Strategy 3: Quantities are never negative after applying valid updates
    proptest! {
        #[test]
        fn quantities_never_negative_after_updates(
            initial_bids in proptest::collection::vec(
                (
                    (100000i64..5000000i64).prop_map(|m| Price::new(m, 2)),
                    arb_nonzero_quantity(),
                ),
                1..10,
            ),
            initial_asks in proptest::collection::vec(
                (
                    (5000001i64..9999999i64).prop_map(|m| Price::new(m, 2)),
                    arb_nonzero_quantity(),
                ),
                1..10,
            ),
            update_bids in proptest::collection::vec(
                (
                    (100000i64..5000000i64).prop_map(|m| Price::new(m, 2)),
                    (0i64..1000000i64).prop_map(|m| Quantity::new(m, 8)),
                ),
                0..10,
            ),
            update_asks in proptest::collection::vec(
                (
                    (5000001i64..9999999i64).prop_map(|m| Price::new(m, 2)),
                    (0i64..1000000i64).prop_map(|m| Quantity::new(m, 8)),
                ),
                0..10,
            ),
        ) {
            let mut book = OrderBook::new(Exchange::Binance, Symbol::new("BTCUSDT"));
            book.apply_snapshot(&initial_bids, &initial_asks, 100);

            let update = BookUpdate {
                exchange: Exchange::Binance,
                symbol: Symbol::new("BTCUSDT"),
                timestamp: Timestamp::from_millis(1706000000000),
                bids: update_bids,
                asks: update_asks,
                is_snapshot: false,
            };
            let _ = book.apply_update(&update, 101);

            // Verify no negative quantities in the book
            for level in book.bid_depth(1000) {
                prop_assert!(!level.quantity.is_negative());
            }
            for level in book.ask_depth(1000) {
                prop_assert!(!level.quantity.is_negative());
            }
        }
    }
}

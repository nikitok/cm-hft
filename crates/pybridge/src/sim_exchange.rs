//! Simulated exchange matching engine for backtesting.
//!
//! Provides a deterministic fill model that processes resting limit orders
//! against incoming book updates. Supports configurable fees and latency
//! simulation.

use std::collections::BTreeMap;

use cm_core::types::*;
use cm_strategy::traits::Fill;

/// Fee configuration for the simulated exchange.
#[derive(Debug, Clone)]
pub struct FeeConfig {
    /// Maker fee rate (negative = rebate). E.g., -0.0001 for -1bps.
    pub maker_fee: f64,
    /// Taker fee rate. E.g., 0.0004 for 4bps.
    pub taker_fee: f64,
}

impl Default for FeeConfig {
    fn default() -> Self {
        Self {
            maker_fee: -0.0001,
            taker_fee: 0.0004,
        }
    }
}

/// Latency simulation configuration.
#[derive(Debug, Clone)]
pub struct LatencyConfig {
    /// Fixed latency added to order submission (nanoseconds).
    pub order_latency_ns: u64,
    /// Fixed latency added to cancel (nanoseconds).
    pub cancel_latency_ns: u64,
}

impl Default for LatencyConfig {
    fn default() -> Self {
        Self {
            order_latency_ns: 1_000_000,
            cancel_latency_ns: 500_000,
        }
    }
}

/// A resting order in the simulated exchange.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SimOrder {
    id: OrderId,
    side: Side,
    price: Price,
    quantity: Quantity,
    remaining: Quantity,
    submitted_at: u64,
    available_at: u64,
}

/// Simulated exchange matching engine for backtesting.
///
/// Maintains a book of resting limit orders and matches them against
/// incoming market data updates. Fills are generated with configurable
/// fee rates and latency simulation.
pub struct SimExchange {
    fee_config: FeeConfig,
    latency_config: LatencyConfig,
    /// Resting bid orders (sorted by price descending via Reverse key).
    bids: BTreeMap<std::cmp::Reverse<i64>, Vec<SimOrder>>,
    /// Resting ask orders (sorted by price ascending).
    asks: BTreeMap<i64, Vec<SimOrder>>,
    /// Generated fills awaiting drain.
    pending_fills: Vec<Fill>,
    /// Order ID counter.
    next_order_id: u64,
    /// Current simulated time in nanoseconds.
    current_time_ns: u64,
    /// Cumulative fees paid.
    total_fees: f64,
}

impl SimExchange {
    /// Create a new simulated exchange with the given fee and latency config.
    pub fn new(fee_config: FeeConfig, latency_config: LatencyConfig) -> Self {
        Self {
            fee_config,
            latency_config,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            pending_fills: Vec::new(),
            next_order_id: 1,
            current_time_ns: 0,
            total_fees: 0.0,
        }
    }

    /// Advance the simulated clock.
    pub fn set_time(&mut self, time_ns: u64) {
        self.current_time_ns = time_ns;
    }

    /// Submit a limit order to the exchange.
    ///
    /// The order becomes active after `order_latency_ns` has elapsed from
    /// the current time. Returns the assigned order ID.
    pub fn submit_order(&mut self, side: Side, price: Price, quantity: Quantity) -> OrderId {
        let id = OrderId(self.next_order_id);
        self.next_order_id += 1;

        let order = SimOrder {
            id,
            side,
            price,
            quantity,
            remaining: quantity,
            submitted_at: self.current_time_ns,
            available_at: self.current_time_ns + self.latency_config.order_latency_ns,
        };

        // Use mantissa as the price key for BTreeMap ordering
        let price_key = price.mantissa();

        match side {
            Side::Buy => {
                self.bids
                    .entry(std::cmp::Reverse(price_key))
                    .or_default()
                    .push(order);
            }
            Side::Sell => {
                self.asks.entry(price_key).or_default().push(order);
            }
        }

        id
    }

    /// Cancel a specific order by ID. Returns `true` if found and removed.
    pub fn cancel_order(&mut self, order_id: OrderId) -> bool {
        for (_, orders) in self.bids.iter_mut() {
            if let Some(pos) = orders.iter().position(|o| o.id == order_id) {
                orders.remove(pos);
                return true;
            }
        }
        for (_, orders) in self.asks.iter_mut() {
            if let Some(pos) = orders.iter().position(|o| o.id == order_id) {
                orders.remove(pos);
                return true;
            }
        }
        // Clean up empty price levels
        self.bids.retain(|_, v| !v.is_empty());
        self.asks.retain(|_, v| !v.is_empty());
        false
    }

    /// Cancel all resting orders.
    pub fn cancel_all(&mut self) {
        self.bids.clear();
        self.asks.clear();
    }

    /// Process a book update and generate fills for any crossing orders.
    ///
    /// For resting bids: fill if the best ask price crosses (ask <= bid price)
    /// and the order has passed its latency window (available_at <= current_time).
    ///
    /// For resting asks: fill if the best bid price crosses (bid >= ask price)
    /// and the order is active.
    pub fn on_book_update(&mut self, book: &BookUpdate) {
        // Get best bid/ask from the incoming book update
        let best_ask = book.asks.iter().map(|(p, _)| *p).min();
        let best_bid = book.bids.iter().map(|(p, _)| *p).max();

        let best_ask_qty = best_ask.and_then(|target_price| {
            book.asks
                .iter()
                .find(|(p, _)| *p == target_price)
                .map(|(_, q)| *q)
        });

        let best_bid_qty = best_bid.and_then(|target_price| {
            book.bids
                .iter()
                .find(|(p, _)| *p == target_price)
                .map(|(_, q)| *q)
        });

        // Check resting bids against incoming best ask
        if let (Some(ask_price), Some(ask_qty)) = (best_ask, best_ask_qty) {
            let mut fills_to_add = Vec::new();
            let mut levels_to_remove = Vec::new();
            let mut remaining_ask_qty = ask_qty.to_f64();

            for (&std::cmp::Reverse(bid_price_key), orders) in self.bids.iter_mut() {
                let bid_price_mantissa = bid_price_key;
                // Check if bid crosses the ask
                // We need to compare using the same scale; use the actual Price stored in orders
                if let Some(first_order) = orders.first() {
                    if first_order.price < ask_price {
                        break; // No more crossing since bids are sorted descending
                    }
                } else {
                    continue;
                }

                if remaining_ask_qty <= 0.0 {
                    break;
                }
                let mut indices_to_remove = Vec::new();

                for (idx, order) in orders.iter_mut().enumerate() {
                    // Order must be active (past latency)
                    if order.available_at > self.current_time_ns {
                        continue;
                    }

                    if remaining_ask_qty <= 0.0 {
                        break;
                    }

                    let order_remaining = order.remaining.to_f64();
                    let fill_qty = order_remaining.min(remaining_ask_qty);
                    remaining_ask_qty -= fill_qty;

                    let fill_price = order.price.to_f64();
                    let fee = fill_price * fill_qty * self.fee_config.maker_fee;

                    fills_to_add.push(Fill {
                        order_id: order.id,
                        exchange: Exchange::Binance,
                        symbol: Symbol::new("BTCUSDT"),
                        side: Side::Buy,
                        price: order.price,
                        quantity: Quantity::from(fill_qty),
                        timestamp: Timestamp(self.current_time_ns),
                        is_maker: true,
                    });
                    self.total_fees += fee;

                    if fill_qty >= order_remaining - 1e-12 {
                        indices_to_remove.push(idx);
                    } else {
                        order.remaining = Quantity::from(order_remaining - fill_qty);
                    }
                }

                // Remove fully filled orders (in reverse to preserve indices)
                for idx in indices_to_remove.into_iter().rev() {
                    orders.remove(idx);
                }

                if orders.is_empty() {
                    levels_to_remove.push(std::cmp::Reverse(bid_price_mantissa));
                }
            }

            for key in levels_to_remove {
                self.bids.remove(&key);
            }
            self.pending_fills.extend(fills_to_add);
        }

        // Check resting asks against incoming best bid
        if let (Some(bid_price), Some(bid_qty)) = (best_bid, best_bid_qty) {
            let mut fills_to_add = Vec::new();
            let mut levels_to_remove = Vec::new();
            let mut remaining_bid_qty = bid_qty.to_f64();

            for (&ask_price_key, orders) in self.asks.iter_mut() {
                // Check if ask crosses the bid
                if let Some(first_order) = orders.first() {
                    if first_order.price > bid_price {
                        break; // No more crossing since asks are sorted ascending
                    }
                } else {
                    continue;
                }

                if remaining_bid_qty <= 0.0 {
                    break;
                }
                let mut indices_to_remove = Vec::new();

                for (idx, order) in orders.iter_mut().enumerate() {
                    if order.available_at > self.current_time_ns {
                        continue;
                    }

                    if remaining_bid_qty <= 0.0 {
                        break;
                    }

                    let order_remaining = order.remaining.to_f64();
                    let fill_qty = order_remaining.min(remaining_bid_qty);
                    remaining_bid_qty -= fill_qty;

                    let fill_price = order.price.to_f64();
                    let fee = fill_price * fill_qty * self.fee_config.maker_fee;

                    fills_to_add.push(Fill {
                        order_id: order.id,
                        exchange: Exchange::Binance,
                        symbol: Symbol::new("BTCUSDT"),
                        side: Side::Sell,
                        price: order.price,
                        quantity: Quantity::from(fill_qty),
                        timestamp: Timestamp(self.current_time_ns),
                        is_maker: true,
                    });
                    self.total_fees += fee;

                    if fill_qty >= order_remaining - 1e-12 {
                        indices_to_remove.push(idx);
                    } else {
                        order.remaining = Quantity::from(order_remaining - fill_qty);
                    }
                }

                for idx in indices_to_remove.into_iter().rev() {
                    orders.remove(idx);
                }

                if orders.is_empty() {
                    levels_to_remove.push(ask_price_key);
                }
            }

            for key in levels_to_remove {
                self.asks.remove(&key);
            }
            self.pending_fills.extend(fills_to_add);
        }
    }

    /// Drain all pending fills, returning them and clearing the buffer.
    pub fn drain_fills(&mut self) -> Vec<Fill> {
        std::mem::take(&mut self.pending_fills)
    }

    /// Returns the total number of resting orders across all price levels.
    pub fn open_order_count(&self) -> usize {
        let bid_count: usize = self.bids.values().map(|v| v.len()).sum();
        let ask_count: usize = self.asks.values().map(|v| v.len()).sum();
        bid_count + ask_count
    }

    /// Returns the cumulative fees paid (positive = cost, negative = rebate).
    pub fn total_fees(&self) -> f64 {
        self.total_fees
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_exchange() -> SimExchange {
        SimExchange::new(FeeConfig::default(), LatencyConfig::default())
    }

    fn zero_latency_exchange() -> SimExchange {
        SimExchange::new(
            FeeConfig::default(),
            LatencyConfig {
                order_latency_ns: 0,
                cancel_latency_ns: 0,
            },
        )
    }

    fn make_book_update(bid: f64, bid_qty: f64, ask: f64, ask_qty: f64) -> BookUpdate {
        BookUpdate {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            timestamp: Timestamp(1_000_000_000),
            bids: vec![(Price::from(bid), Quantity::from(bid_qty))],
            asks: vec![(Price::from(ask), Quantity::from(ask_qty))],
            is_snapshot: true,
        }
    }

    // -- 1. Submit limit order, feed matching book update, fill generated --
    #[test]
    fn test_submit_and_fill_bid() {
        let mut exchange = zero_latency_exchange();
        exchange.set_time(1_000_000);

        // Submit a bid at 50000
        let oid = exchange.submit_order(Side::Buy, Price::from(50000.0), Quantity::from(0.1));

        // Book update with ask at 49999 (crosses our bid)
        let book = make_book_update(49998.0, 1.0, 49999.0, 1.0);
        exchange.on_book_update(&book);

        let fills = exchange.drain_fills();
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].order_id, oid);
        assert_eq!(fills[0].side, Side::Buy);
        assert!(fills[0].is_maker);
        assert!((fills[0].quantity.to_f64() - 0.1).abs() < 1e-8);
    }

    // -- 2. Submit order that doesn't cross, no fill --
    #[test]
    fn test_no_fill_when_no_cross() {
        let mut exchange = zero_latency_exchange();
        exchange.set_time(1_000_000);

        // Submit a bid at 49000 (well below market)
        exchange.submit_order(Side::Buy, Price::from(49000.0), Quantity::from(0.1));

        // Book update with ask at 50000 (doesn't cross our bid)
        let book = make_book_update(49900.0, 1.0, 50000.0, 1.0);
        exchange.on_book_update(&book);

        let fills = exchange.drain_fills();
        assert!(fills.is_empty());
        assert_eq!(exchange.open_order_count(), 1);
    }

    // -- 3. Cancel order removes it --
    #[test]
    fn test_cancel_order() {
        let mut exchange = zero_latency_exchange();
        exchange.set_time(1_000_000);

        let oid = exchange.submit_order(Side::Buy, Price::from(50000.0), Quantity::from(0.1));
        assert_eq!(exchange.open_order_count(), 1);

        let cancelled = exchange.cancel_order(oid);
        assert!(cancelled);
        assert_eq!(exchange.open_order_count(), 0);
    }

    // -- 4. Cancel nonexistent order returns false --
    #[test]
    fn test_cancel_nonexistent() {
        let mut exchange = zero_latency_exchange();
        let cancelled = exchange.cancel_order(OrderId(999));
        assert!(!cancelled);
    }

    // -- 5. Maker fee applied correctly --
    #[test]
    fn test_maker_fee() {
        let mut exchange = SimExchange::new(
            FeeConfig {
                maker_fee: -0.0001,
                taker_fee: 0.0004,
            },
            LatencyConfig {
                order_latency_ns: 0,
                cancel_latency_ns: 0,
            },
        );
        exchange.set_time(1_000_000);

        // Submit bid at 50000, qty 1.0
        exchange.submit_order(Side::Buy, Price::from(50000.0), Quantity::from(1.0));

        // Matching ask at 49999
        let book = make_book_update(49998.0, 10.0, 49999.0, 10.0);
        exchange.on_book_update(&book);

        let fills = exchange.drain_fills();
        assert_eq!(fills.len(), 1);

        // Fee = 50000 * 1.0 * (-0.0001) = -5.0 (rebate)
        let expected_fee = 50000.0 * 1.0 * (-0.0001);
        assert!(
            (exchange.total_fees() - expected_fee).abs() < 0.01,
            "total_fees={}, expected={}",
            exchange.total_fees(),
            expected_fee
        );
    }

    // -- 6. Latency prevents early fill --
    #[test]
    fn test_latency_prevents_fill() {
        let mut exchange = SimExchange::new(
            FeeConfig::default(),
            LatencyConfig {
                order_latency_ns: 1_000_000, // 1ms
                cancel_latency_ns: 500_000,
            },
        );

        // Submit at time 0
        exchange.set_time(0);
        exchange.submit_order(Side::Buy, Price::from(50000.0), Quantity::from(0.1));

        // Book update at time 500_000 (0.5ms) -- order not yet active
        exchange.set_time(500_000);
        let book = make_book_update(49998.0, 1.0, 49999.0, 1.0);
        exchange.on_book_update(&book);

        let fills = exchange.drain_fills();
        assert!(
            fills.is_empty(),
            "order should not fill before latency window"
        );

        // Book update at time 1_000_000 (1ms) -- order now active
        exchange.set_time(1_000_000);
        exchange.on_book_update(&book);

        let fills = exchange.drain_fills();
        assert_eq!(fills.len(), 1, "order should fill after latency window");
    }

    // -- 7. Partial fill --
    #[test]
    fn test_partial_fill() {
        let mut exchange = zero_latency_exchange();
        exchange.set_time(1_000_000);

        // Submit a bid for 1.0 BTC
        let _oid = exchange.submit_order(Side::Buy, Price::from(50000.0), Quantity::from(1.0));

        // Book update with ask that only has 0.3 quantity
        let book = make_book_update(49998.0, 1.0, 49999.0, 0.3);
        exchange.on_book_update(&book);

        let fills = exchange.drain_fills();
        assert_eq!(fills.len(), 1);
        assert!((fills[0].quantity.to_f64() - 0.3).abs() < 1e-8);

        // Order should still be resting (partially filled)
        assert_eq!(exchange.open_order_count(), 1);

        // Feed another book update with remaining quantity
        let book2 = make_book_update(49998.0, 1.0, 49999.0, 0.8);
        exchange.on_book_update(&book2);

        let fills2 = exchange.drain_fills();
        assert_eq!(fills2.len(), 1);
        assert!((fills2[0].quantity.to_f64() - 0.7).abs() < 1e-8);

        // Order should be fully filled now
        assert_eq!(exchange.open_order_count(), 0);
    }

    // -- 8. Multiple orders at same price level (FIFO) --
    #[test]
    fn test_fifo_same_price() {
        let mut exchange = zero_latency_exchange();
        exchange.set_time(1_000_000);

        let oid1 = exchange.submit_order(Side::Buy, Price::from(50000.0), Quantity::from(0.5));
        let _oid2 = exchange.submit_order(Side::Buy, Price::from(50000.0), Quantity::from(0.5));

        // Book update with ask crossing, but only 0.5 qty available
        let book = make_book_update(49998.0, 1.0, 49999.0, 0.5);
        exchange.on_book_update(&book);

        let fills = exchange.drain_fills();
        assert_eq!(fills.len(), 1);
        // First order should be filled first (FIFO)
        assert_eq!(fills[0].order_id, oid1);
        assert!((fills[0].quantity.to_f64() - 0.5).abs() < 1e-8);

        // Second order still resting
        assert_eq!(exchange.open_order_count(), 1);
    }

    // -- 9. Cancel all clears everything --
    #[test]
    fn test_cancel_all() {
        let mut exchange = zero_latency_exchange();
        exchange.set_time(1_000_000);

        exchange.submit_order(Side::Buy, Price::from(50000.0), Quantity::from(0.1));
        exchange.submit_order(Side::Buy, Price::from(49999.0), Quantity::from(0.2));
        exchange.submit_order(Side::Sell, Price::from(50001.0), Quantity::from(0.1));
        exchange.submit_order(Side::Sell, Price::from(50002.0), Quantity::from(0.3));

        assert_eq!(exchange.open_order_count(), 4);

        exchange.cancel_all();
        assert_eq!(exchange.open_order_count(), 0);
    }

    // -- 10. Ask order fill --
    #[test]
    fn test_submit_and_fill_ask() {
        let mut exchange = zero_latency_exchange();
        exchange.set_time(1_000_000);

        // Submit ask at 50000
        let oid = exchange.submit_order(Side::Sell, Price::from(50000.0), Quantity::from(0.1));

        // Book update with bid at 50001 (crosses our ask)
        let book = make_book_update(50001.0, 1.0, 50002.0, 1.0);
        exchange.on_book_update(&book);

        let fills = exchange.drain_fills();
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].order_id, oid);
        assert_eq!(fills[0].side, Side::Sell);
        assert!(fills[0].is_maker);
    }

    // -- 11. Multiple price levels fill best first --
    #[test]
    fn test_multiple_price_levels() {
        let mut exchange = zero_latency_exchange();
        exchange.set_time(1_000_000);

        // Submit bids at different prices
        let oid1 = exchange.submit_order(Side::Buy, Price::from(50001.0), Quantity::from(0.5));
        let _oid2 = exchange.submit_order(Side::Buy, Price::from(50000.0), Quantity::from(0.5));

        // Book update with ask at 49999, qty 0.5 (only fills best bid)
        let book = make_book_update(49998.0, 1.0, 49999.0, 0.5);
        exchange.on_book_update(&book);

        let fills = exchange.drain_fills();
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].order_id, oid1); // Higher bid fills first
    }

    // -- 12. Drain fills clears buffer --
    #[test]
    fn test_drain_fills_clears() {
        let mut exchange = zero_latency_exchange();
        exchange.set_time(1_000_000);

        exchange.submit_order(Side::Buy, Price::from(50000.0), Quantity::from(0.1));
        let book = make_book_update(49998.0, 1.0, 49999.0, 1.0);
        exchange.on_book_update(&book);

        let fills1 = exchange.drain_fills();
        assert_eq!(fills1.len(), 1);

        let fills2 = exchange.drain_fills();
        assert!(fills2.is_empty());
    }

    // -- 13. Open order count tracks correctly --
    #[test]
    fn test_open_order_count() {
        let mut exchange = zero_latency_exchange();
        exchange.set_time(1_000_000);

        assert_eq!(exchange.open_order_count(), 0);

        exchange.submit_order(Side::Buy, Price::from(50000.0), Quantity::from(0.1));
        assert_eq!(exchange.open_order_count(), 1);

        exchange.submit_order(Side::Sell, Price::from(50002.0), Quantity::from(0.2));
        assert_eq!(exchange.open_order_count(), 2);

        exchange.submit_order(Side::Buy, Price::from(49999.0), Quantity::from(0.3));
        assert_eq!(exchange.open_order_count(), 3);
    }

    // -- 14. Total fees accumulate --
    #[test]
    fn test_total_fees_accumulate() {
        let mut exchange = SimExchange::new(
            FeeConfig {
                maker_fee: -0.0001,
                taker_fee: 0.0004,
            },
            LatencyConfig {
                order_latency_ns: 0,
                cancel_latency_ns: 0,
            },
        );
        exchange.set_time(1_000_000);

        // Submit two bids
        exchange.submit_order(Side::Buy, Price::from(50000.0), Quantity::from(1.0));
        exchange.submit_order(Side::Buy, Price::from(50000.0), Quantity::from(1.0));

        // Both should fill
        let book = make_book_update(49998.0, 10.0, 49999.0, 10.0);
        exchange.on_book_update(&book);

        let fills = exchange.drain_fills();
        assert_eq!(fills.len(), 2);

        // Each fee = 50000 * 1.0 * (-0.0001) = -5.0
        let expected_total = 2.0 * 50000.0 * 1.0 * (-0.0001);
        assert!(
            (exchange.total_fees() - expected_total).abs() < 0.01,
            "total_fees={}, expected={}",
            exchange.total_fees(),
            expected_total
        );
    }

    // -- 15. New exchange starts clean --
    #[test]
    fn test_new_exchange_clean() {
        let exchange = default_exchange();
        assert_eq!(exchange.open_order_count(), 0);
        assert!((exchange.total_fees()).abs() < 1e-12);
    }

    // -- 16. Fill at exact crossing price --
    #[test]
    fn test_fill_at_exact_crossing() {
        let mut exchange = zero_latency_exchange();
        exchange.set_time(1_000_000);

        // Bid at exactly 50000
        exchange.submit_order(Side::Buy, Price::from(50000.0), Quantity::from(0.1));

        // Ask at exactly 50000 (crosses)
        let book = make_book_update(49999.0, 1.0, 50000.0, 1.0);
        exchange.on_book_update(&book);

        let fills = exchange.drain_fills();
        assert_eq!(fills.len(), 1);
    }

    // -- 17. Submit ask order and check it rests --
    #[test]
    fn test_ask_order_rests() {
        let mut exchange = zero_latency_exchange();
        exchange.set_time(1_000_000);

        // Submit ask above market
        exchange.submit_order(Side::Sell, Price::from(51000.0), Quantity::from(0.1));

        // Book update where bid is below our ask
        let book = make_book_update(50000.0, 1.0, 50001.0, 1.0);
        exchange.on_book_update(&book);

        let fills = exchange.drain_fills();
        assert!(fills.is_empty());
        assert_eq!(exchange.open_order_count(), 1);
    }

    // -- 18. Order ID monotonically increases --
    #[test]
    fn test_order_id_monotonic() {
        let mut exchange = zero_latency_exchange();
        exchange.set_time(1_000_000);

        let oid1 = exchange.submit_order(Side::Buy, Price::from(50000.0), Quantity::from(0.1));
        let oid2 = exchange.submit_order(Side::Buy, Price::from(50001.0), Quantity::from(0.1));
        let oid3 = exchange.submit_order(Side::Sell, Price::from(50002.0), Quantity::from(0.1));

        assert!(oid1.0 < oid2.0);
        assert!(oid2.0 < oid3.0);
    }

    // -- 19. Cancel after fill has no effect --
    #[test]
    fn test_cancel_after_fill() {
        let mut exchange = zero_latency_exchange();
        exchange.set_time(1_000_000);

        let oid = exchange.submit_order(Side::Buy, Price::from(50000.0), Quantity::from(0.1));

        // Fill it
        let book = make_book_update(49998.0, 1.0, 49999.0, 1.0);
        exchange.on_book_update(&book);

        let fills = exchange.drain_fills();
        assert_eq!(fills.len(), 1);

        // Try to cancel — should return false since it's already filled/removed
        let cancelled = exchange.cancel_order(oid);
        assert!(!cancelled);
    }

    // -- 20. Book update with empty sides --
    #[test]
    fn test_book_update_empty_sides() {
        let mut exchange = zero_latency_exchange();
        exchange.set_time(1_000_000);

        exchange.submit_order(Side::Buy, Price::from(50000.0), Quantity::from(0.1));

        // Book update with no asks
        let book = BookUpdate {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            timestamp: Timestamp(1_000_000_000),
            bids: vec![(Price::from(49998.0), Quantity::from(1.0))],
            asks: vec![],
            is_snapshot: true,
        };
        exchange.on_book_update(&book);

        let fills = exchange.drain_fills();
        assert!(fills.is_empty());
    }

    // -- 21. Positive taker fee config --
    #[test]
    fn test_taker_fee_config() {
        let fee_config = FeeConfig {
            maker_fee: 0.0002,
            taker_fee: 0.0006,
        };
        let mut exchange = SimExchange::new(
            fee_config,
            LatencyConfig {
                order_latency_ns: 0,
                cancel_latency_ns: 0,
            },
        );
        exchange.set_time(1_000_000);

        exchange.submit_order(Side::Buy, Price::from(50000.0), Quantity::from(1.0));
        let book = make_book_update(49998.0, 10.0, 49999.0, 10.0);
        exchange.on_book_update(&book);

        let fills = exchange.drain_fills();
        assert_eq!(fills.len(), 1);

        // Maker fill: 50000 * 1.0 * 0.0002 = 10.0
        let expected_fee = 50000.0 * 1.0 * 0.0002;
        assert!(
            (exchange.total_fees() - expected_fee).abs() < 0.01,
            "total_fees={}, expected={}",
            exchange.total_fees(),
            expected_fee
        );
    }
}

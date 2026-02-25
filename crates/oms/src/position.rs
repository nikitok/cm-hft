//! Position tracking across symbols and exchanges.
//!
//! Maintains net position, volume-weighted average entry price, and realized PnL
//! for each (exchange, symbol) pair. Also tracks pending orders for worst-case
//! position calculations.

use cm_core::types::*;

/// A position in a single symbol on a single exchange.
#[derive(Debug, Clone)]
pub struct Position {
    /// Exchange this position is on.
    pub exchange: Exchange,
    /// Trading pair.
    pub symbol: Symbol,
    /// Net quantity (positive = long, negative = short).
    pub net_quantity: Quantity,
    /// Volume-weighted average entry price.
    pub avg_entry_price: Price,
    /// Realized PnL from closed trades.
    pub realized_pnl: Price,
    /// Number of fills contributing to this position.
    pub fill_count: u64,
}

/// Manages positions across all symbols and exchanges.
///
/// Thread-safe: uses `DashMap` for concurrent access.
pub struct PositionTracker {
    positions: dashmap::DashMap<(Exchange, Symbol), Position>,
    pending_orders: dashmap::DashMap<OrderId, (Side, Quantity)>,
}

impl PositionTracker {
    /// Create a new, empty position tracker.
    pub fn new() -> Self {
        Self {
            positions: dashmap::DashMap::new(),
            pending_orders: dashmap::DashMap::new(),
        }
    }

    /// Update position on a fill event.
    ///
    /// Handles long/short accumulation, position reduction, position flipping,
    /// and realized PnL calculation.
    ///
    /// PnL math uses `to_f64()` for Price * Quantity multiplication, which is
    /// acceptable here since PnL is not on the order hot path.
    pub fn on_fill(
        &self,
        exchange: Exchange,
        symbol: Symbol,
        side: Side,
        fill_price: Price,
        fill_qty: Quantity,
    ) {
        let key = (exchange, symbol.clone());
        let mut entry = self.positions.entry(key).or_insert_with(|| Position {
            exchange,
            symbol: symbol.clone(),
            net_quantity: Quantity::zero(fill_qty.scale()),
            avg_entry_price: Price::zero(fill_price.scale()),
            realized_pnl: Price::zero(fill_price.scale()),
            fill_count: 0,
        });

        let pos = entry.value_mut();
        pos.fill_count += 1;

        // Signed fill quantity: positive for buys, negative for sells
        let signed_qty = match side {
            Side::Buy => fill_qty,
            Side::Sell => -fill_qty,
        };

        let old_net = pos.net_quantity;
        let new_net = old_net + signed_qty;

        // Determine if this fill is increasing, reducing, or flipping the position
        let old_net_f64 = old_net.to_f64();
        let signed_qty_f64 = signed_qty.to_f64();
        let new_net_f64 = new_net.to_f64();
        let fill_price_f64 = fill_price.to_f64();

        if old_net_f64 == 0.0 {
            // Opening a new position
            pos.avg_entry_price = fill_price;
        } else if (old_net_f64 > 0.0 && signed_qty_f64 > 0.0)
            || (old_net_f64 < 0.0 && signed_qty_f64 < 0.0)
        {
            // Adding to existing position: update VWAP
            let old_cost = pos.avg_entry_price.to_f64() * old_net_f64.abs();
            let new_cost = fill_price_f64 * signed_qty_f64.abs();
            let total_qty = old_net_f64.abs() + signed_qty_f64.abs();
            if total_qty != 0.0 {
                let new_avg = (old_cost + new_cost) / total_qty;
                pos.avg_entry_price = Price::from(new_avg);
            }
        } else {
            // Reducing or flipping position
            let reduce_qty = signed_qty_f64.abs().min(old_net_f64.abs());
            let avg_entry_f64 = pos.avg_entry_price.to_f64();

            // Realized PnL: (fill_price - avg_entry) * reduce_qty for longs,
            // (avg_entry - fill_price) * reduce_qty for shorts
            let pnl = if old_net_f64 > 0.0 {
                // Was long, selling to reduce
                (fill_price_f64 - avg_entry_f64) * reduce_qty
            } else {
                // Was short, buying to reduce
                (avg_entry_f64 - fill_price_f64) * reduce_qty
            };
            pos.realized_pnl = pos.realized_pnl + Price::from(pnl);

            // Check if position flipped
            if (old_net_f64 > 0.0 && new_net_f64 < 0.0)
                || (old_net_f64 < 0.0 && new_net_f64 > 0.0)
            {
                // Position flipped: new entry price is the fill price
                pos.avg_entry_price = fill_price;
            } else if new_net_f64 == 0.0 {
                // Position fully closed
                pos.avg_entry_price = Price::zero(fill_price.scale());
            }
            // If just partially reduced (not flipped), avg_entry_price stays the same
        }

        pos.net_quantity = new_net;
    }

    /// Get a clone of the position for the given exchange and symbol.
    pub fn get_position(&self, exchange: &Exchange, symbol: &Symbol) -> Option<Position> {
        self.positions
            .get(&(*exchange, symbol.clone()))
            .map(|p| p.clone())
    }

    /// Get just the net quantity for the given exchange and symbol.
    pub fn net_position(&self, exchange: &Exchange, symbol: &Symbol) -> Quantity {
        self.positions
            .get(&(*exchange, symbol.clone()))
            .map(|p| p.net_quantity)
            .unwrap_or(Quantity::zero(8))
    }

    /// Calculate unrealized PnL (mark-to-market) for the given position.
    ///
    /// Uses `to_f64()` for Price * Quantity math (acceptable for PnL, not hot path).
    pub fn unrealized_pnl(
        &self,
        exchange: &Exchange,
        symbol: &Symbol,
        mark_price: Price,
    ) -> Price {
        match self.positions.get(&(*exchange, symbol.clone())) {
            Some(pos) => {
                let net_f64 = pos.net_quantity.to_f64();
                if net_f64 == 0.0 {
                    return Price::zero(mark_price.scale());
                }
                let avg_entry_f64 = pos.avg_entry_price.to_f64();
                let mark_f64 = mark_price.to_f64();
                // Long: (mark - entry) * qty, Short: (entry - mark) * |qty|
                let pnl = (mark_f64 - avg_entry_f64) * net_f64;
                Price::from(pnl)
            }
            None => Price::zero(mark_price.scale()),
        }
    }

    /// Calculate worst-case position: net position + all pending order quantities.
    ///
    /// Returns the maximum absolute exposure considering all pending orders.
    pub fn worst_case_position(
        &self,
        exchange: &Exchange,
        symbol: &Symbol,
    ) -> Quantity {
        let net = self.net_position(exchange, symbol);
        let mut buy_pending = Quantity::zero(net.scale());
        let mut sell_pending = Quantity::zero(net.scale());

        for entry in self.pending_orders.iter() {
            let (side, qty) = entry.value();
            match side {
                Side::Buy => buy_pending = buy_pending + *qty,
                Side::Sell => sell_pending = sell_pending + *qty,
            }
        }

        // Worst case: net + all buys (if long) or net - all sells (if short)
        // We return the larger absolute exposure
        let long_worst = net + buy_pending;
        let short_worst = net - sell_pending;

        if long_worst.abs() >= short_worst.abs() {
            long_worst
        } else {
            short_worst
        }
    }

    /// Register a pending order for worst-case position tracking.
    pub fn add_pending(&self, order_id: OrderId, side: Side, quantity: Quantity) {
        self.pending_orders.insert(order_id, (side, quantity));
    }

    /// Remove a pending order (after fill, cancel, or reject).
    pub fn remove_pending(&self, order_id: &OrderId) {
        self.pending_orders.remove(order_id);
    }

    /// Get all positions across all exchanges and symbols.
    pub fn all_positions(&self) -> Vec<Position> {
        self.positions.iter().map(|e| e.value().clone()).collect()
    }
}

impl Default for PositionTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn btc() -> Symbol {
        Symbol::new("BTCUSDT")
    }

    fn binance() -> Exchange {
        Exchange::Binance
    }

    fn price(val: f64) -> Price {
        Price::from(val)
    }

    fn qty(val: f64) -> Quantity {
        Quantity::from(val)
    }

    fn assert_price_approx(actual: Price, expected: f64, tolerance: f64) {
        let diff = (actual.to_f64() - expected).abs();
        assert!(
            diff < tolerance,
            "expected ~{}, got {} (diff={})",
            expected,
            actual.to_f64(),
            diff
        );
    }

    fn assert_qty_approx(actual: Quantity, expected: f64, tolerance: f64) {
        let diff = (actual.to_f64() - expected).abs();
        assert!(
            diff < tolerance,
            "expected ~{}, got {} (diff={})",
            expected,
            actual.to_f64(),
            diff
        );
    }

    #[test]
    fn test_single_buy_fill_long_position() {
        let tracker = PositionTracker::new();
        tracker.on_fill(binance(), btc(), Side::Buy, price(50000.0), qty(1.0));

        let pos = tracker.get_position(&binance(), &btc()).unwrap();
        assert_qty_approx(pos.net_quantity, 1.0, 1e-6);
        assert_price_approx(pos.avg_entry_price, 50000.0, 0.01);
        assert_eq!(pos.fill_count, 1);
    }

    #[test]
    fn test_single_sell_fill_short_position() {
        let tracker = PositionTracker::new();
        tracker.on_fill(binance(), btc(), Side::Sell, price(50000.0), qty(1.0));

        let pos = tracker.get_position(&binance(), &btc()).unwrap();
        assert_qty_approx(pos.net_quantity, -1.0, 1e-6);
        assert_price_approx(pos.avg_entry_price, 50000.0, 0.01);
    }

    #[test]
    fn test_buy_then_sell_close_position_realized_pnl() {
        let tracker = PositionTracker::new();
        // Buy 1 BTC at 50000
        tracker.on_fill(binance(), btc(), Side::Buy, price(50000.0), qty(1.0));
        // Sell 1 BTC at 51000 -> profit of 1000
        tracker.on_fill(binance(), btc(), Side::Sell, price(51000.0), qty(1.0));

        let pos = tracker.get_position(&binance(), &btc()).unwrap();
        assert_qty_approx(pos.net_quantity, 0.0, 1e-6);
        assert_price_approx(pos.realized_pnl, 1000.0, 0.01);
    }

    #[test]
    fn test_sell_then_buy_close_short_realized_pnl() {
        let tracker = PositionTracker::new();
        // Short 1 BTC at 50000
        tracker.on_fill(binance(), btc(), Side::Sell, price(50000.0), qty(1.0));
        // Buy back at 49000 -> profit of 1000
        tracker.on_fill(binance(), btc(), Side::Buy, price(49000.0), qty(1.0));

        let pos = tracker.get_position(&binance(), &btc()).unwrap();
        assert_qty_approx(pos.net_quantity, 0.0, 1e-6);
        assert_price_approx(pos.realized_pnl, 1000.0, 0.01);
    }

    #[test]
    fn test_multiple_fills_accumulating() {
        let tracker = PositionTracker::new();
        // Buy 0.5 BTC at 50000
        tracker.on_fill(binance(), btc(), Side::Buy, price(50000.0), qty(0.5));
        // Buy 0.5 BTC at 52000
        tracker.on_fill(binance(), btc(), Side::Buy, price(52000.0), qty(0.5));

        let pos = tracker.get_position(&binance(), &btc()).unwrap();
        assert_qty_approx(pos.net_quantity, 1.0, 1e-6);
        // VWAP = (50000 * 0.5 + 52000 * 0.5) / 1.0 = 51000
        assert_price_approx(pos.avg_entry_price, 51000.0, 0.01);
        assert_eq!(pos.fill_count, 2);
    }

    #[test]
    fn test_worst_case_position_with_pending() {
        let tracker = PositionTracker::new();
        tracker.on_fill(binance(), btc(), Side::Buy, price(50000.0), qty(1.0));

        // Add pending buy order
        tracker.add_pending(OrderId(1), Side::Buy, qty(0.5));

        let worst = tracker.worst_case_position(&binance(), &btc());
        // Net = 1.0, pending buy = 0.5 -> worst long = 1.5
        assert_qty_approx(worst, 1.5, 1e-6);
    }

    #[test]
    fn test_remove_pending() {
        let tracker = PositionTracker::new();
        tracker.on_fill(binance(), btc(), Side::Buy, price(50000.0), qty(1.0));
        tracker.add_pending(OrderId(1), Side::Buy, qty(0.5));

        tracker.remove_pending(&OrderId(1));
        let worst = tracker.worst_case_position(&binance(), &btc());
        assert_qty_approx(worst, 1.0, 1e-6);
    }

    #[test]
    fn test_unrealized_pnl_long() {
        let tracker = PositionTracker::new();
        tracker.on_fill(binance(), btc(), Side::Buy, price(50000.0), qty(1.0));

        // Mark price up -> unrealized profit
        let upnl = tracker.unrealized_pnl(&binance(), &btc(), price(51000.0));
        assert_price_approx(upnl, 1000.0, 0.01);

        // Mark price down -> unrealized loss
        let upnl = tracker.unrealized_pnl(&binance(), &btc(), price(49000.0));
        assert_price_approx(upnl, -1000.0, 0.01);
    }

    #[test]
    fn test_unrealized_pnl_short() {
        let tracker = PositionTracker::new();
        tracker.on_fill(binance(), btc(), Side::Sell, price(50000.0), qty(1.0));

        // Mark price down -> unrealized profit for short
        let upnl = tracker.unrealized_pnl(&binance(), &btc(), price(49000.0));
        assert_price_approx(upnl, 1000.0, 0.01);

        // Mark price up -> unrealized loss for short
        let upnl = tracker.unrealized_pnl(&binance(), &btc(), price(51000.0));
        assert_price_approx(upnl, -1000.0, 0.01);
    }

    #[test]
    fn test_unrealized_pnl_no_position() {
        let tracker = PositionTracker::new();
        let upnl = tracker.unrealized_pnl(&binance(), &btc(), price(50000.0));
        assert!(upnl.is_zero() || upnl.to_f64().abs() < 1e-10);
    }

    #[test]
    fn test_position_flip_long_to_short() {
        let tracker = PositionTracker::new();
        // Buy 1 BTC at 50000
        tracker.on_fill(binance(), btc(), Side::Buy, price(50000.0), qty(1.0));
        // Sell 2 BTC at 51000 -> close 1 long (profit 1000) + open 1 short
        tracker.on_fill(binance(), btc(), Side::Sell, price(51000.0), qty(2.0));

        let pos = tracker.get_position(&binance(), &btc()).unwrap();
        assert_qty_approx(pos.net_quantity, -1.0, 1e-6);
        // Realized PnL from closing the 1 BTC long: (51000 - 50000) * 1 = 1000
        assert_price_approx(pos.realized_pnl, 1000.0, 0.01);
        // New entry for the short is 51000
        assert_price_approx(pos.avg_entry_price, 51000.0, 0.01);
    }

    #[test]
    fn test_position_flip_short_to_long() {
        let tracker = PositionTracker::new();
        // Short 1 BTC at 50000
        tracker.on_fill(binance(), btc(), Side::Sell, price(50000.0), qty(1.0));
        // Buy 2 BTC at 49000 -> close 1 short (profit 1000) + open 1 long
        tracker.on_fill(binance(), btc(), Side::Buy, price(49000.0), qty(2.0));

        let pos = tracker.get_position(&binance(), &btc()).unwrap();
        assert_qty_approx(pos.net_quantity, 1.0, 1e-6);
        assert_price_approx(pos.realized_pnl, 1000.0, 0.01);
        assert_price_approx(pos.avg_entry_price, 49000.0, 0.01);
    }

    #[test]
    fn test_net_position_no_position() {
        let tracker = PositionTracker::new();
        let net = tracker.net_position(&binance(), &btc());
        assert!(net.is_zero());
    }

    #[test]
    fn test_all_positions() {
        let tracker = PositionTracker::new();
        tracker.on_fill(binance(), btc(), Side::Buy, price(50000.0), qty(1.0));
        tracker.on_fill(
            Exchange::Bybit,
            Symbol::new("ETHUSDT"),
            Side::Sell,
            price(3000.0),
            qty(10.0),
        );

        let positions = tracker.all_positions();
        assert_eq!(positions.len(), 2);
    }

    #[test]
    fn test_partial_close_preserves_avg_entry() {
        let tracker = PositionTracker::new();
        // Buy 2 BTC at 50000
        tracker.on_fill(binance(), btc(), Side::Buy, price(50000.0), qty(2.0));
        // Sell 1 BTC at 51000 -> partial close, avg entry stays at 50000
        tracker.on_fill(binance(), btc(), Side::Sell, price(51000.0), qty(1.0));

        let pos = tracker.get_position(&binance(), &btc()).unwrap();
        assert_qty_approx(pos.net_quantity, 1.0, 1e-6);
        assert_price_approx(pos.avg_entry_price, 50000.0, 0.01);
        assert_price_approx(pos.realized_pnl, 1000.0, 0.01);
    }

    #[test]
    fn test_worst_case_with_sell_pending() {
        let tracker = PositionTracker::new();
        tracker.on_fill(binance(), btc(), Side::Buy, price(50000.0), qty(1.0));
        tracker.add_pending(OrderId(1), Side::Sell, qty(2.0));

        let worst = tracker.worst_case_position(&binance(), &btc());
        // Net = 1.0, sell pending = 2.0 -> worst short = 1.0 - 2.0 = -1.0
        // worst long = 1.0, |long| == |short|, so either is valid; check absolute value
        assert_qty_approx(worst.abs(), 1.0, 1e-6);
    }

    #[test]
    fn test_worst_case_sell_pending_larger() {
        let tracker = PositionTracker::new();
        tracker.on_fill(binance(), btc(), Side::Buy, price(50000.0), qty(1.0));
        tracker.add_pending(OrderId(1), Side::Sell, qty(3.0));

        let worst = tracker.worst_case_position(&binance(), &btc());
        // Net = 1.0, sell pending = 3.0 -> worst short = 1.0 - 3.0 = -2.0
        // worst long = 1.0, |-2.0| > |1.0|, so returns -2.0
        assert_qty_approx(worst, -2.0, 1e-6);
    }

    #[test]
    fn test_realized_pnl_losing_trade() {
        let tracker = PositionTracker::new();
        // Buy 1 BTC at 50000
        tracker.on_fill(binance(), btc(), Side::Buy, price(50000.0), qty(1.0));
        // Sell 1 BTC at 49000 -> loss of 1000
        tracker.on_fill(binance(), btc(), Side::Sell, price(49000.0), qty(1.0));

        let pos = tracker.get_position(&binance(), &btc()).unwrap();
        assert_price_approx(pos.realized_pnl, -1000.0, 0.01);
    }
}

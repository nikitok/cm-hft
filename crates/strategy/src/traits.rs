//! Core strategy trait and supporting types.
//!
//! Strategies are compiled Rust code, not interpreted. All callbacks are
//! synchronous -- no async, no locks, no I/O on the hot path.

use cm_core::types::*;
use cm_market_data::orderbook::OrderBook;

use super::context::TradingContext;

/// Core strategy trait. Strategies are compiled Rust code, not interpreted.
/// All callbacks are synchronous -- no async, no locks, no I/O on the hot path.
pub trait Strategy: Send + 'static {
    /// Called on every L2 book update. This is the primary decision point.
    fn on_book_update(&mut self, ctx: &mut TradingContext, book: &OrderBook);

    /// Called on every trade event.
    fn on_trade(&mut self, ctx: &mut TradingContext, trade: &Trade);

    /// Called when one of the strategy's orders is filled.
    fn on_fill(&mut self, ctx: &mut TradingContext, fill: &Fill);

    /// Called on timer tick (configurable interval).
    fn on_timer(&mut self, ctx: &mut TradingContext, timestamp: Timestamp);

    /// Called when strategy parameters are updated at runtime.
    fn on_params_update(&mut self, params: &StrategyParams);

    /// Strategy name for logging and metrics.
    fn name(&self) -> &str;
}

/// A fill event delivered to the strategy.
#[derive(Debug, Clone)]
pub struct Fill {
    /// Internal order identifier.
    pub order_id: OrderId,
    /// Exchange the fill occurred on.
    pub exchange: Exchange,
    /// Trading pair.
    pub symbol: Symbol,
    /// Order side.
    pub side: Side,
    /// Fill price.
    pub price: Price,
    /// Fill quantity.
    pub quantity: Quantity,
    /// Nanosecond timestamp of the fill.
    pub timestamp: Timestamp,
    /// Whether this fill was as a maker (resting order).
    pub is_maker: bool,
}

/// Dynamic strategy parameters (reloadable without restart).
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct StrategyParams {
    /// Raw JSON parameter map.
    pub params: serde_json::Value,
}

impl StrategyParams {
    /// Get a parameter as `f64`, returning `None` if absent or wrong type.
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.params.get(key).and_then(|v| v.as_f64())
    }

    /// Get a parameter as `i64`, returning `None` if absent or wrong type.
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.params.get(key).and_then(|v| v.as_i64())
    }

    /// Get a parameter as `bool`, returning `None` if absent or wrong type.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.params.get(key).and_then(|v| v.as_bool())
    }
}

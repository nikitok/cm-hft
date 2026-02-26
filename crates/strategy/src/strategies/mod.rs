//! Built-in strategy implementations.

pub mod adaptive_mm;
pub mod market_making;
pub mod signals;

pub use adaptive_mm::AdaptiveMarketMaker;
pub use market_making::MarketMakingStrategy;

//! Core types for the CM.HFT trading platform.
//!
//! All types in this module are designed for the hot path where possible:
//! fixed-point arithmetic avoids floating-point, timestamps use raw nanoseconds,
//! and allocations are minimized.

pub mod market_data;
pub mod order;
pub mod price;
pub mod quantity;
pub mod timestamp;

// Re-export primary types for convenient access via `cm_core::types::*`.
pub use market_data::{BookLevel, BookUpdate, NormalizedTick, Trade};
pub use order::{Exchange, ExchangeOrderId, OrderId, OrderType, Side, Symbol};
pub use price::Price;
pub use quantity::Quantity;
pub use timestamp::Timestamp;

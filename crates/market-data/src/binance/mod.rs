//! Binance WebSocket client and wire types.
//!
//! This module provides a WebSocket client for Binance's spot/futures market
//! data API including order book depth updates and public trade feeds.

pub mod client;
pub mod types;

pub use client::{BinanceMessage, BinanceWsClient, BinanceWsStream};
pub use types::{BinanceDepthSnapshot, BinanceDepthUpdate, BinanceTrade};

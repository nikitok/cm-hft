//! Bybit exchange WebSocket client and wire types.
//!
//! This module provides a WebSocket client for Bybit's v5 public linear
//! perpetual API, including orderbook depth and public trade feeds.

pub mod client;
pub mod types;

pub use client::{BybitConfig, BybitWsClient};
pub use types::{BybitOrderbook, BybitSubscribeRequest, BybitTrade, BybitWsResponse};

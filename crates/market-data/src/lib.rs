//! # cm-market-data
//!
//! WebSocket clients and order book reconstruction for Binance and Bybit
//! exchange feeds. Handles connection management, message parsing, and
//! normalization into the internal [`cm_core::types`] format.

pub mod binance;
pub mod bybit;
pub mod orderbook;
pub mod ws;

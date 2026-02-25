//! # cm-execution
//!
//! Exchange gateway implementations for Binance and Bybit. Handles REST API
//! communication, HMAC request signing, order placement/cancellation, and
//! per-endpoint rate limiting.

pub mod binance_rest;
pub mod bybit_rest;
pub mod gateway;
pub mod rate_limiter;
pub mod signing;

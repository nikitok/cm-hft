//! # cm-strategy
//!
//! Strategy trait definitions and implementations. Strategies are compiled Rust
//! code that react to market data events and produce order actions through a
//! [`TradingContext`]. The hot path is synchronous, lock-free, and
//! allocation-free.

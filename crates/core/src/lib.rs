//! # cm-core
//!
//! Shared types, traits, and utilities for the CM.HFT trading platform.
//!
//! This crate provides the foundational building blocks used across all other
//! crates in the workspace: fixed-point price/quantity types, order definitions,
//! nanosecond timestamps, normalized market data structures, and the logging
//! framework.

pub mod config;
pub mod logging;
pub mod types;

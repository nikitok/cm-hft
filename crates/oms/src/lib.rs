//! # cm-oms
//!
//! Order Management System. Tracks every order through its full lifecycle
//! using a deterministic state machine, manages position tracking, fill
//! deduplication, and periodic reconciliation against exchange state.

pub mod dedup;
pub mod journal;
pub mod order;
pub mod position;

pub use dedup::FillDeduplicator;
pub use journal::OrderJournal;
pub use order::{Order, OrderError, OrderEvent, OrderEventType, OrderManager, OrderStatus};
pub use position::{Position, PositionTracker};

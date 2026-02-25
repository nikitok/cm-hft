//! # cm-recorder
//!
//! Market data recording to QuestDB via the InfluxDB Line Protocol. Buffers
//! normalized market events in a ring buffer and flushes in batches for
//! high-throughput time-series ingestion.

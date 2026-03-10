# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A cryptocurrency HFT (high-frequency trading) platform. Rust for the hot path (market data, strategy, execution), Python for backtesting/analysis, Go for monitoring infrastructure.

## Build & Test Commands

### Rust

```bash
# Build
cargo build --workspace
cargo build --workspace --release

# Lint (CI enforces both)
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings

# Test
cargo test --workspace --all-targets
cargo test --workspace --doc

# Single crate
cargo test --lib -p cm-core
cargo test --lib -p cm-strategy
```

### Python

```bash
uv sync --all-extras
uv run ruff check backtest/
uv run ruff format --check backtest/
uv run pytest backtest/tests/ -v
```

### PyO3 Bridge (build native module for Python)

```bash
maturin develop --manifest-path crates/pybridge/Cargo.toml --features extension-module
```

### Local Infrastructure

```bash
docker compose up -d   # QuestDB (9000/9009), Redis (6379), MinIO (9100/9101)
```

## Architecture

### Crate Dependency Graph

```
cm-trading (binary: cm-trading, cm-record)
├── cm-strategy      Strategy trait + implementations (market making, adaptive MM)
├── cm-oms           Order state machine, position tracking, fill dedup
├── cm-risk          Pre-trade risk pipeline, circuit breaker, kill switch
├── cm-execution     Exchange REST gateways (Binance, Bybit), rate limiting, HMAC signing
├── cm-market-data   WebSocket clients (Binance, Bybit), L2 order book reconstruction
├── cm-recorder      QuestDB ingestion via InfluxDB Line Protocol
└── cm-core          Shared types (Price, Quantity, Symbol, Side, OrderType), config, logging

cm-pybridge          PyO3 bindings exposing Rust strategies to Python backtest engine
```

### Hot Path Flow

Exchange WS → Market Data Handler → Order Book Reconstruction → Strategy callback (`on_book_update`) → Order Actions buffer → Risk Pipeline → Order Manager → Execution Gateway → Exchange REST API → Fill Event → Fill Dedup → Position Tracker → Strategy callback (`on_fill`)

### Threading Model

- **Tokio async tasks:** WebSocket feeds, HTTP server (port 8080), fill processor, timers
- **Dedicated OS thread:** Strategy evaluation (avoids async jitter)
- **crossbeam channels:** `md_tx/md_rx` (bounded 4096), `action_tx/action_rx` (bounded 1024), `fill_tx/fill_rx` (unbounded)

### Strategy Hot Path Constraints

The `Strategy` trait callbacks must be synchronous, lock-free, and allocation-free. No async, no heap allocations, no locks on the critical path. Fixed-point arithmetic (`Price`, `Quantity` with mantissa + scale) avoids floating-point.

### Configuration

Layered: compiled defaults → TOML file (`--config`) → env vars (prefix `CM_HFT_`, nested with `__`). API keys via `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `BYBIT_API_KEY`, `BYBIT_API_SECRET`. See `config/default.toml` for all options.

### HTTP API (port 8080)

`GET /health`, `GET /status`, `POST /kill` (emergency halt), `POST /reset` (requires `CM_RESET_TOKEN`), `GET /positions`, `GET /orders`, `GET /metrics` (Prometheus format).

### Backtesting (Python)

Entry point: `backtest/engine/orchestrator.py`. Loads historical Parquet data, replays through Rust strategies via PyO3 bridge. Supports walk-forward optimization (`backtest/engine/walk_forward.py`). Data pipeline scripts in `backtest/data/`.

## Code Style

### Rust
- `rustfmt.toml`: max_width=100, imports_granularity="Crate", group_imports="StdExternalCrate"
- Clippy treats all warnings as errors (`-D warnings`)

### Python
- ruff: line-length=100, target-version="py312"
- Lint rules: E, F, I, N, W, UP, B, A, SIM

## Issue Tracking

This project uses **Linear** for issue tracking. See the `nikitok-linear-spec-workflow` rule for the full workflow: Linear issues map to `/spec` plans, sub-issues to plan tasks, one PR per epic.

# CM.HFT -- High-Frequency Trading Platform for BTC

> Polyglot HFT platform for cryptocurrency trading on Binance and Bybit exchanges.
> Rust (trading core) | Python (research & backtesting) | Go (infrastructure)

---

## Table of Contents

- [Overview](#overview)
- [Language Choice](#language-choice)
- [Architecture](#architecture)
- [Components](#components)
  - [Market Data Handler](#1-market-data-handler-rust)
  - [Strategy Engine](#2-strategy-engine-rust)
  - [Order Management System](#3-order-management-system-rust)
  - [Risk Manager](#4-risk-manager-rust-pre-trade)
  - [Execution Engine / Exchange Gateway](#5-execution-engine--exchange-gateway-rust)
  - [Data Layer](#6-data-layer)
- [Backtesting Framework](#backtesting-framework-python--rust)
- [Infrastructure](#infrastructure)
- [Anti-patterns to Avoid](#anti-patterns-to-avoid)
- [Project Structure](#project-structure)

---

## Overview

CM.HFT is a high-frequency trading platform purpose-built for BTC perpetual and spot markets on Binance and Bybit. The system is designed around three core principles:

1. **Minimal latency** -- wire-to-decision in microseconds, not milliseconds.
2. **Correctness under pressure** -- memory safety and data-race freedom enforced at compile time.
3. **Reproducibility** -- the exact same strategy code runs in production and backtesting.

The platform uses a polyglot architecture where each language is chosen for what it does best: Rust for the latency-critical trading core, Python for research and data analysis, and Go for infrastructure tooling.

---

## Language Choice

| Component | Language | Rationale |
|---|---|---|
| Trading engine, OMS, market data | **Rust** | Zero-cost abstractions, no GC pauses, memory safety. Mature ecosystem (`tokio`, `crossbeam`, `serde`). |
| Strategy research, backtesting | **Python** | Ecosystem (`numpy`, `pandas`, `polars`, `scikit-learn`, `pytorch`). |
| Infrastructure, monitoring | **Go** | Simplicity, goroutines, fast compilation. |

### Why Rust over C++

In a greenfield project, Rust provides comparable latency to C++ but eliminates entire classes of bugs at compile time -- use-after-free, data races, null pointer dereferences, and buffer overflows. For an HFT system where **bugs equal money loss**, this tradeoff is decisive. The borrow checker is a strict reviewer that never sleeps.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE LAYER                  │
│  Monitoring · Alerting · Logging · Config Management     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│                     TRADING CORE (Rust)                  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ Market Data  │→ │  Strategy    │→ │    Order      │ │
│  │   Handler    │  │   Engine     │  │  Management   │ │
│  └──────────────┘  └──────────────┘  └───────┬───────┘ │
│         ↑                                     │         │
│  ┌──────┴───────┐  ┌──────────────┐  ┌───────▼───────┐ │
│  │  Order Book  │  │    Risk      │← │  Execution    │ │
│  │ Reconstructor│  │   Manager    │  │    Engine     │ │
│  └──────────────┘  └──────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────┘
                       │           ↑
                       ▼           │
┌──────────────────────────────────────────────────────────┐
│              EXCHANGE CONNECTIVITY (Rust)                 │
│  ┌─────────────────┐        ┌──────────────────────┐    │
│  │ Binance Gateway │        │  Bybit Gateway       │    │
│  │  WS + REST      │        │  WS + REST           │    │
│  └─────────────────┘        └──────────────────────┘    │
└──────────────────────────────────────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────────────┐
│              DATA LAYER                                   │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │ Time-series│  │  Redis     │  │  Object Storage    │ │
│  │ (QuestDB)  │  │ (state)   │  │  (historical ticks)│ │
│  └────────────┘  └────────────┘  └────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

Data flows top-down from exchange connections through the trading core and back out as orders. The infrastructure layer wraps everything with observability. Each box is a separate crate or service with well-defined interfaces.

---

## Components

### 1. Market Data Handler (Rust)

The market data handler is the first point of contact with exchange data. Every microsecond saved here propagates through the entire pipeline.

**Design decisions:**

- **Custom WebSocket client** built on `tokio-tungstenite` -- exchange SDKs add unnecessary overhead and abstraction layers that inflate latency.
- **Lock-free SPSC ring buffer** (`crossbeam`) for zero-contention inter-thread data transfer between the network I/O thread and the strategy thread.
- **Normalized internal format** that abstracts away exchange-specific wire formats:

```rust
pub struct NormalizedTick {
    pub exchange: Exchange,
    pub symbol: Symbol,
    pub bid: Decimal,
    pub ask: Decimal,
    pub bid_qty: Decimal,
    pub ask_qty: Decimal,
    pub timestamp_ns: u64,
}
```

- **Incremental L2 order book** with diff-updates. Implementation uses a `BTreeMap` for correctness or a sorted array for cache locality depending on the price level count.
- **Timestamping** via `clock_gettime(CLOCK_MONOTONIC)` for nanosecond-precision local timing, independent of NTP drift.

**Target metrics:**

| Metric | Target |
|---|---|
| Wire-to-internal latency | < 50 us |
| Book update latency | < 5 us |

---

### 2. Strategy Engine (Rust)

Strategies are compiled Rust code, not interpreted scripts. This eliminates FFI overhead and enables the compiler to inline and optimize the hot path aggressively.

```rust
pub trait Strategy: Send + Sync {
    fn on_book_update(&mut self, book: &OrderBook, ctx: &mut TradingContext);
    fn on_trade(&mut self, trade: &Trade, ctx: &mut TradingContext);
    fn on_fill(&mut self, fill: &Fill, ctx: &mut TradingContext);
    fn on_timer(&mut self, now_ns: u64, ctx: &mut TradingContext);
}
```

`TradingContext` provides the strategy with everything it needs to act:

- Order submission (new, cancel, amend)
- Current position and balance state
- Risk limit parameters
- Pre-allocated scratch buffers

**Key constraint:** A strategy is a pure function of market state. No heap allocations on the hot path. All buffers are pre-allocated at initialization. No locks, no async, no I/O.

---

### 3. Order Management System (Rust)

The OMS tracks every order through its full lifecycle using a deterministic state machine:

```
New → Sent → Acked → PartialFill → Filled
                  ↘                ↗
                   → Cancelled
                   → Rejected
```

**Core responsibilities:**

- **Position tracker** -- maintains two views:
  - *Real position*: derived exclusively from confirmed fills.
  - *Theoretical position*: real position + pending order exposure.
- **Reconciliation** -- periodic comparison of internal state against exchange REST API to detect and correct drift. Exchanges lose messages; this is not optional.
- **Idempotency** -- every order carries a `client_order_id` generated from a monotonic counter. Fill deduplication prevents double-counting in the event of duplicate execution reports.

---

### 4. Risk Manager (Rust, pre-trade)

Every order passes through a chain of risk checks before reaching the exchange gateway. If any check fails, the order is rejected locally -- no round-trip to the exchange.

| Check | Description |
|---|---|
| **Max Position** | Reject if resulting position would exceed configured limit. |
| **Max Order Size** | Reject orders above per-order size threshold. |
| **Rate Limit** | Enforce exchange rate limits locally to avoid IP bans. |
| **PnL Drawdown** | Reject new orders if unrealized + realized PnL drawdown exceeds threshold. |
| **Fat Finger** | Reject orders with price deviating more than N% from mid-market. |
| **Circuit Breaker** | Hard stop on anomalous conditions (see below). |

**Circuit breaker triggers:**

- PnL drawdown exceeding N% of capital
- Anomalous latency (exchange response times outside normal distribution)
- Position mismatch between internal state and exchange reconciliation
- Rapid fill rate indicating possible adverse selection or market dislocation

**Kill switch:** A dedicated HTTP endpoint that immediately cancels all open orders, flattens positions, and halts the strategy engine. Accessible from monitoring dashboards and alerting pipelines.

---

### 5. Execution Engine / Exchange Gateway (Rust)

The gateway abstraction allows strategies to be exchange-agnostic while each implementation handles the specifics of wire protocol, authentication, and rate limiting.

```rust
pub trait ExchangeGateway: Send + Sync {
    async fn place_order(&self, order: &NewOrder) -> Result<OrderAck>;
    async fn cancel_order(&self, id: &OrderId) -> Result<CancelAck>;
    async fn amend_order(
        &self,
        id: &OrderId,
        new_price: Decimal,
        new_qty: Decimal,
    ) -> Result<AmendAck>;
}
```

**Binance gateway:**

- Persistent HTTP/1.1 connections with keep-alive to avoid TLS handshake overhead on every request.
- Pre-signed URL templates -- the HMAC signature is computed over a template and only the variable parts (price, quantity, timestamp) are substituted at send time.

**Bybit gateway:**

- v5 unified API for both spot and derivatives.

**Shared infrastructure:**

- HMAC signing via the `ring` crate (no OpenSSL dependency, pure Rust, constant-time comparison).
- Token bucket rate limiter per endpoint, configured per exchange rate limit documentation.

---

### 6. Data Layer

| Storage | Purpose | Details |
|---|---|---|
| **QuestDB** | Time-series data | Ticks, order book snapshots, fills, PnL curves. Column-oriented, SQL-compatible, optimized for append-heavy workloads. |
| **Redis** | Runtime state | Current positions, strategy configuration, feature flags. Used for state that must survive process restarts but does not require durability guarantees. |
| **S3 / MinIO** | Historical archive | Long-term storage of tick data and book snapshots in Parquet format. Partitioned by date and symbol for efficient range scans. |

---

## Backtesting Framework (Python + Rust)

The backtesting framework bridges the research and production worlds. The central design goal is **no strategy rewrite** -- the same compiled Rust strategy runs in both environments.

```
Historical Data (Parquet/S3)
        │
        ▼
┌─────────────────────────┐
│   Event Replay Engine   │  ← Python orchestration
│   (Rust core via PyO3)  │  ← Rust for speed
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Strategy (same Rust     │
│  code as production!)    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Simulated Exchange      │
│  (matching, fees,        │
│   latency model)         │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Analytics / Reporting   │  ← Python (pandas, plotly)
└─────────────────────────┘
```

### Key Principles

1. **Code parity.** Strategy code is compiled into a shared library (`.so`) and loaded into the Python backtester via PyO3 bindings. The exact same binary logic executes in production and simulation.

2. **Realistic simulation.** The simulated exchange models:
   - Latency injection (configurable distribution matching observed exchange latencies)
   - Fee model (maker/taker rates, VIP tier discounts)
   - Slippage based on order book depth
   - Funding rate impact for perpetual contracts

3. **Data pipeline.**
   - **Collection:** Exchange historical API --> raw JSON --> Parquet --> S3
   - **Live recording:** Market data handler --> QuestDB --> nightly export to Parquet
   - **Replay:** Parquet files streamed through the event engine in timestamp order

4. **Walk-forward optimization.** In-sample/out-of-sample splits to protect against overfitting. Parameter optimization is performed on training windows and validated on unseen data before advancing the window.

### Evaluation Metrics

| Metric | Target / Purpose |
|---|---|
| Sharpe ratio | > 2.0 (annualized) |
| Max drawdown | Minimize; hard limit per strategy |
| Win rate | Directional accuracy |
| Profit factor | Gross profit / gross loss |
| Fill ratio | Percentage of orders that execute |
| Latency sensitivity | Strategy PnL as a function of added latency |

---

## Infrastructure

### Server Placement

Binance and Bybit matching engines are located in **AWS ap-northeast-1 (Tokyo)**.

**Production deployment:** AWS ap-northeast-1, instance type `c6i.2xlarge` or bare metal for kernel bypass capabilities.

- **Isolated CPU cores** (`isolcpus` kernel parameter) -- trading threads are pinned to dedicated cores with no other processes scheduled on them.
- **NUMA-aware thread pinning** -- memory allocations for trading threads are local to the same NUMA node as their pinned CPU core.
- **Tuned network stack** -- see below.

### Network Optimizations

```bash
# Enable busy polling to reduce network latency
net.core.busy_poll = 50
net.core.busy_read = 50

# Disable Nagle's algorithm
net.ipv4.tcp_nodelay = 1

# Increase socket buffer sizes
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
```

These settings reduce tail latency by keeping the NIC active during idle periods (`busy_poll`), eliminating write coalescing (`tcp_nodelay`), and preventing buffer-related drops under burst conditions.

### Monitoring (Go + Prometheus + Grafana)

**Dashboards:**

- Latency distribution: p50 / p99 / p99.9 for each stage of the pipeline
- Real-time PnL and cumulative returns
- Open positions and exposure
- Exchange connectivity status and error rates
- System metrics: CPU utilization per core, memory, network I/O

**Alerting rules:**

| Condition | Action |
|---|---|
| PnL drawdown exceeds threshold | PagerDuty escalation |
| Latency spike (p99 > 2x baseline) | Slack notification |
| WebSocket connection drop | Automatic reconnect + alert |
| Position mismatch detected | Kill switch activation |

---

## Anti-patterns to Avoid

These are hard-won lessons. Each one has a direct cost in latency, correctness, or both.

1. **Do not use exchange SDKs in the hot path.** They are designed for convenience, not performance. Build your own WebSocket and REST clients.

2. **Do not use standard JSON parsing.** Use `simd-json` for SIMD-accelerated parsing or zero-copy deserialization. Standard `serde_json` is fast but not fast enough for the innermost loop.

3. **Do not store state in a database on the hot path.** All trading state lives in memory. Persist asynchronously to QuestDB/Redis on a separate thread.

4. **Do not use async in the strategy hot path.** The strategy callback must be synchronous, lock-free, and allocation-free. Async runtimes introduce unpredictable latency from task scheduling.

5. **Do not ignore reconciliation.** Exchanges lose messages, duplicate fills, and lag on status updates. Periodic reconciliation against the REST API is mandatory, not optional.

---

## Project Structure

```
cm.hft/
├── crates/
│   ├── core/           # Shared types, traits, configuration
│   ├── market-data/    # WebSocket client, order book reconstruction
│   ├── strategy/       # Strategy trait + implementations
│   ├── oms/            # Order management system
│   ├── risk/           # Pre-trade risk checks
│   ├── execution/      # Exchange gateways (Binance, Bybit)
│   ├── recorder/       # Market data recording to QuestDB
│   └── pybridge/       # PyO3 bindings for backtester
├── backtest/
│   ├── engine/         # Python backtesting orchestration
│   ├── data/           # Data pipeline scripts
│   └── notebooks/      # Jupyter analysis notebooks
├── infra/
│   ├── monitoring/     # Go monitoring service
│   ├── deploy/         # Terraform, Docker, Kubernetes configs
│   └── scripts/        # Operational scripts
├── Cargo.toml          # Rust workspace root
├── pyproject.toml      # Python project configuration
├── README.md
└── TASKS.md
```

---

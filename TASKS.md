# CM.HFT -- Implementation Task List

> **BTC HFT Trading Platform** targeting Binance and Bybit.
> Stack: Rust (trading core) | Python (backtesting) | Go (monitoring).

---

## Phase Summary

| Phase | Name                              | Priority | Status      |
|-------|-----------------------------------|----------|-------------|
| 0     | Project Bootstrap                 | P0       | Not started |
| 1     | Core Types & Exchange Connectivity| P0       | Not started |
| 2     | Market Data Recording             | P0       | Not started |
| 3     | Order Management System           | P0       | Not started |
| 4     | Risk Management                   | P0       | Not started |
| 5     | Strategy Framework                | P1       | Not started |
| 6     | Backtesting Framework             | P1       | Not started |
| 7     | Infrastructure & Monitoring       | P1       | Not started |
| 8     | Paper Trading & Validation        | P0       | Not started |
| 9     | Production Launch                 | P1       | Not started |

**Priority key:** P0 = blocks everything downstream, must complete first. P1 = important but can proceed in parallel with other P1 work once P0 dependencies are met.

---

## Phase 0: Project Bootstrap

**Dependencies:** None
**Priority:** P0 -- must be done first, every other phase depends on this.

### Tasks

- [ ] **0.1 Initialize Rust workspace**
  Create `Cargo.toml` at the project root as a workspace manifest. Define all member crates:
  `crates/core`, `crates/market-data`, `crates/strategy`, `crates/oms`, `crates/risk`,
  `crates/execution`, `crates/recorder`, `crates/pybridge`.
  Each crate gets its own `Cargo.toml` with appropriate dependencies.
  Set workspace-level dependency versions for shared crates (`tokio`, `serde`, `tracing`, `anyhow`).
  Use Rust edition 2021. Set `opt-level = 3` for release, `opt-level = 0` for dev.
  Add a `[profile.release]` section with `lto = "thin"` and `codegen-units = 1`.

- [ ] **0.2 Set up CI/CD pipeline**
  Create `.github/workflows/ci.yml` with jobs:
  - `check`: `cargo fmt --check`, `cargo clippy -- -D warnings`
  - `test`: `cargo test --workspace`
  - `build`: `cargo build --release`
  - `python`: `ruff check backtest/`, `pytest backtest/`
  - `go`: `go vet ./...`, `go test ./...` in `infra/monitoring/`
  Trigger on push to `main` and `develop`, and on all pull requests.
  Cache `~/.cargo/registry`, `target/`, `.venv/`.

- [ ] **0.3 Configure development environment**
  Create `rustfmt.toml` (max width 100, imports granularity = "Crate").
  Create `clippy.toml` or workspace-level clippy lints in `Cargo.toml`.
  Create `.pre-commit-config.yaml` with hooks: `cargo fmt`, `cargo clippy`, `ruff`, `ruff format`.
  Add `.editorconfig` for consistent formatting across languages.
  Document setup steps in `README.md`.

- [ ] **0.4 Set up Python project**
  Create `pyproject.toml` at project root. Use `uv` as package manager.
  Configure `ruff` for linting and formatting. Set Python 3.12+ as minimum.
  Define dependencies: `numpy`, `pandas`, `pyarrow`, `polars`, `maturin`, `pytest`, `jupyter`.
  Create `backtest/engine/`, `backtest/data/`, `backtest/notebooks/` directories with `__init__.py` files.
  Verify `uv sync && uv run pytest` works (with a trivial passing test).

- [ ] **0.5 Create Docker Compose for local infrastructure**
  Create `docker-compose.yml` with services:
  - **QuestDB** (port 9000 for web console, 9009 for ILP ingestion) -- for market data storage.
  - **Redis** (port 6379) -- for runtime configuration, circuit breaker state, pub/sub.
  - **MinIO** (port 9000/9001) -- S3-compatible storage for Parquet files and backtest data.
  Add health checks for each service. Create `infra/scripts/init-minio.sh` to create default buckets.
  Document usage in README.

- [ ] **0.6 Set up logging framework**
  Add `tracing`, `tracing-subscriber`, `tracing-appender` to `crates/core/`.
  Configure structured JSON output for production, pretty-print for development.
  Add span timing for critical paths (order submission, market data processing).
  Create a shared `init_tracing()` function that all binaries call.
  Include fields: timestamp (nanos), level, module, span, message, and custom fields.
  Ensure log output does NOT include API keys or secrets (add a sanitization layer).

---

## Phase 1: Core Types & Exchange Connectivity

**Dependencies:** Phase 0
**Priority:** P0

**Acceptance criteria:**
- Can connect to both Binance and Bybit and receive real-time L2 book updates.
- Order book reconstruction matches exchange snapshot (verified by periodic snapshot comparison).
- Can place and cancel orders on testnet without errors.
- Latency logging shows wire-to-internal processing < 100us (initial target).

### Tasks

- [ ] **1.1 Define core types in `crates/core/`**
  Implement:
  - `Price`: fixed-point decimal (i64 mantissa + u8 scale). No floating-point on the hot path.
    Support arithmetic ops via `std::ops` traits. Include `From<f64>` for convenience (non-hot-path only).
  - `Quantity`: same fixed-point approach.
  - `OrderId`: newtype around `u64` (internal) and `String` (exchange-assigned).
  - `Symbol`: enum or newtype (`BTCUSDT`, `BTCUSD`). Include exchange-specific symbol mapping.
  - `Exchange`: enum `{ Binance, Bybit }`.
  - `Side`: enum `{ Buy, Sell }`.
  - `OrderType`: enum `{ Limit, Market, PostOnly }`.
  - `Timestamp`: newtype around `u64` nanoseconds since epoch. Include conversion utilities.
  Derive `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq`, `Hash`, `serde::Serialize/Deserialize` where appropriate.
  Write comprehensive unit tests for `Price` arithmetic (overflow, rounding).

- [ ] **1.2 Implement configuration system**
  Use `config` crate or custom TOML parser in `crates/core/config.rs`.
  Support layered config: defaults -> TOML file -> environment variable overrides.
  Define config structs:
  - `ExchangeConfig`: API key (from env only), secret (from env only), testnet flag, base URLs.
  - `MarketDataConfig`: symbols, depth levels, reconnect parameters.
  - `RiskConfig`: max position, max order size, drawdown limits.
  - `RecorderConfig`: QuestDB endpoint, batch size, flush interval.
  API keys and secrets must ONLY come from environment variables or a secrets manager, never from config files.
  Write tests for config loading and env var overrides.

- [ ] **1.3 Implement Binance WebSocket client**
  In `crates/market-data/src/binance.rs`:
  - Connect to Binance WebSocket stream (`wss://stream.binance.com:9443/ws/`).
  - Subscribe to `depth@100ms` (L2 incremental updates) and `trade` streams for BTCUSDT.
  - Parse JSON messages into internal types using `serde_json`.
  - Implement initial snapshot fetch via REST (`GET /api/v3/depth?symbol=BTCUSDT&limit=1000`).
  - Apply incremental updates, validating `lastUpdateId` sequencing.
  - Use `tokio-tungstenite` for async WebSocket. Pin to a dedicated tokio runtime thread.
  - Log wire-to-parsed latency on every message (timestamp delta).

- [ ] **1.4 Implement Bybit WebSocket client**
  In `crates/market-data/src/bybit.rs`:
  - Connect to Bybit WebSocket v5 (`wss://stream.bybit.com/v5/public/linear`).
  - Subscribe to `orderbook.200.BTCUSDT` and `publicTrade.BTCUSDT`.
  - Handle Bybit's snapshot + delta update protocol.
  - Parse messages, apply updates with sequence validation.
  - Same latency logging as Binance client.

- [ ] **1.5 Implement WebSocket reconnection logic**
  Create a generic `WsConnection` wrapper in `crates/market-data/src/ws.rs`:
  - Exponential backoff: 100ms, 200ms, 400ms, ..., capped at 30s.
  - Jitter: add random 0-50% to backoff interval to avoid thundering herd.
  - On reconnect: re-subscribe, re-fetch snapshot, log gap duration.
  - Emit `ConnectionState` events (Connected, Disconnected, Reconnecting) for monitoring.
  - Configurable max reconnect attempts before alerting (default: unlimited, but alert after 5 failures).

- [ ] **1.6 Implement normalized market data types**
  In `crates/core/src/market_data.rs`:
  - `BookUpdate { exchange, symbol, timestamp, bids: Vec<(Price, Quantity)>, asks: Vec<(Price, Quantity)>, is_snapshot: bool }`
  - `Trade { exchange, symbol, timestamp, price, quantity, side, trade_id }`
  - `BookLevel { price: Price, quantity: Quantity }`
  - Implement `From<BinanceDepthUpdate>` and `From<BybitBookUpdate>` for `BookUpdate`.
  Normalization must be zero-copy where possible (avoid cloning strings).

- [ ] **1.7 Implement L2 order book**
  In `crates/market-data/src/orderbook.rs`:
  - Use `BTreeMap<Price, Quantity>` for bids (reverse order) and asks (natural order).
  - Methods: `apply_update()`, `best_bid()`, `best_ask()`, `mid_price()`, `spread()`, `depth_at(levels)`.
  - Handle quantity = 0 as level removal.
  - Implement checksum validation (Binance and Bybit have different checksum algorithms).
  - If checksum fails, log error and trigger full snapshot re-fetch.
  - Benchmark: `apply_update` must complete in < 500ns for typical update sizes.
  Write property-based tests (using `proptest`): random update sequences must never produce negative quantities.

- [ ] **1.8 Write integration tests against testnet**
  Create `tests/` directory in `crates/market-data/`:
  - Test: connect to Binance testnet, receive 100 book updates, verify book integrity.
  - Test: connect to Bybit testnet, receive 100 book updates, verify book integrity.
  - Test: disconnect network (simulate), verify reconnection and book recovery.
  Mark tests with `#[ignore]` by default (require `--ignored` flag or CI environment variable).
  Document required testnet API keys in `README.md`.

- [ ] **1.9 Implement Binance REST client**
  In `crates/execution/src/binance.rs`:
  - Implement order placement: `POST /api/v3/order` (LIMIT, MARKET, LIMIT_MAKER).
  - Implement order cancel: `DELETE /api/v3/order`.
  - Implement order amend: `PUT /api/v3/order` (cancel-replace).
  - Implement account info: `GET /api/v3/account`.
  - Implement open orders query: `GET /api/v3/openOrders`.
  - Use `reqwest` with connection pooling. Set timeouts: connect 2s, read 5s.
  - Parse all responses into typed structs. Handle Binance error codes (map to internal error enum).

- [ ] **1.10 Implement Bybit REST v5 client**
  In `crates/execution/src/bybit.rs`:
  - Implement order placement: `POST /v5/order/create`.
  - Implement order cancel: `POST /v5/order/cancel`.
  - Implement order amend: `POST /v5/order/amend`.
  - Implement position info: `GET /v5/position/list`.
  - Implement wallet balance: `GET /v5/account/wallet-balance`.
  Handle Bybit-specific error codes and rate limit headers.

- [ ] **1.11 Implement HMAC-SHA256 request signing**
  In `crates/execution/src/signing.rs`:
  - Binance: sort query params, HMAC-SHA256 of query string, append `signature` param.
  - Bybit: HMAC-SHA256 of `timestamp + api_key + recv_window + body`.
  - Use `ring` crate for HMAC (constant-time, audited crypto).
  - NEVER log the secret key or the signature in production logs.
  Write tests with known test vectors from exchange documentation.

- [ ] **1.12 Implement rate limiter**
  In `crates/execution/src/rate_limiter.rs`:
  - Token bucket algorithm: configurable rate and burst per endpoint category.
  - Binance limits: 1200 request weight/minute (orders = 1 weight, heavy queries = 5-20).
  - Bybit limits: 120 requests/second for order endpoints.
  - Track used weight from response headers (`X-MBX-USED-WEIGHT-1M` for Binance).
  - If approaching limit (>80%), slow down. If exceeded, queue and wait.
  - Expose metrics: current usage, queue depth.
  Write tests: submit 200 requests, verify rate stays within bounds.

---

## Phase 2: Market Data Recording & Data Pipeline

**Dependencies:** Phase 1 (market data handler must be working)
**Priority:** P0

**Acceptance criteria:**
- Continuous recording with no gaps for 24h+ (verified by gap detection tool).
- Parquet files contain correct L2 book snapshots and trades with nanosecond timestamps.
- Data can be replayed in correct chronological order across exchanges.

### Tasks

- [ ] **2.1 Implement market data recorder**
  In `crates/recorder/src/lib.rs`:
  - Subscribe to normalized market data events from `crates/market-data`.
  - Buffer events in memory (ring buffer, configurable size, default 10,000 events).
  - Flush to QuestDB in batches (configurable interval, default 100ms).
  - Record both L2 book updates (top 20 levels) and individual trades.
  - Include metadata: exchange, symbol, local receive timestamp, exchange timestamp.
  - Handle backpressure: if QuestDB is slow, drop oldest buffered events and log warning.

- [ ] **2.2 Implement QuestDB client (ILP protocol)**
  In `crates/recorder/src/questdb.rs`:
  - Use InfluxDB Line Protocol (ILP) over TCP for fast ingestion.
  - Implement connection pooling (single persistent TCP connection with reconnect).
  - Format: `book_update,exchange=binance,symbol=BTCUSDT bid1=50000.50,ask1=50001.00 1234567890000000000`
  - Batch writes: accumulate lines, flush with newline delimiter.
  - Target ingestion rate: 100,000+ rows/second.
  - Implement health check (ping QuestDB REST endpoint).
  Write integration test: write 1M rows, query back, verify count and values.

- [ ] **2.3 Create data export pipeline**
  In `infra/scripts/export_data.py` (or `backtest/data/export.py`):
  - Query QuestDB via REST SQL endpoint for a date range.
  - Convert to Parquet format using `pyarrow` or `polars`.
  - Partition files by date and exchange: `data/BTCUSDT/binance/2025-01-15.parquet`.
  - Upload to MinIO/S3 with checksums.
  - Create a manifest file listing all available data ranges.
  - Schedule via cron or Airflow (document both options).

- [ ] **2.4 Build historical data downloader**
  In `backtest/data/download_historical.py`:
  - Binance: use `GET /api/v3/klines` for OHLCV and `GET /api/v3/aggTrades` for tick data.
  - Bybit: use `GET /v5/market/kline` and `GET /v5/market/recent-trade`.
  - Download in paginated chunks, respecting rate limits (1200 weight/min for Binance).
  - Save directly to Parquet with consistent schema.
  - Support resumable downloads (track last downloaded timestamp).
  - Target: download 6 months of tick-level data for BTCUSDT from both exchanges.

- [ ] **2.5 Implement data validation**
  In `backtest/data/validate.py`:
  - Gap detection: find time gaps > 1 second in book update stream.
  - Duplicate detection: find duplicate timestamps or sequence IDs.
  - Consistency checks: bid < ask (no crossed book), quantities >= 0.
  - Statistics report: message count, average update rate, gap summary.
  - Generate validation report as JSON for automated monitoring.
  Run validation on every exported Parquet file before it enters the backtest pipeline.

- [ ] **2.6 Record 2 weeks of live L2 BTC data**
  Operational task:
  - Deploy recorder on a stable server (or local machine with UPS).
  - Record BTCUSDT L2 (20 levels) + trades from both Binance and Bybit.
  - Monitor for gaps daily. Re-fetch snapshots if gaps detected.
  - Export to Parquet daily. Validate each day's data.
  - Store in MinIO/S3. This dataset becomes the baseline for backtesting.

---

## Phase 3: Order Management System

**Dependencies:** Phase 1 (exchange REST clients)
**Priority:** P0

**Acceptance criteria:**
- All order lifecycle states handled correctly (unit test coverage > 95% for state machine).
- Position never drifts from exchange reported position (reconciliation catches mismatches within 1 minute).
- Full audit trail of every order event, queryable by order ID or time range.

### Tasks

- [ ] **3.1 Implement order state machine**
  In `crates/oms/src/order.rs`:
  - States: `New -> Sent -> Acked -> PartialFill -> Filled | Cancelled | Rejected`
  - Transitions validated at compile time where possible (use enum + match).
  - Each transition emits an `OrderEvent` with timestamp and metadata.
  - Invalid transitions log error and trigger alert (do NOT panic in production).
  - Store: `HashMap<OrderId, OrderState>` with O(1) lookup.
  - Methods: `submit()`, `on_ack()`, `on_fill()`, `on_cancel_ack()`, `on_reject()`.
  Write exhaustive unit tests for every valid and invalid state transition.

- [ ] **3.2 Implement position tracker**
  In `crates/oms/src/position.rs`:
  - Track per-symbol, per-exchange: `net_quantity`, `avg_entry_price`, `unrealized_pnl`, `realized_pnl`.
  - Update on every fill event.
  - Also track "theoretical" position including pending (sent but not yet filled) orders.
  - Expose `worst_case_position()` = realized + all pending orders filled.
  - PnL calculation: use mark-to-market with current mid price.
  - Thread-safe: use `Arc<RwLock<>>` or lock-free structure if on hot path.

- [ ] **3.3 Implement fill deduplication**
  In `crates/oms/src/dedup.rs`:
  - Generate unique `client_order_id` using monotonic counter + instance ID.
  - Format: `cm_{instance_id}_{counter}` (e.g., `cm_01_000042`).
  - Maintain a set of processed fill IDs (exchange trade IDs).
  - On duplicate fill: log warning, skip processing.
  - Periodically prune old entries (older than 24h) to prevent memory growth.

- [ ] **3.4 Implement position reconciliation**
  In `crates/oms/src/reconciliation.rs`:
  - Periodically (every 30s configurable) query exchange REST API for current position.
  - Compare exchange-reported position with internal position tracker.
  - If mismatch detected:
    1. Log error with full details (internal vs exchange values).
    2. Emit alert event.
    3. If difference > threshold (configurable, default 0.001 BTC): trigger circuit breaker.
    4. Adjust internal position to match exchange (exchange is source of truth).
  - Run reconciliation on startup before accepting any new orders.

- [ ] **3.5 Implement order event journal**
  In `crates/oms/src/journal.rs`:
  - Append-only log of all `OrderEvent` entries.
  - Write to memory-mapped file for crash recovery (`memmap2` crate).
  - Schema: `[timestamp_ns, event_type, order_id, exchange, symbol, side, price, qty, status, metadata]`.
  - Support replay from journal on startup (recover in-flight orders).
  - Rotate journal files daily. Compress old journals with zstd.
  - Expose query interface: events by order_id, events in time range.

- [ ] **3.6 Write unit tests for OMS**
  Cover:
  - Full order lifecycle: New -> Sent -> Acked -> Filled (happy path).
  - Partial fills: 3 partial fills -> Filled.
  - Cancel flow: Sent -> Acked -> CancelSent -> Cancelled.
  - Reject flow: Sent -> Rejected.
  - Invalid transitions: Filled -> Cancelled (must fail gracefully).
  - Position tracker: 10 fills, verify final position and PnL.
  - Deduplication: send same fill twice, verify position updates only once.
  Target: > 95% line coverage on `crates/oms/`.

- [ ] **3.7 Write integration tests with exchange testnet**
  - Submit limit order on Binance testnet, verify Acked state.
  - Cancel order, verify Cancelled state.
  - Submit market order, verify Filled state and position update.
  - Run reconciliation, verify position matches exchange.
  - Same tests for Bybit testnet.
  Mark as `#[ignore]` (testnet tests require API keys and network).

---

## Phase 4: Risk Management

**Dependencies:** Phase 3 (OMS and position tracker must be working)
**Priority:** P0 -- absolutely no real money trading without this.

**Acceptance criteria:**
- Every order passes through ALL risk checks before exchange submission. No bypass possible.
- Circuit breaker cancels all open orders and blocks new orders within 100ms of trigger.
- Kill switch works from any HTTP client (curl, browser, monitoring service).
- All risk parameters configurable without restart (via Redis).

### Tasks

- [ ] **4.1 Implement pre-trade risk check pipeline**
  In `crates/risk/src/pipeline.rs`:
  - Define `RiskCheck` trait: `fn check(&self, order: &Order, context: &RiskContext) -> Result<(), RiskReject>`.
  - Pipeline executes all checks in sequence. First rejection stops the pipeline.
  - `RiskContext` contains: current positions, open orders, recent PnL, market data.
  - Pipeline is called by OMS before every order submission. This is NOT optional.
  - Log every check result (pass/fail) with order details for audit.

- [ ] **4.2 Implement max position size check**
  In `crates/risk/src/checks/max_position.rs`:
  - Reject if `current_position + order_quantity > max_position_size`.
  - Use `worst_case_position()` from position tracker (includes pending orders).
  - Configurable per symbol. Default for BTCUSDT: 0.1 BTC (start very small).
  - Log rejection reason with current position and requested size.

- [ ] **4.3 Implement max order size check**
  In `crates/risk/src/checks/max_order.rs`:
  - Reject if `order_quantity > max_single_order_size`.
  - Default: 0.01 BTC per order (force small orders during early trading).
  - Separate limits for different order types (market orders get tighter limits).

- [ ] **4.4 Implement rate limit check**
  In `crates/risk/src/checks/rate_limit.rs`:
  - Track orders submitted per second / per minute.
  - Reject if exceeding internal rate limit (stricter than exchange limits).
  - Default: max 10 orders/second, 200 orders/minute.
  - This prevents bugs from flooding the exchange with orders.

- [ ] **4.5 Implement PnL drawdown check**
  In `crates/risk/src/checks/drawdown.rs`:
  - Track rolling PnL over configurable window (default: 1 hour).
  - Reject new orders if drawdown exceeds threshold (default: -$100 for initial testing).
  - Also track daily PnL. Hard stop at daily loss limit (default: -$500).
  - On daily loss limit hit: trigger circuit breaker, not just reject.

- [ ] **4.6 Implement fat finger check**
  In `crates/risk/src/checks/fat_finger.rs`:
  - Reject limit orders with price > X% away from current mid price.
  - Default: 1% for limit orders, 0.5% for post-only orders.
  - This catches stale prices, calculation errors, and off-by-10x bugs.

- [ ] **4.7 Implement circuit breaker**
  In `crates/risk/src/circuit_breaker.rs`:
  - Trigger conditions: PnL drawdown exceeded, reconciliation mismatch, manual kill switch.
  - On trigger:
    1. Set global `TRADING_ENABLED` flag to `false` (atomic bool).
    2. Cancel ALL open orders on ALL exchanges (fire-and-forget, then verify).
    3. Log the trigger reason and current state.
    4. Send alert via configured channel (webhook/PagerDuty).
    5. Block all new order submissions until manually reset.
  - Reset requires explicit action: REST API call with confirmation token.
  - Store circuit breaker state in Redis for persistence across restarts.

- [ ] **4.8 Implement HTTP kill switch endpoint**
  In the main binary (or a separate lightweight HTTP server using `axum` or `warp`):
  - `POST /kill` -- triggers circuit breaker immediately. No authentication required for safety.
  - `GET /status` -- returns current trading status, positions, PnL, circuit breaker state.
  - `POST /reset` -- resets circuit breaker (requires auth token from config).
  - `GET /risk/params` -- returns current risk parameters.
  - Bind to `0.0.0.0:8080` (configurable). Must start even if trading fails to initialize.

- [ ] **4.9 Implement risk parameter hot-reload**
  In `crates/risk/src/config.rs`:
  - Store risk parameters in Redis hash: `risk:params:{symbol}`.
  - Poll Redis every 5 seconds for parameter changes.
  - On change: validate new parameters (sanity checks), then swap atomically.
  - Log every parameter change with old and new values.
  - If Redis is unavailable: continue with last known parameters, log warning.

- [ ] **4.10 Write unit tests for every risk check**
  For each check:
  - Test: order within limits -> pass.
  - Test: order exceeding limit -> reject with correct reason.
  - Test: edge case at exact boundary -> document whether it passes or rejects.
  - Test: multiple orders accumulating to exceed limit.
  For circuit breaker:
  - Test: trigger -> verify all cancels sent, flag set, new orders blocked.
  - Test: reset -> verify trading resumes.

- [ ] **4.11 Write integration test: rapid loss triggers circuit breaker**
  Simulate scenario:
  1. Position tracker shows -$50 PnL.
  2. Three more losing fills arrive in quick succession, pushing to -$110.
  3. Verify circuit breaker fires when threshold (-$100) is crossed.
  4. Verify cancel-all is issued.
  5. Verify subsequent order submission is rejected.
  6. Verify alert event is emitted.

---

## Phase 5: Strategy Framework

**Dependencies:** Phase 3 (OMS), Phase 4 (risk management)
**Priority:** P1

**Acceptance criteria:**
- Strategy trait is clean and minimal (no more than 5 required methods).
- Reference market-making strategy runs end-to-end on testnet with real order flow.
- Hot path benchmarks show < 1us per `on_book_update` with zero heap allocations.
- Strategy parameters are reloadable without restart.

### Tasks

- [ ] **5.1 Define Strategy trait**
  In `crates/strategy/src/traits.rs`:
  ```rust
  pub trait Strategy: Send + 'static {
      fn on_book_update(&mut self, ctx: &mut TradingContext, book: &OrderBook);
      fn on_trade(&mut self, ctx: &mut TradingContext, trade: &Trade);
      fn on_fill(&mut self, ctx: &mut TradingContext, fill: &Fill);
      fn on_timer(&mut self, ctx: &mut TradingContext, timestamp: Timestamp);
      fn on_params_update(&mut self, params: &StrategyParams);
  }
  ```
  `TradingContext` provides: `submit_order()`, `cancel_order()`, `get_position()`, `get_open_orders()`.
  The trait must NOT require async (strategies run on a dedicated thread, no awaiting).

- [ ] **5.2 Implement TradingContext**
  In `crates/strategy/src/context.rs`:
  - Wraps OMS order submission with pre-trade risk checks.
  - Provides read-only access to current positions and open orders.
  - Provides market data access (current book, recent trades).
  - Buffers order actions during a single strategy callback, flushes after return.
  - This prevents strategies from seeing their own orders before risk checks.

- [ ] **5.3 Implement strategy loader**
  In `crates/strategy/src/loader.rs`:
  - Compile-time strategy registration using inventory pattern or a simple match statement.
  - Config specifies which strategy to run: `strategy = "market_making"`.
  - Each strategy gets its own configuration section.
  - Support running multiple strategies on different symbols (future extension, design for it now).

- [ ] **5.4 Implement reference market-making strategy**
  In `crates/strategy/src/strategies/market_making.rs`:
  - Basic bid/ask quoting around mid-price with configurable spread.
  - Parameters: `spread_bps` (default 5), `order_size` (default 0.001 BTC), `num_levels` (default 1), `skew_factor`.
  - On book update: recalculate fair price, adjust quotes if they have moved > `reprice_threshold`.
  - On fill: adjust inventory, skew quotes toward reducing position.
  - Inventory management: widen spread as position grows, bias toward flat.
  - This is a reference implementation for testing the framework, not a production strategy.

- [ ] **5.5 Implement strategy metrics collection**
  In `crates/strategy/src/metrics.rs`:
  - Track per strategy: trade count, fill count, PnL (realized + unrealized), position.
  - Track latency: time from book update to order submission.
  - Track order statistics: fill ratio, cancel ratio, average fill size.
  - Export metrics via Prometheus format (expose via HTTP metrics endpoint).
  - Update metrics atomically on each event (use `AtomicU64` or `parking_lot::Mutex`).

- [ ] **5.6 Build strategy configuration system**
  In `crates/strategy/src/config.rs`:
  - Strategy parameters stored in Redis: `strategy:params:{strategy_name}`.
  - Hot-reload: poll every 5 seconds, call `on_params_update()` on change.
  - Validate parameter ranges before applying (e.g., spread_bps must be > 0).
  - Support A/B parameter sets (switch between "conservative" and "aggressive" profiles).

- [ ] **5.7 Ensure zero-allocation hot path**
  - Run `cargo bench` with `criterion` on `on_book_update` with a realistic book.
  - Profile with `dhat` or `heaptrack` to verify zero allocations in the hot loop.
  - Use stack-allocated buffers for order batches (e.g., `ArrayVec<Order, 8>`).
  - Avoid `String`, `Vec`, `Box` on the hot path. Pre-allocate everything at startup.
  - Target: `on_book_update` < 1us p99 on a modern x86_64 CPU.
  Document benchmark results in `crates/strategy/BENCHMARKS.md`.

---

## Phase 6: Backtesting Framework

**Dependencies:** Phase 2 (historical data), Phase 5 (strategy trait)
**Priority:** P1

**Acceptance criteria:**
- Same Rust strategy code runs in both production and backtest without modification.
- Backtest of market-making strategy on 2 weeks of data produces realistic PnL curve.
- Simulated fills account for queue position, fees, and configurable latency.
- Walk-forward optimization shows out-of-sample performance.

### Tasks

- [ ] **6.1 Implement event replay engine**
  In `crates/pybridge/src/replay.rs` (Rust core):
  - Read Parquet files using `arrow` / `parquet` crates.
  - Replay events in timestamp order (merge-sort across files/exchanges).
  - Feed events to strategy via same `Strategy` trait interface.
  - Configurable replay speed: real-time, 10x, max speed.
  - Support date range filtering and symbol filtering.

- [ ] **6.2 Implement simulated exchange**
  In `crates/pybridge/src/sim_exchange.rs`:
  - Matching engine: limit orders fill when market price crosses order price.
  - Model queue position: orders at same price level fill FIFO, estimate position in queue.
  - Apply fees: maker fee (e.g., -0.01% rebate on Binance), taker fee (e.g., 0.04%).
  - Model latency: configurable delay between order submission and exchange acknowledgment.
  - Model partial fills: large orders may fill across multiple price levels.
  - Model rejects: orders failing price/size filters return reject.
  - This is the most critical component for backtest accuracy. Get the fill model right.

- [ ] **6.3 Build PyO3 bridge**
  In `crates/pybridge/`:
  - Use `pyo3` and `maturin` to build a Python-importable module.
  - Expose: `run_backtest(strategy_name, params_dict, data_path, config) -> BacktestResult`.
  - `BacktestResult` contains: PnL series, trade list, order list, metrics dict.
  - Expose individual components for advanced users: `ReplayEngine`, `SimExchange`, `OrderBook`.
  - Build with `maturin develop` for development, `maturin build --release` for production.

- [ ] **6.4 Implement Python backtesting orchestrator**
  In `backtest/engine/orchestrator.py`:
  - High-level API for running backtests:
    ```python
    result = run_backtest(
        strategy="market_making",
        params={"spread_bps": 5, "order_size": 0.001},
        data="s3://bucket/BTCUSDT/2025-01/",
        start="2025-01-01", end="2025-01-14",
    )
    ```
  - Support parameter sweeps: `sweep_params(strategy, param_grid, data)`.
  - Parallel execution using `multiprocessing` or `concurrent.futures`.
  - Save results to Parquet files for analysis.

- [ ] **6.5 Implement analytics module**
  In `backtest/engine/analytics.py`:
  - Metrics: Sharpe ratio, Sortino ratio, max drawdown, profit factor, win rate, fill ratio.
  - Equity curve plotting (matplotlib/plotly).
  - Trade analysis: average trade duration, PnL distribution, time-of-day analysis.
  - Comparison tools: overlay multiple backtest runs on same chart.
  - Generate HTML report with all metrics and charts.

- [ ] **6.6 Implement walk-forward optimization**
  In `backtest/engine/walk_forward.py`:
  - Split data into in-sample (IS) and out-of-sample (OOS) windows.
  - Optimize parameters on IS window, test on OOS window.
  - Roll forward, repeat.
  - Report: IS performance vs OOS performance (detect overfitting).
  - Configurable window sizes: IS = 5 days, OOS = 2 days (default).

- [ ] **6.7 Implement latency sensitivity analysis**
  In `backtest/engine/latency_analysis.py`:
  - Run same strategy with added latency: +0ms, +1ms, +5ms, +10ms, +50ms.
  - Plot PnL vs latency curve.
  - This reveals how sensitive the strategy is to execution speed.
  - Strategies that degrade sharply with latency require co-location.
  - Output: report with latency sensitivity coefficient.

- [ ] **6.8 Create Jupyter notebook templates**
  In `backtest/notebooks/`:
  - `01_data_exploration.ipynb`: load Parquet, plot price, spread, volume, book depth.
  - `02_backtest_single.ipynb`: run single backtest, analyze results.
  - `03_parameter_sweep.ipynb`: sweep parameters, plot heatmap of Sharpe vs params.
  - `04_walk_forward.ipynb`: run walk-forward, plot IS vs OOS performance.
  - `05_latency_analysis.ipynb`: run latency sweep, plot sensitivity.
  Each notebook should be self-contained with clear markdown explanations.

- [ ] **6.9 Validate backtest results**
  - Compare backtest fill prices with actual historical trades at same timestamps.
  - Verify that fee calculations match exchange fee schedule.
  - Cross-check PnL calculation: manual calculation on a subset of trades.
  - Document known limitations of the backtest model.

---

## Phase 7: Infrastructure & Monitoring

**Dependencies:** Phases 1-6 (all core functionality should be working)
**Priority:** P1

**Acceptance criteria:**
- All critical metrics visible in real-time Grafana dashboards.
- Alerts fire within 5 seconds of threshold breach.
- One-command deployment to AWS Tokyo (ap-northeast-1).

### Tasks

- [ ] **7.1 Implement Go monitoring service**
  In `infra/monitoring/`:
  - Go service that reads metrics from the Rust trading core.
  - Communication: Unix domain socket or shared memory ring buffer.
  - Metrics collected: latency histograms, PnL, positions, order counts, connection status.
  - Expose Prometheus `/metrics` endpoint on port 9090.
  - Health check endpoint: `/health`.
  - Go module init: `go mod init github.com/cm-hft/monitoring`.

- [ ] **7.2 Set up Prometheus metrics exporter**
  - Configure Prometheus scrape targets: trading core, monitoring service, QuestDB, Redis.
  - Set scrape interval: 5 seconds for trading metrics, 15 seconds for infrastructure.
  - Configure retention: 30 days for high-resolution, 1 year for downsampled.
  - Add Prometheus to Docker Compose and Kubernetes manifests.

- [ ] **7.3 Create Grafana dashboards**
  In `infra/monitoring/dashboards/`:
  - **Trading dashboard**: PnL (real-time), positions, open orders, fill rate, spread captured.
  - **Latency dashboard**: wire-to-internal (p50/p99/p99.9), order-to-ack, internal processing time.
  - **Exchange dashboard**: connection status, message rate, error rate, rate limit usage.
  - **System dashboard**: CPU usage, memory, disk I/O, network, GC pauses (for Go service).
  - Export dashboards as JSON for version control.

- [ ] **7.4 Implement alerting rules**
  In `infra/monitoring/alerts/`:
  - **Critical (PagerDuty)**: PnL drawdown > threshold, circuit breaker triggered, connection lost > 30s.
  - **Warning (Slack)**: latency p99 > 10ms, rate limit usage > 70%, reconciliation mismatch, memory > 80%.
  - **Info (Slack)**: daily PnL summary, strategy parameter changes, deployment events.
  - Configure alert routing in `alertmanager.yml`.
  - Test every alert rule during paper trading phase.

- [ ] **7.5 Set up structured logging aggregation**
  - Trading core outputs JSON logs to stdout/file.
  - Ship logs to Grafana Loki (or ELK stack) via Promtail/Fluentd.
  - Configure log retention: 30 days.
  - Create Grafana explore queries for common investigations:
    - "Show all events for order X"
    - "Show all errors in the last hour"
    - "Show all circuit breaker events"
  - Ensure NO API keys, secrets, or signatures appear in logs (grep-test for this).

- [ ] **7.6 Write Terraform configs**
  In `infra/deploy/terraform/`:
  - AWS provider, region `ap-northeast-1` (Tokyo -- closest to exchange servers in that region).
  - EC2 instance: `c5.xlarge` or `c5n.xlarge` (network optimized) for trading core.
  - Security group: allow SSH (from specific IPs), HTTP (kill switch), outbound HTTPS (exchanges).
  - VPC with private subnet for internal services.
  - S3 bucket for Parquet data and backups.
  - IAM roles for EC2 instances (least privilege).
  - CloudWatch log group for system logs.
  - Use `terraform plan` before any apply.

- [ ] **7.7 Create Docker images**
  In `infra/deploy/docker/`:
  - `Dockerfile.trading`: Multi-stage build. Build Rust release binary, copy to minimal base (distroless or alpine).
  - `Dockerfile.monitoring`: Build Go binary, copy to scratch.
  - `Dockerfile.backtest`: Python image with maturin-built wheel installed.
  - All images tagged with git commit SHA and semantic version.
  - Scan images with Trivy for vulnerabilities.

- [ ] **7.8 Write deployment manifests**
  In `infra/deploy/k8s/` (or `infra/deploy/compose/`):
  - Option A (Kubernetes): Deployment, Service, ConfigMap, Secrets for each component.
  - Option B (Docker Compose): production-grade compose file with resource limits.
  - Include QuestDB, Redis, Prometheus, Grafana, Loki.
  - Configure resource limits: trading core gets CPU pinning (2 dedicated cores).
  - Configure restartPolicy: always for monitoring, never for trading (manual restart only).

- [ ] **7.9 Implement health check endpoints**
  For each service:
  - Trading core: `GET /health` returns `{ "status": "ok", "trading_enabled": true, "exchanges": {...} }`.
  - Monitoring service: `GET /health` returns `{ "status": "ok", "metrics_age_ms": 50 }`.
  - Health checks used by Docker/K8s for liveness and readiness probes.
  - If a health check fails, the orchestrator can take automated action (alert, restart).

---

## Phase 8: Paper Trading & Validation

**Dependencies:** All previous phases (0-7)
**Priority:** P0 -- mandatory before any real money trading.

**Acceptance criteria:**
- Paper trading runs stable for 7+ consecutive days without crashes or memory leaks.
- PnL within 20% of backtest prediction for the same time period.
- System handles exchange disconnects gracefully (reconnect < 5 seconds, no position corruption).
- Zero API keys or secrets found in any log output.

### Tasks

- [ ] **8.1 Implement paper trading mode**
  - Use the full production code path: market data, strategy, risk checks, OMS.
  - Replace exchange execution layer with a simulated executor:
    - Orders are "filled" at current market price + simulated latency.
    - Apply realistic fees.
    - Simulate partial fills for large orders.
  - Configurable via config: `mode = "paper"` vs `mode = "live"`.
  - Paper mode must exercise the SAME risk checks as live mode.

- [ ] **8.2 Run paper trading for 1+ week**
  Operational task:
  - Deploy paper trading setup on stable infrastructure.
  - Run market-making strategy on BTCUSDT (both exchanges or one to start).
  - Collect all metrics: PnL, positions, latency, order counts, error counts.
  - Monitor daily. Investigate any anomalies.
  - Document issues found and fixes applied.

- [ ] **8.3 Compare paper trading PnL with backtest predictions**
  - Run backtest on the same time period with same strategy parameters.
  - Compare: total PnL, Sharpe ratio, max drawdown, trade count.
  - Investigate differences > 20%. Common causes:
    - Fill model inaccuracy in backtest.
    - Latency differences.
    - Market regime change.
  - Document findings and adjust backtest model if needed.

- [ ] **8.4 Validate latency metrics**
  - Wire-to-internal: target < 100us p99.
  - Strategy processing: target < 1us p99.
  - Order-to-exchange: target < 5ms p99 (network dependent).
  - If targets not met, profile and optimize. Common fixes:
    - Reduce allocations on hot path.
    - Pin threads to specific CPUs.
    - Use `SO_BUSY_POLL` for network sockets.
  - Document actual latency numbers.

- [ ] **8.5 Stress test**
  - Simulate exchange disconnect: kill WebSocket mid-stream, verify reconnection and book recovery.
  - Simulate rapid price moves: replay flash crash data, verify risk checks fire correctly.
  - Simulate burst of fills: 100 fills in 1 second, verify OMS handles without dropping.
  - Simulate QuestDB outage: verify trading continues (recording degrades gracefully).
  - Simulate Redis outage: verify last-known risk parameters are used.

- [ ] **8.6 Security audit**
  - Grep all log output for API key patterns (regex for base64-like strings of key length).
  - Verify API keys are loaded ONLY from environment variables.
  - Verify HMAC signatures are not logged.
  - Verify kill switch endpoint is accessible.
  - Verify circuit breaker cannot be bypassed by any code path.
  - Review all network endpoints: no unnecessary ports open.
  - Document security model and threat surface.

- [ ] **8.7 Load test**
  - Replay market data at 10x speed. Verify system keeps up.
  - Replay at 100x speed. Identify breaking point.
  - Monitor memory usage over 24h at normal speed: must be stable (no leaks).
  - Use `valgrind` or `heaptrack` to verify no memory leaks in Rust code.
  - Monitor file descriptor count: must be stable.

---

## Phase 9: Production Launch

**Dependencies:** Phase 8 (paper trading must be validated)
**Priority:** P1

### Tasks

- [ ] **9.1 Deploy to AWS ap-northeast-1**
  - Apply Terraform to create infrastructure.
  - Deploy Docker images.
  - Configure secrets (API keys) via AWS Secrets Manager or env vars.
  - Verify all health checks pass.
  - Verify all Grafana dashboards show data.
  - Verify alerts fire (send test alert).

- [ ] **9.2 Configure CPU isolation and thread pinning**
  - Set kernel boot parameter `isolcpus=2,3` (reserve 2 cores for trading).
  - Pin market data thread to CPU 2, strategy thread to CPU 3.
  - Disable CPU frequency scaling: `cpupower frequency-set -g performance`.
  - Disable hyperthreading on trading cores if possible.
  - Verify with `htop` that trading threads stay on designated cores.

- [ ] **9.3 Apply network stack tuning**
  - Set `TCP_NODELAY` on all exchange connections (disable Nagle's algorithm).
  - Set `SO_BUSY_POLL` for lower latency on receive.
  - Tune kernel parameters:
    ```
    net.core.busy_read = 50
    net.core.busy_poll = 50
    net.ipv4.tcp_fastopen = 3
    ```
  - Verify with `ss -ti` that connections show expected settings.

- [ ] **9.4 Start with minimum position size**
  - Set risk parameters to absolute minimums:
    - Max position: 0.001 BTC (roughly $100 at $100k/BTC).
    - Max order: 0.001 BTC.
    - Daily loss limit: $10.
  - Verify these limits are enforced (intentionally submit oversized order, confirm rejection).
  - Monitor every trade manually for the first 24 hours.

- [ ] **9.5 Monitor for 48 hours**
  - Watch dashboards continuously (or have alerts for any anomaly).
  - Verify:
    - PnL is within expected range.
    - Latency is within targets.
    - No errors or warnings in logs.
    - Memory and CPU usage are stable.
    - Exchange connections are stable (no unexpected reconnects).
  - Document any issues and fixes.

- [ ] **9.6 Gradually increase position size**
  - Week 1: 0.001 BTC max position.
  - Week 2: 0.01 BTC max position (if Week 1 is clean).
  - Week 3: 0.05 BTC max position.
  - Week 4+: target position size (based on account size and risk appetite).
  - Each increase requires: review of previous period's metrics, explicit sign-off.
  - Never increase more than 5x in a single step.

- [ ] **9.7 Document operational runbook**
  Create `docs/runbook.md` covering:
  - How to start/stop the trading system.
  - How to trigger the kill switch.
  - How to reset the circuit breaker.
  - How to change strategy parameters.
  - How to deploy a new version (rolling update).
  - How to investigate a PnL discrepancy.
  - How to recover from exchange API key rotation.
  - Emergency contacts and escalation procedures.
  - Post-mortem template for incidents.

---

## Cross-Cutting Concerns

These apply across all phases and should be kept in mind throughout development.

### Performance
- Profile regularly with `cargo flamegraph`.
- Benchmark critical paths with `criterion`.
- No heap allocations on the hot path (market data -> strategy -> order submission).
- Target tick-to-trade latency: < 10us (internal processing, excluding network).

### Testing
- Unit test coverage > 80% for all crates.
- Integration tests against exchange testnet (run weekly in CI).
- Property-based tests for order book and state machine.
- Fuzz testing for message parsers (`cargo-fuzz`).

### Security
- API keys NEVER in source code, config files, or logs.
- All exchange connections over TLS.
- Kill switch always accessible.
- Regular dependency audit: `cargo audit`, `pip-audit`.

### Documentation
- Each crate has a `README.md` with purpose, usage, and examples.
- Public APIs have rustdoc comments.
- Architecture decision records (ADRs) for significant design choices.
- Keep this TASKS.md updated as work progresses.

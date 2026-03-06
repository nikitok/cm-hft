# New Symbol Analysis Pipeline Implementation Plan

Created: 2026-03-06
Status: COMPLETE
Approved: Yes
Iterations: 0
Worktree: Yes
Type: Feature

## Summary

**Goal:** CLI binary `cm-analyze-symbol` that connects to Binance WebSocket, collects orderbook and trade data for a given symbol over a configurable duration (default 5 min), computes microstructure metrics, and outputs a JSON report with a go/no-go verdict for HFT market-making suitability.

**Architecture:** New binary in `cm-trading` crate (alongside `cm-record`). Reuses existing `BinanceWsClient` for data collection, `OrderBook` for L2 reconstruction. New `SymbolAnalyzer` struct accumulates metrics in a streaming fashion (no full history storage). Verdict based on configurable thresholds.

**Tech Stack:** Rust, tokio, clap, serde_json. Reuses cm-market-data (BinanceWsClient, OrderBook) and cm-core (Price, Quantity, types).

## Scope

### In Scope
- CLI binary with `--symbol`, `--duration`, `--thresholds` flags
- Real-time data collection via Binance WebSocket (depth + trades)
- OrderBook reconstruction with snapshot + incremental updates
- Streaming metric computation: spread_bps (mean/min/max/p50/p95), bid/ask depth in USD (top-5 levels), trade rate (trades/sec), USD volume, mid-price volatility (rolling std dev), bid/ask imbalance ratio
- JSON output with all metrics + go/no-go verdict
- Unit tests for metric computation logic
- Integration test with recorded data

### Out of Scope
- Bybit support (future extension)
- HTTP endpoint integration
- Persistent storage of reports
- Historical/offline analysis mode
- Extended microstructure metrics (order flow correlation, book change velocity)

## Context for Implementer

> Write for an implementer who has never seen the codebase.

- **Patterns to follow:** `crates/trading/src/bin/record.rs` — the `cm-record` binary is the closest analog. It uses `clap::Parser` for CLI args, `BinanceWsClient` for WS connection, `mpsc` channels for book/trade events, `CancellationToken` for shutdown, and `tokio::select!` for the main loop.

- **Conventions:**
  - Binaries live in `crates/trading/src/bin/` and are declared as `[[bin]]` entries in `crates/trading/Cargo.toml`
  - All logging via `tracing` macros (`tracing::info!`, `tracing::debug!`)
  - Init logging with `cm_core::logging::init_tracing(true)` (JSON format)
  - Fixed-point arithmetic: `Price` and `Quantity` have `mantissa: i64` + `scale: u8`. Use `.to_f64()` for floating-point computations.
  - `Symbol::new("BTCUSDT")` wraps a string symbol name

- **Key files:**
  - `crates/market-data/src/binance/client.rs` — `BinanceWsClient`: `new()`, `connect()`, `subscribe()`, `fetch_snapshot()`, `read_message()`
  - `crates/market-data/src/orderbook.rs` — `OrderBook`: `apply_snapshot()`, `apply_update()`, `spread_bps()`, `best_bid()`, `best_ask()`, `mid_price()`, `bid_volume()`, `ask_volume()`, `bid_depth()`, `ask_depth()`
  - `crates/core/src/types/market_data.rs` — `BookUpdate`, `Trade`, `NormalizedTick`, `BookLevel`
  - `crates/core/src/types/price.rs` — `Price::to_f64()`, `Price::new(mantissa, scale)`
  - `crates/core/src/types/quantity.rs` — `Quantity::to_f64()`, `Quantity::is_zero()`
  - `crates/core/src/config.rs:116` — `ExchangeConfig` struct

- **Gotchas:**
  - `OrderBook` must receive a snapshot before incremental updates work (`OrderBookError::NotInitialized`). The binary must call `fetch_snapshot()` after connecting and apply it before processing deltas.
  - `BinanceWsClient::read_message()` returns `Option<BinanceMessage>` — `None` means non-data frame (subscription confirmation), not end of stream.
  - `BookUpdate` has both `is_snapshot: bool` and `bids`/`asks` as `Vec<(Price, Quantity)>`. The depth updates from WS are NOT snapshots — the snapshot comes from REST API.
  - `run_binance_ws` in record.rs hardcodes the WS URL — reuse same pattern.
  - **⚠️ update_id lost in BinanceMessage**: `BinanceWsClient::read_message()` converts `BinanceDepthUpdate` → `BookUpdate` via `From`, discarding `first_update_id`/`last_update_id`. But `OrderBook::apply_update(&mut self, update: &BookUpdate, update_id: u64)` requires a separate `update_id` parameter. **Solution:** maintain a monotonic counter initialized to `snapshot.last_update_id + 1`, increment on each depth event. This is correct for in-order WS messages.
  - **⚠️ Snapshot string conversion**: `BinanceDepthSnapshot` has `bids`/`asks` as `Vec<[String; 2]>` (raw strings from REST API). The `parse_level` helper in `binance::types` is private. **Solution:** make `parse_level` pub, OR duplicate the conversion inline: `Price::from(pair[0].parse::<f64>().unwrap())` / `Quantity::from(pair[1].parse::<f64>().unwrap())`. The snapshot's `last_update_id` is used as the `update_id` for `apply_snapshot`.
  - **⚠️ Do NOT copy ws_feeds.rs snapshot pattern**: `ws_feeds.rs` applies snapshots with hardcoded `update_id=1` and starts its monotonic counter from 1. This works for the hot path but diverges from Binance's synchronization protocol. `analyze_symbol.rs` MUST use `snapshot.last_update_id` as the update_id for `apply_snapshot` and initialize `update_counter = snapshot.last_update_id`. This preserves the Binance sync guarantee that buffered pre-snapshot events (with lower exchange sequence IDs) are correctly rejected via the monotonic counter.
  - **⚠️ Depth USD formula**: `OrderBook::bid_depth(n)` returns `Vec<BookLevel>` (price + quantity per level), NOT a single USD value. Depth in USD must be computed as: `book.bid_depth(5).iter().map(|l| l.price.to_f64() * l.quantity.to_f64()).sum::<f64>()`. Do NOT use `bid_volume() * mid_price` — that gives quantity sum × mid, not actual notional.

- **Domain context:** The analyzer evaluates whether a symbol has sufficient liquidity, tight spreads, and active trading for a market-making strategy to operate profitably. Key thresholds: spread should be narrow enough to quote inside, depth should be sufficient to handle position risk, trade frequency indicates demand.

## Progress Tracking

- [x] Task 1: SymbolAnalyzer metric accumulator
- [x] Task 2: Verdict and threshold logic
- [x] Task 3: CLI binary cm-analyze-symbol
- [x] Task 4: Unit tests for metrics and verdict
- [x] Task 5: Integration test with mock WS data

**Total Tasks:** 5 | **Completed:** 5 | **Remaining:** 0

## Implementation Tasks

### Task 1: SymbolAnalyzer metric accumulator

**Objective:** Create a `SymbolAnalyzer` struct that accumulates orderbook and trade metrics in a streaming fashion (constant memory, no full history). This is the core computation engine.

**Dependencies:** None

**Files:**
- Create: `crates/trading/src/analyzer.rs`
- Modify: `crates/trading/src/lib.rs` (add `pub mod analyzer;`)

**Key Decisions / Notes:**
- Streaming computation: maintain running sums, counts, min/max, and a small fixed-size window for percentile estimation (reservoir sampling or last-N for spread_bps p50/p95)
- For p50/p95 of spread_bps: keep a fixed circular buffer of last 1000 samples, sort on demand at report time. 1000 * 8 bytes = 8KB, negligible.
- Mid-price volatility: use Welford's online algorithm for running variance
- Struct fields:
  - `spread_samples: VecDeque<f64>` (sliding window, max 1000 entries — on_book_update: if len == 1000, pop_front() before push_back(). Ensures p50/p95 reflects the most recent 1000 samples, not the first 1000.)
  - `spread_sum: f64`, `spread_count: u64`, `spread_min: f64`, `spread_max: f64`
  - `trade_count: u64`, `trade_volume_usd: f64`
  - `bid_depth_usd_sum: f64`, `ask_depth_usd_sum: f64`, `depth_sample_count: u64`
  - `imbalance_sum: f64`, `imbalance_count: u64`
  - `mid_price_welford: (u64, f64, f64)` — (count, mean, M2) for online variance
  - `start_time: Instant`, `first_trade_ts: Option<u64>`, `last_trade_ts: Option<u64>`
- Methods:
  - `new() -> Self`
  - `on_book_update(&mut self, book: &OrderBook)` — called after each book update. Computes:
    - `spread_bps` from `book.spread_bps()` → push to circular buffer + update running sum/min/max
    - `bid_depth_usd` = `book.bid_depth(5).iter().map(|l| l.price.to_f64() * l.quantity.to_f64()).sum::<f64>()` (per-level notional, NOT volume × mid)
    - `ask_depth_usd` = same for `book.ask_depth(5)`
    - `imbalance` = `bid_depth_usd / (bid_depth_usd + ask_depth_usd)`
    - `mid_price` from `book.mid_price()` → feed into Welford's for volatility
  - `on_trade(&mut self, trade: &Trade)` — called on each trade. Accumulates trade_count, trade_volume_usd (`price.to_f64() * quantity.to_f64()`).
  - `report(&self, duration: Duration) -> AnalysisReport` — compute final metrics

**Definition of Done:**
- [ ] `SymbolAnalyzer` struct compiles with all fields
- [ ] `on_book_update` correctly accumulates spread, depth, imbalance metrics
- [ ] `on_trade` correctly accumulates trade count, volume
- [ ] `report()` returns a serializable `AnalysisReport` struct
- [ ] No diagnostics errors

**Verify:**
- `cargo check -p cm-trading`

---

### Task 2: Verdict and threshold logic

**Objective:** Define `AnalysisVerdict` enum and `Thresholds` struct. Implement the go/no-go decision based on configurable thresholds applied to computed metrics.

**Dependencies:** Task 1

**Files:**
- Modify: `crates/trading/src/analyzer.rs` (add verdict logic)

**Key Decisions / Notes:**
- `Thresholds` struct with defaults:
  - `max_spread_bps: f64` = 5.0 (mean spread must be below 5 bps)
  - `min_trade_rate: f64` = 1.0 (at least 1 trade/sec on average)
  - `min_depth_usd: f64` = 10_000.0 (mean top-5 depth per side >= $10K)
  - `max_volatility_bps: f64` = 50.0 (mid-price std dev in bps over the window)
- `AnalysisVerdict`: `Go`, `NoGo { reasons: Vec<String> }`, `Insufficient { reason: String }` (not enough data collected)
- `Thresholds` should be serializable/deserializable for CLI `--thresholds` JSON override
- `AnalysisReport` includes both raw metrics and verdict

**Definition of Done:**
- [ ] `Thresholds` struct with `Default` impl
- [ ] `AnalysisVerdict` enum is serializable
- [ ] `report()` method applies thresholds and returns verdict with reasons for NoGo
- [ ] Insufficient data case handled (< 10 book samples or < 5 trades → Insufficient)
- [ ] No diagnostics errors

**Verify:**
- `cargo check -p cm-trading`

---

### Task 3: CLI binary cm-analyze-symbol

**Objective:** Create the `cm-analyze-symbol` binary that wires together BinanceWsClient, OrderBook, and SymbolAnalyzer into a working pipeline with CLI argument parsing and JSON output.

**Dependencies:** Task 1, Task 2

**Files:**
- Create: `crates/trading/src/bin/analyze_symbol.rs`
- Modify: `crates/trading/Cargo.toml` (add `[[bin]]` entry)

**Key Decisions / Notes:**
- Follow `record.rs` pattern for WS connection and main loop
- CLI args via `clap::Parser`:
  - `--symbol BTCUSDT` (required)
  - `--duration 5m` (default "5m", reuse `parse_duration` pattern from record.rs)
  - `--thresholds '{"max_spread_bps": 3.0}'` (optional JSON string, merged with defaults)
- Flow:
  1. Parse args, init tracing
  2. Create `BinanceWsClient` with hardcoded production URLs (like record.rs)
  3. Connect WS, subscribe to symbol's depth + trade streams
  4. Fetch REST snapshot via `client.fetch_snapshot(&symbol)` → returns `BinanceDepthSnapshot`
  5. Convert snapshot levels: `snapshot.bids.iter().map(|p| (Price::from(p[0].parse::<f64>().unwrap()), Quantity::from(p[1].parse::<f64>().unwrap()))).collect()` (same for asks). Apply to `OrderBook::apply_snapshot(&bids, &asks, snapshot.last_update_id)`.
  6. Create `SymbolAnalyzer`. Init monotonic `update_counter = snapshot.last_update_id`.
  7. Main loop: `tokio::select!` on book_rx, trade_rx, shutdown timer, Ctrl+C
  8. On each book update: `update_counter += 1;` then call `book.apply_update(&update, update_counter)`. **Only call `analyzer.on_book_update(&book)` if apply_update returns `Ok(())`**. If it returns `Err(StaleUpdate)` or `Err(NotInitialized)`, log and skip — do NOT feed stale state to the analyzer. Pre-snapshot buffered events in the mpsc channel will trigger `StaleUpdate` — this is expected and safe.
  9. On each trade: `analyzer.on_trade(&trade)`
  10. On shutdown (timer OR channel close): `let report = analyzer.report(elapsed);` — use **actual elapsed time**, not configured duration, so that if the WS feed dies early the trade_rate denominator is correct.
  11. Print `serde_json::to_string_pretty(&report)` to stdout
  12. If either book_rx or trade_rx returns `None` (sender dropped = WS task died) before the timer fires, log a warning and break the main loop. If elapsed < 50% of configured duration, exit with code 2 (Insufficient).
  13. Exit with code 0 (Go), 1 (NoGo), 2 (Insufficient)
- Extract `parse_duration` to a shared utility or duplicate it (it's 12 lines)
- Tracing output goes to stderr (default), JSON report to stdout — they don't mix

**Definition of Done:**
- [ ] Binary compiles: `cargo build --bin cm-analyze-symbol`
- [ ] `--help` works and shows all flags
- [ ] WS connection, snapshot fetch, and main loop work for a real symbol
- [ ] JSON report printed to stdout on completion
- [ ] Exit code reflects verdict (0=Go, 1=NoGo, 2=Insufficient)
- [ ] Ctrl+C produces a partial report instead of crashing
- [ ] If WS feed dies before timer, binary logs warning, produces partial report with actual elapsed time, exits 2 if elapsed < 50% of configured duration
- [ ] apply_update errors are handled — on_book_update only called after Ok(())
- [ ] No diagnostics errors

**Verify:**
- `cargo build --bin cm-analyze-symbol`
- `cargo run --bin cm-analyze-symbol -- --symbol BTCUSDT --duration 30s 2>/dev/null | jq .`

---

### Task 4: Unit tests for metrics and verdict

**Objective:** Test the `SymbolAnalyzer` metric computation and verdict logic with controlled inputs. No network access needed.

**Dependencies:** Task 1, Task 2

**Files:**
- Modify: `crates/trading/src/analyzer.rs` (add `#[cfg(test)] mod tests`)

**Key Decisions / Notes:**
- Test cases:
  1. Empty analyzer → report returns `Insufficient`
  2. Feed 100 book updates with known spread → verify mean/min/max spread_bps
  3. Feed trades → verify trade_count, volume_usd, trade_rate calculation
  4. Verify depth USD calculation: create a book with 5 bid levels at different prices and quantities; compute `expected_depth_usd = sum(price_i * qty_i for i in 1..5)`; assert `analyzer.mean_bid_depth_usd == expected_depth_usd`. Explicitly verifies per-level notional formula (NOT volume × mid_price).
  5. Verify mid-price volatility with constant mid → volatility ≈ 0
  6. Verify mid-price volatility with alternating mid → non-zero volatility
  7. Go verdict: good metrics below all thresholds
  8. NoGo verdict: wide spread → verdict has "spread" in reasons
  9. NoGo verdict: low trade rate → verdict has "trade_rate" in reasons
  10. Custom thresholds override defaults
- Use `OrderBook` directly (not mocked) — it's a pure data structure
- Create helper `fn make_book_with_spread(bid: f64, ask: f64) -> OrderBook`

**Definition of Done:**
- [ ] All 10+ test cases pass
- [ ] Tests cover all metric fields in the report
- [ ] Tests cover all three verdict variants (Go, NoGo, Insufficient)
- [ ] Tests verify threshold customization
- [ ] No diagnostics errors

**Verify:**
- `cargo test --lib -p cm-trading -- analyzer`

---

### Task 5: Integration test with mock WS data

**Objective:** Verify the full pipeline works end-to-end by replaying recorded data through the analyzer (without live WS connection).

**Dependencies:** Task 1, Task 2, Task 3

**Files:**
- Create: `crates/trading/tests/analyze_symbol_test.rs`

**Key Decisions / Notes:**
- Don't test live WS — test the analyzer pipeline with synthetic data
- Create a sequence of `BookUpdate` and `Trade` events that simulate 30 seconds of market data
- Feed them through `SymbolAnalyzer` and verify the report structure
- Test that JSON serialization/deserialization roundtrips correctly
- Test edge cases: only books (no trades), only trades (no books), very short duration

**Definition of Done:**
- [ ] Integration test passes with synthetic data
- [ ] Report JSON roundtrip works
- [ ] Edge cases handled without panics
- [ ] No diagnostics errors

**Verify:**
- `cargo test --test analyze_symbol_test -p cm-trading`

## Testing Strategy

- **Unit tests (Task 4):** Pure computation tests for `SymbolAnalyzer`. No I/O, no network. Tests metric accuracy and verdict correctness.
- **Integration test (Task 5):** Synthetic data pipeline test. Verifies the analyzer processes a realistic sequence of events and produces valid output.
- **Manual verification:** Run `cm-analyze-symbol --symbol BTCUSDT --duration 30s` against live Binance and verify the JSON output is reasonable.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Binance WS rate limit during snapshot fetch | Low | Medium | Same pattern as cm-record which works in production; 5s retry on failure |
| Spread percentile computation on large windows | Low | Low | Capped circular buffer at 1000 samples; sort only at report time |
| Insufficient data in very short durations (< 30s) | Medium | Low | Return `Insufficient` verdict with reason when sample count too low |
| `OrderBook::apply_update` before snapshot | Medium | High | Fetch snapshot immediately after subscribe, skip updates until snapshot applied |

## Goal Verification

### Truths
1. `cargo build --bin cm-analyze-symbol` succeeds without errors
2. Running with `--symbol BTCUSDT --duration 30s` produces valid JSON with all metric fields
3. The `verdict` field in the JSON is one of "Go", "NoGo", or "Insufficient"
4. NoGo verdict includes human-readable `reasons` array explaining which thresholds failed
5. Custom thresholds via `--thresholds` JSON override the defaults
6. Ctrl+C during collection produces a partial report (not a crash)
7. All unit and integration tests pass: `cargo test --lib -p cm-trading -- analyzer` and `cargo test --test analyze_symbol_test -p cm-trading`

### Artifacts
- `crates/trading/src/analyzer.rs` — SymbolAnalyzer, AnalysisReport, Thresholds, AnalysisVerdict
- `crates/trading/src/bin/analyze_symbol.rs` — CLI binary
- `crates/trading/Cargo.toml` — `[[bin]]` entry for cm-analyze-symbol
- `crates/trading/src/lib.rs` — `pub mod analyzer`
- `crates/trading/tests/analyze_symbol_test.rs` — integration test

### Key Links
- `BinanceWsClient` (market-data) → `analyze_symbol.rs` binary (data collection)
- `OrderBook` (market-data) → `SymbolAnalyzer` (metric computation from book state)
- `SymbolAnalyzer::report()` → JSON stdout (final output)
- `Thresholds` → `AnalysisVerdict` (decision logic)

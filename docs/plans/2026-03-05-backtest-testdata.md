# Backtest Strategies on Recorded Test Data

Created: 2026-03-05
Status: VERIFIED
Approved: Yes
Iterations: 0
Worktree: Yes
Type: Feature

## Summary

**Goal:** Extend the existing replay backtest infrastructure (`crates/trading/tests/replay_harness.rs`, `replay_strategy.rs`, `replay_bench.rs`) to support Binance market data alongside Bybit, enabling all registered strategies to be backtested on all available testdata files.

**Architecture:** The existing harness is fully functional but hardcodes `bybit_` filename prefix and `Exchange::Bybit` throughout. Changes make file discovery exchange-agnostic (detecting exchange from filename), pass the correct `Exchange` enum variant to `ReplayTestHarness`, and add Binance-specific replay tests.

**Tech Stack:** Rust (test infrastructure only — no new crates or binaries)

## Scope

### In Scope
- Make `data_path()`, `find_series_files()`, `discover_symbols()` exchange-agnostic
- Pass detected `Exchange` to `ReplayTestHarness::new()` / `with_config()`
- Update `run_and_report_with_config()` and `run_strategy_on_events()` to accept `Exchange`
- Add Binance replay tests in `replay_strategy.rs`
- Update benchmark print headers (remove "Bybit" hardcoding)

### Out of Scope
- New CLI binary (`cm-backtest`)
- Python integration / PyO3 bridge changes
- New metrics beyond what `ReplayResult` already provides
- Recording new test data

## Context for Implementer

> Write for an implementer who has never seen the codebase.

- **Patterns to follow:** `replay_strategy.rs:17-31` — `data_path()` helper showing file discovery pattern. `replay_harness.rs:238-253` — `ReplayTestHarness::new()` constructor taking `Exchange` and `symbol`.
- **Conventions:** Tests skip gracefully if data files are missing (`eprintln!("SKIP: ...")` + `return`). Exchange type is `cm_core::types::Exchange` with variants `Binance` and `Bybit`.
- **Key files:**
  - `crates/trading/tests/replay_harness.rs` (810 lines) — core replay engine, SimExchange, event loading
  - `crates/trading/tests/replay_strategy.rs` (261 lines) — strategy correctness tests
  - `crates/trading/tests/replay_bench.rs` (1563 lines) — benchmarks, param sweep, optimizer
- **Gotchas:**
  - Binance data uses full-size UNIX nanosecond timestamps (`1772279055914000000`), while Bybit data uses smaller relative timestamps. The harness already handles this by using monotonic sequence numbers.
  - `data_path()` exists in BOTH `replay_strategy.rs` and `replay_bench.rs` — both must be updated.
  - `run_and_report_with_config()` and `run_strategy_on_events()` in bench currently hardcode `Exchange::Bybit`.
  - `find_series_files()` in `replay_harness.rs` hardcodes `bybit_` prefix.
  - `discover_symbols()` in `replay_bench.rs` uses a regex matching only `bybit_`.
  - Binance testdata files are much smaller (507KB vs 28-33MB) — shorter recording duration.
- **Domain context:** The `Exchange` enum affects `OrderBook::new(exchange, symbol)` construction, which may influence internal behavior. Both `Binance` and `Bybit` are supported variants.

## Progress Tracking

- [x] Task 1: Make replay_harness exchange-agnostic
- [x] Task 2: Add Binance support to replay_strategy tests
- [x] Task 3: Add Binance support to replay_bench
**Total Tasks:** 3 | **Completed:** 3 | **Remaining:** 0

## Implementation Tasks

### Task 1: Make replay_harness exchange-agnostic

**Objective:** Update `find_series_files()` to discover files from any exchange, not just Bybit.
**Dependencies:** None

**Files:**
- Modify: `crates/trading/tests/replay_harness.rs`

**Key Decisions / Notes:**
- `find_series_files()` currently takes `dir` and `symbol`, hardcodes `bybit_` prefix.
- Add an `exchange` parameter (or make it discover all exchanges). Better approach: return `(Exchange, Vec<String>)` tuples or accept an exchange filter string.
- Simplest: change `find_series_files(dir, symbol)` → `find_series_files(dir, exchange_prefix, symbol)` where `exchange_prefix` is `"bybit"` or `"binance"`.
- Or: add `find_all_series_files(dir, symbol) -> Vec<(String, Vec<String>)>` that returns `(exchange_name, files)` pairs.
- Preferred: add a helper `fn detect_exchange(filename: &str) -> Option<Exchange>` and update `find_series_files` to accept an exchange prefix string parameter.

**Definition of Done:**
- [ ] `find_series_files()` can discover Binance timestamped files
- [ ] Existing Bybit discovery still works (no regression)
- [ ] All existing tests pass: `cargo test --lib -p cm-trading`

**Verify:**
- `cargo test --test replay_strategy -p cm-trading -- --nocapture 2>&1 | head -30`

### Task 2: Add Binance support to replay_strategy tests

**Objective:** Update `data_path()` in `replay_strategy.rs` to find Binance data files, add Binance replay tests.
**Dependencies:** Task 1

**Files:**
- Modify: `crates/trading/tests/replay_strategy.rs`

**Key Decisions / Notes:**
- Current `data_path()` only looks for `bybit_{symbol}_{dur}.jsonl.gz`. Need to also search `binance_{symbol}_{dur}.jsonl.gz`.
- Return type should include the exchange: `Option<(String, Exchange)>` instead of `Option<String>`.
- Or: add a separate `data_path_for_exchange(exchange: &str, symbol: &str)` helper.
- Add test functions: `test_mm_strategy_binance_btc`, `test_mm_strategy_binance_eth`, `test_adaptive_mm_binance_btc`, `test_adaptive_mm_binance_eth`.
- Tests should use `Exchange::Binance` when constructing `ReplayTestHarness`.
- Binance data is shorter (~1min vs 30min) so thresholds may need to be more lenient.

**Definition of Done:**
- [ ] `data_path()` can find both Binance and Bybit data files
- [ ] New Binance tests exist and pass (or skip gracefully if no data)
- [ ] Existing Bybit tests still pass unchanged
- [ ] All strategies generate orders on Binance data without panicking

**Verify:**
- `cargo test --test replay_strategy -p cm-trading -- --nocapture 2>&1`

### Task 3: Add Binance support to replay_bench

**Objective:** Update benchmarks to run on both Binance and Bybit data.
**Dependencies:** Task 1

**Files:**
- Modify: `crates/trading/tests/replay_bench.rs`

**Key Decisions / Notes:**
- `data_path()` (line 13-27) — same change as Task 2 (search both prefixes).
- `discover_symbols()` (line 620-636) — regex only matches `bybit_`. Need to also match `binance_`. Return should include exchange info.
- `run_and_report_with_config()` (line 33-127) — hardcodes `Exchange::Bybit` at line 52. Accept `Exchange` param.
- `run_strategy_on_events()` (line 655-670) — hardcodes `Exchange::Bybit` at line 665. Accept `Exchange` param.
- `load_series()` (line 606-616) — calls `find_series_files()` with Bybit prefix.
- `bench_all_strategies()` — calls `run_and_report()` which uses `Exchange::Bybit`.
- `bench_improvement_stages()` — `Exchange::Bybit` at line 323.
- `bench_diagnostic()` — `Exchange::Bybit` at lines 437-438.
- `bench_param_sweep()` — `Exchange::Bybit` at line 532.
- `bench_series()` — uses `discover_symbols()` and `run_strategy_on_events()`.
- `bench_optimizer()` — uses `discover_symbols()` and `run_strategy_on_events()`.
- `bench_sim_realism()` — uses `discover_symbols()`, `data_path()`, `run_strategy_on_events()`.
- Print headers like "real Bybit data" should become "real market data" or include exchange name.
- Key approach: `discover_symbols()` returns `Vec<(Exchange, String)>` or a struct. Then all bench functions iterate over exchange+symbol pairs.

**Definition of Done:**
- [ ] `discover_symbols()` discovers both Binance and Bybit symbols
- [ ] All bench functions run on both exchanges' data
- [ ] Exchange name displayed in benchmark output
- [ ] No hardcoded `Exchange::Bybit` remains (all derived from data file)
- [ ] Existing bench output format preserved (just with exchange info added)

**Verify:**
- `cargo test --test replay_bench -p cm-trading -- bench_all_strategies --nocapture 2>&1 | head -50`

## Testing Strategy

- **Unit:** No new unit tests needed — changes are to test infrastructure itself.
- **Integration:** Run all existing replay tests + new Binance tests: `cargo test --test replay_strategy --test replay_bench -p cm-trading -- --nocapture`
- **Manual verification:** Run `bench_all_strategies` and confirm output includes both Binance and Bybit results.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Binance data too short for meaningful benchmark | Medium | Low | Tests use lenient thresholds; benchmark shows data as-is |
| OrderBook behavior differs by exchange | Low | Medium | `Exchange` enum already passed to `OrderBook::new()` — behavior handled internally |
| Breaking existing test expectations | Low | High | Run full test suite before and after; existing Bybit paths unchanged |

## Goal Verification

### Truths
1. Running `cargo test --test replay_strategy -p cm-trading` executes tests on both Bybit AND Binance data
2. Running `cargo test --test replay_bench -p cm-trading -- bench_all_strategies` shows benchmark results for both exchanges
3. All strategies (market_making, adaptive_mm) produce orders and fills on Binance data without panicking
4. No hardcoded `Exchange::Bybit` or `bybit_` prefix remains in the test infrastructure (except in string literals for display)
5. Existing Bybit test results are identical before and after changes

### Artifacts
- `crates/trading/tests/replay_harness.rs` — exchange-agnostic file discovery
- `crates/trading/tests/replay_strategy.rs` — Binance + Bybit strategy tests
- `crates/trading/tests/replay_bench.rs` — multi-exchange benchmarks

### Key Links
- `replay_harness.rs:find_series_files()` → `replay_bench.rs:load_series()` → `bench_series()`, `bench_optimizer()`, `bench_sim_realism()`
- `replay_harness.rs:load_events()` → `replay_strategy.rs` tests + `replay_bench.rs` benchmarks
- `Exchange` enum (`cm_core::types::order.rs:8`) → `OrderBook::new()` → `ReplayTestHarness`

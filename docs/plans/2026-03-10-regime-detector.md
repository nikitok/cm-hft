# Market Regime Detector for adaptive_mm Implementation Plan

Created: 2026-03-10
Status: VERIFIED
Approved: Yes
Iterations: 0
Worktree: No
Type: Feature

## Summary

**Goal:** Add a `RegimeDetector` signal component that classifies market regime (flat vs trending) based on mid-price drift rate, and integrate it into adaptive_mm as a spread multiplier to reduce losses during trending markets while preserving profitability in flat markets.

**Architecture:** New `RegimeDetector` struct in `signals.rs` tracks mid-price drift over a configurable rolling time window. Outputs a `regime_spread_mult` value (1.0 in flat, up to `regime_max_mult` in trending). Integrated into adaptive_mm's `on_book_update` as a multiplier on `half_spread`, applied after the existing VPIN multiplier. Disabled by default (`regime_window_secs=0`).

**Tech Stack:** Rust, no new dependencies.

## Scope

### In Scope
- `RegimeDetector` signal component with time-based rolling window
- Mid-price drift rate computation (bps/hr)
- Spread multiplier output with hysteresis
- Integration into adaptive_mm `on_book_update`
- Runtime parameter updates via `on_params_update`
- Unit tests for RegimeDetector
- Integration tests for adaptive_mm regime behavior
- Benchmark sweep of regime parameters on ETHUSDT data

### Out of Scope
- Volatility-based regime classification (deferred — drift-only first)
- Trade rate signal (deferred)
- Stop-quoting mode (user chose spread multiplier instead)
- Dynamic `max_position` adjustment per regime (follow-up)
- K8s configmap updates (no prod deployment yet — sweep first)

## Context for Implementer

> Write for an implementer who has never seen the codebase.

- **Patterns to follow:** The `TradeImbalanceTracker` in `signals.rs:119-201` is the closest pattern — a `VecDeque`-based rolling window with `update()` + query methods. The `RegimeDetector` follows the same shape but uses time-based eviction (nanosecond timestamps) instead of count-based.

- **Conventions:**
  - Signal components are pure structs in `crates/strategy/src/strategies/signals.rs`
  - Each has `new()`, `update()`, and one or more query methods
  - Parameters loaded in `adaptive_mm.rs` `from_params()` with `DEFAULT_*` constants
  - All params support runtime update via `on_params_update()`
  - All new features disabled by default (window=0 or factor=0)

- **Key files:**
  - `crates/strategy/src/strategies/signals.rs` — signal components (Ema, VolatilityTracker, TradeFlowSignal, TradeImbalanceTracker, VpinTracker)
  - `crates/strategy/src/strategies/adaptive_mm.rs` — strategy with ~730 lines, integrates all signals
  - `crates/trading/tests/replay_bench.rs` — benchmark tests with sweep patterns (see `bench_vpin_factor_sweep` at line 1949)

- **Gotchas:**
  - `on_book_update` is the hot path — RegimeDetector must be O(1) amortized per call, no allocations
  - The `half_spread` computation is at line 504: `(as_spread / 2.0).max(min_spread_floor) * vpin_multiplier` — regime multiplier goes here
  - **Timestamps:** `OrderBook` has NO `timestamp()` method. Use `ctx.timestamp.as_nanos()` from `TradingContext` (field at `context.rs:54`). `on_trade` uses `trade.timestamp.as_nanos()` but regime feeds from `on_book_update` via `ctx.timestamp`.
  - `on_params_update` for regime must reconstruct the entire `RegimeDetector` (same pattern as `imbalance_tracker` at line 744) — changing window_secs requires clearing stale VecDeque entries
  - The strategy is single-symbol per instance — no cross-symbol state needed

- **Domain context:**
  - **Drift rate** = `(current_mid - oldest_mid_in_window) / oldest_mid * 10_000` → bps. Converted to bps/hr by dividing by window duration in hours.
  - **Hysteresis:** regime should not flip-flop on every tick. Use two thresholds: `drift_enter_bps_hr` (to enter trending) and `drift_exit_bps_hr` (to exit trending, lower than enter). Classic Schmitt trigger pattern.
  - **Spread multiplier:** In flat regime, `regime_spread_mult = 1.0` (no change). In trending, ramp linearly: `regime_spread_mult = 1.0 + (regime_max_mult - 1.0) * (abs_drift - drift_exit) / (drift_enter - drift_exit)` clamped to `[1.0, regime_max_mult]`. Only one parameter (`regime_max_mult`) controls the ramp ceiling. Guard: if `drift_enter <= drift_exit`, treat as binary (1.0 or max_mult, no ramp — avoids divide-by-zero).

## Progress Tracking

- [x] Task 1: RegimeDetector signal component
- [x] Task 2: Integrate RegimeDetector into adaptive_mm
- [x] Task 3: Benchmark sweep — regime parameter optimization

**Total Tasks:** 3 | **Completed:** 3 | **Remaining:** 0

## Implementation Tasks

### Task 1: RegimeDetector Signal Component

**Objective:** Create a `RegimeDetector` struct in `signals.rs` that tracks mid-price drift over a time-based rolling window and outputs a spread multiplier.

**Dependencies:** None

**Files:**
- Modify: `crates/strategy/src/strategies/signals.rs`

**Key Decisions / Notes:**

- Use `VecDeque<(f64, u64)>` for `(mid_price, timestamp_ns)` pairs
- Time-based eviction: on each `update()`, pop entries older than `window_ns` from front
- Drift = `(newest_mid - oldest_mid) / oldest_mid * 10_000` → bps
- Convert to bps/hr: `drift_bps * 3_600_000_000_000 / elapsed_ns`
- Hysteresis via Schmitt trigger: two thresholds `drift_enter_bps_hr` and `drift_exit_bps_hr`
- Output: `spread_multiplier()` returns `f64` in range `[1.0, max_mult]`
- When `window_secs == 0`, the detector is disabled: `update()` is a no-op, `spread_multiplier()` returns 1.0
- Follow `TradeImbalanceTracker` pattern: `new(window_secs, drift_enter_bps_hr, drift_exit_bps_hr, max_mult)` → `update(mid, timestamp_ns)` → `spread_multiplier()`
- Additional query methods: `drift_bps_hr()` → current drift, `is_trending()` → bool

**Definition of Done:**
- [ ] `RegimeDetector::new(0, ..)` creates disabled detector; `spread_multiplier()` returns 1.0
- [ ] `RegimeDetector` with flat price series returns `spread_multiplier()` == 1.0
- [ ] `RegimeDetector` with strong uptrend (>drift_enter) returns `spread_multiplier()` > 1.0
- [ ] `RegimeDetector` with strong downtrend returns same multiplier as uptrend (uses abs drift)
- [ ] Hysteresis works: after entering trending, `is_trending()` stays true until drift drops below `drift_exit`
- [ ] Time-based eviction: old entries are dropped, drift recalculates correctly
- [ ] When `drift_enter == drift_exit`, `spread_multiplier()` returns either 1.0 or max_mult (no NaN/panic)
- [ ] RegimeDetector with exactly 1 entry returns `spread_multiplier()` == 1.0 and `drift_bps_hr()` == 0.0
- [ ] All tests pass, no clippy warnings

**Verify:**
- `cargo test --lib -p cm-strategy -- signals::tests::test_regime`

---

### Task 2: Integrate RegimeDetector into adaptive_mm

**Objective:** Wire `RegimeDetector` into `AdaptiveMarketMaker` — feed it mid prices in `on_book_update`, apply spread multiplier, support params.

**Dependencies:** Task 1

**Files:**
- Modify: `crates/strategy/src/strategies/adaptive_mm.rs`

**Key Decisions / Notes:**

- Add `regime_detector: RegimeDetector` field to `AdaptiveMarketMaker` struct (after line 87)
- Add default constants:
  - `DEFAULT_REGIME_WINDOW_SECS: u64 = 0` (disabled by default)
  - `DEFAULT_REGIME_DRIFT_ENTER_BPS_HR: f64 = 100.0`
  - `DEFAULT_REGIME_DRIFT_EXIT_BPS_HR: f64 = 50.0`
  - `DEFAULT_REGIME_MAX_MULT: f64 = 3.0`
- In `from_params()`: read `regime_window_secs`, `regime_drift_enter_bps_hr`, `regime_drift_exit_bps_hr`, `regime_max_mult`
- In `on_book_update()`:
  - After line 334 (`self.vol_tracker.update(mid)`), add: `self.regime_detector.update(mid, ctx.timestamp.as_nanos());`
  - At line 504, change: `half_spread = (as_spread / 2.0).max(min_spread_floor) * vpin_multiplier;`
    to: `let regime_mult = self.regime_detector.spread_multiplier();`
    then: `half_spread = (as_spread / 2.0).max(min_spread_floor) * (vpin_multiplier * regime_mult).min(max_combined_mult);`
  - Heuristic fallback path (line 479): `half_spread = (mid * self.base_spread_bps / 10_000.0 / 2.0).max(min_spread_floor) * self.regime_detector.spread_multiplier();` (VPIN is not applied on this path today — only add regime)
- Add `DEFAULT_REGIME_MAX_COMBINED_MULT: f64 = 5.0` — caps the product of `vpin_multiplier * regime_mult` to prevent unrealistically wide spreads
- In `on_params_update()`: reconstruct `self.regime_detector = RegimeDetector::new(...)` when any regime param changes (same pattern as `imbalance_tracker` at line 744 — clearing stale VecDeque entries)
- Add import of `RegimeDetector` alongside existing signal imports (line 12)
- Integration tests: create strategy with `regime_window_secs=300`, feed flat prices → spread normal, then feed trending prices → spread widens

**Definition of Done:**
- [ ] adaptive_mm with `regime_window_secs=0` (default) behaves identically to before (backward compat)
- [ ] adaptive_mm with `regime_window_secs=300, regime_max_mult=3.0` produces wider spreads during trending price series
- [ ] adaptive_mm with `regime_window_secs=300` produces normal spreads during flat price series
- [ ] `on_params_update` correctly updates regime parameters at runtime (reconstructs RegimeDetector, no stale entries)
- [ ] Combined `vpin_multiplier * regime_spread_mult` is capped at `max_combined_mult` (default 5.0)
- [ ] Heuristic fallback path (vol < 0.01) also applies regime multiplier
- [ ] All existing tests still pass (no regressions)
- [ ] No clippy warnings

**Verify:**
- `cargo test --lib -p cm-strategy`
- `cargo test --test replay_strategy -p cm-trading`

---

### Task 3: Benchmark Sweep — Regime Parameter Optimization

**Objective:** Add `bench_regime_sweep` test that sweeps regime parameters on ETHUSDT data and ranks by Calmar ratio.

**Dependencies:** Task 2

**Files:**
- Modify: `crates/trading/tests/replay_bench.rs`

**Key Decisions / Notes:**

- Follow the `bench_vpin_factor_sweep` pattern at line 1949
- Sweep parameters:
  - `regime_window_secs`: [60, 300, 900] (1min, 5min, 15min)
  - `regime_drift_enter_bps_hr`: [50, 100, 200]
  - `regime_max_mult`: [2.0, 3.0, 5.0]
  - Fix `drift_exit = drift_enter / 2` (standard hysteresis ratio)
- Base config: `conserv` (γ=1.0, κ=1.5, τ=2.0, vpin=2.0, decay=2.0, boost=0.5, max_pos=0.05, reprice_bps=12)
- Total configs: 3 × 3 × 3 = 27 per exchange
- Output table columns: Rank, Window, Enter, MaxMult, Fills, Realized$, M2M$, Fees$, Max DD$, Calmar, $/fill
- Sort by Calmar descending
- Include baseline row (regime disabled, `regime_window_secs=0`) for comparison
- Skip gracefully when no ETHUSDT data present

**Definition of Done:**
- [ ] `bench_regime_sweep` test exists and compiles
- [ ] Test skips with SKIP message when no ETHUSDT data
- [ ] With data present, outputs ranked table for each exchange
- [ ] Baseline (disabled) row included for comparison
- [ ] Test runs in release mode without errors

**Verify:**
- `cargo test --release -p cm-trading --test replay_bench bench_regime_sweep -- --nocapture`

## Assumptions

- `ctx.timestamp` (from `TradingContext`) provides `Timestamp` with `.as_nanos() -> u64` — confirmed at `context.rs:54`. `OrderBook` has NO timestamp method. Tasks 1, 2 depend on this.
- The `on_book_update` callback receives book updates frequently enough (sub-second) to populate the regime window meaningfully — supported by bench data showing 7.4M book updates over 10 days for ETHUSDT/Binance. Tasks 1, 2 depend on this.
- `half_spread` multiplication is the correct integration point — the spread already has VPIN multiplier applied there (line 504), so stacking regime multiplier is consistent. Task 2 depends on this.
- The `conserv` config (γ=1.0) is the right baseline for regime sweep — it showed best risk-adjusted performance in the bench_series run. Task 3 depends on this.

## Testing Strategy

- **Unit tests** (Task 1): RegimeDetector in isolation — disabled, flat, trending up/down, hysteresis, eviction. All in `signals.rs` test module.
- **Integration tests** (Task 2): adaptive_mm with regime enabled vs disabled, verifying spread width changes. In `replay_strategy.rs` or inline in adaptive_mm.rs.
- **Benchmark tests** (Task 3): Parameter sweep on real ETHUSDT data. In `replay_bench.rs`. Manual — requires data files.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Regime detector adds latency to hot path | Low | High | Time-based eviction is O(amortized 1) — only pops expired entries. VecDeque operations are cache-friendly. Measure in bench. |
| Hysteresis thresholds hard to tune | Medium | Medium | Start with simple ratio (exit = enter/2). Sweep validates in Task 3. |
| Regime multiplier makes spread too wide, killing fill rate | Medium | Medium | Clamp to `regime_max_mult`. Sweep tests multiple values. Disabled by default. |
| Book timestamps not monotonic or have large gaps | Low | Low | RegimeDetector handles gracefully: if window has < 2 entries, return 1.0. |

## Pre-Mortem

*Assume this plan failed. Most likely internal reasons:*

1. **Drift signal is too noisy on short windows** (Task 1) → Trigger: unit tests show spread_multiplier oscillating between 1.0 and max_mult on synthetic data with moderate trend. Fix: add EMA smoothing on drift_bps_hr before threshold comparison, or increase minimum window population before activating.

2. **Regime multiplier stacks badly with VPIN multiplier producing unrealistically wide spreads** (Task 2) → Trigger: bench sweep shows 0 fills for all regime-enabled configs. Fix: apply `min(regime_mult * vpin_mult, absolute_max_combined_mult)` cap, or make regime and VPIN compete (take max, not multiply).

3. **10-day ETHUSDT data is predominantly trending, so regime detector is always ON and sweep shows no differentiation** (Task 3) → Trigger: all sweep configs produce similar results. Fix: the data actually has flat periods between trends — check if window_secs=60 captures the flat segments. If not, the detector needs per-segment analysis (not in scope).

## Goal Verification

### Truths

1. RegimeDetector correctly classifies flat vs trending market conditions based on mid-price drift rate
2. Spread widens during trending periods and returns to normal during flat periods (via spread_multiplier)
3. Hysteresis prevents rapid regime flip-flopping on noisy price data
4. Feature is disabled by default — `regime_window_secs=0` produces identical behavior to current code
5. Benchmark sweep produces ranked results showing whether regime detection improves Calmar ratio vs baseline

### Artifacts

- `crates/strategy/src/strategies/signals.rs` — RegimeDetector struct with unit tests
- `crates/strategy/src/strategies/adaptive_mm.rs` — integration with spread computation
- `crates/trading/tests/replay_bench.rs` — bench_regime_sweep test

### Key Links

1. `RegimeDetector.update()` ← called from `adaptive_mm.on_book_update()` with mid price and timestamp
2. `RegimeDetector.spread_multiplier()` → applied to `half_spread` in adaptive_mm spread computation
3. `regime_*` params → loaded in `from_params()`, updated in `on_params_update()`
4. `bench_regime_sweep` → uses same data loading and strategy execution pattern as `bench_vpin_factor_sweep`

## Open Questions

None.

### Deferred Ideas

- **Volatility regime dimension:** Add high-low range signal alongside drift for a 2D regime classification (flat+calm, flat+volatile, trending+calm, trending+volatile). Each quadrant gets different spread/size parameters.
- **Dynamic max_position:** Reduce max_position in trending regime to limit directional exposure further.
- **Time-based eviction for TradeImbalanceTracker:** RegimeDetector's time-based window pattern could be retrofitted to the existing trade imbalance tracker.

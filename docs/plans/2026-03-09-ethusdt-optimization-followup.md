# ETHUSDT Strategy Optimization Follow-up — Fee Analysis, VPIN Tuning, Bybit Fix

Created: 2026-03-09
Status: VERIFIED
Approved: Yes
Iterations: 0
Worktree: No
Type: Feature

## Summary

**Goal:** Follow up on NOV-13 sweep results: (1) add strategy_params to config and set reprice_bps=12, (2) surface fee breakdown in bench output, (3) sweep vpin_factor [4-10], (4) fix Bybit recorder missing book updates, (5) add short-term trade imbalance filter to adaptive_mm.

**Architecture:** Config changes to `TradingConfig` + `engine.rs` for strategy params plumbing. Bench output enhancements in `replay_bench.rs`. Bybit recorder fix in `record.rs` to pass configurable WS URL. New `TradeImbalanceTracker` signal in `signals.rs` wired into `adaptive_mm.rs` spread calculation.

**Tech Stack:** Rust (config, strategy, bench, recorder)

## Scope

### In Scope
- NOV-15: Add `strategy_params` field to `TradingConfig`, update `config/default.toml` with reprice_bps=12 for ETHUSDT, create K8s configmap for strategy
- NOV-16: Surface `fee_total` in bench output tables (bench_series and bench_ethusdt_optimization)
- NOV-17: New bench sweep `bench_vpin_factor_sweep` with vpin_factor × [4, 6, 8, 10]
- NOV-18: Fix Bybit recorder to use configurable WS URL from `BybitConfig` instead of hardcoded default
- NOV-19: Add `TradeImbalanceTracker` to `signals.rs` and wire into `adaptive_mm` spread calculation

### Out of Scope
- Changing default `reprice_threshold_bps` constant in adaptive_mm (stays 1.5 as code default — prod override via config)
- Taker/maker fee split in SimConfig (all fills are maker in current sim — adding taker logic is a separate task)
- Production deployment of Bybit recorder (fix code only, deploy separately)
- Sweep of `trade_imbalance_threshold` in bench (future task after NOV-19 is validated)

## Context for Implementer

> Write for an implementer who has never seen the codebase.

- **Patterns to follow:**
  - `config.rs:192-198` — `TradingConfig` struct with `mode` and `strategy` fields. Add `strategy_params` here.
  - `config.rs:287-288` — compiled-in defaults for trading section. Add `strategy_params` default `""`.
  - `engine.rs:187-191` — where strategy is created with empty params. Wire config.trading.strategy_params here.
  - `replay_bench.rs` bench output tables — `println!` with box-drawing chars, fixed-width columns.
  - `signals.rs:81-107` — `TradeFlowSignal` with EMA-based buy/sell pressure. New `TradeImbalanceTracker` follows same pattern but uses `VecDeque` rolling window.
  - `adaptive_mm.rs:350-354` — trade_flow_adj applied to mid price. Imbalance filter follows same pattern.

- **Conventions:**
  - Strategy params are JSON: `params.get_f64("key")` with `DEFAULT_*` constants
  - Config uses serde: `#[serde(default = "fn_name")]` for optional fields
  - Tests skip gracefully if data files absent: `eprintln!("SKIP: ..."); return;`

- **Key files:**
  - `crates/core/src/config.rs` (550 lines) — layered config: defaults → TOML → env vars
  - `crates/trading/src/engine.rs` — engine startup, strategy creation
  - `crates/trading/src/bin/record.rs` (~453 lines) — recording binary, Binance/Bybit WS → JSONL
  - `crates/market-data/src/bybit/client.rs` (~530 lines) — Bybit WS client, `BybitConfig`
  - `crates/strategy/src/strategies/signals.rs` — signal trackers (EMA, TradeFlow, Volatility, VPIN)
  - `crates/strategy/src/strategies/adaptive_mm.rs` (~1100 lines) — adaptive MM strategy
  - `crates/trading/tests/replay_bench.rs` (~1900 lines) — benchmark tests
  - `crates/trading/tests/replay_harness.rs` (~450 lines) — replay engine, SimConfig, ReplayResult

- **Gotchas:**
  - `BybitConfig::default()` has `testnet: false` and `ws_url: None` → resolves to `wss://stream.bybit.com/v5/public/linear`. The recorder at `record.rs:308` uses this default — it does NOT read from `AppConfig` or env vars. To make the URL configurable, the recorder must either read from env vars or accept a `--ws-url` CLI arg.
  - `fee_total` in `ReplayResult` already exists (harness.rs:52) but is not printed in any bench output. All fees are computed as maker fees (`maker_fee_bps`). There is no taker fee simulation.
  - `engine.rs:189` creates `StrategyParams { params: serde_json::json!({}) }` — always empty. The new `strategy_params` config field must be parsed from JSON string to `serde_json::Value` here.
  - `TradeFlowSignal` uses EMA (exponentially weighted, infinite memory). `TradeImbalanceTracker` needs a **bounded window** (VecDeque) to capture short-term directional pressure that decays within seconds, not minutes.

- **Domain context:**
  - **reprice_threshold_bps** — minimum mid-price movement (in basis points) before the strategy cancels and re-quotes. Higher = fewer orders but potentially more stale quotes.
  - **vpin_factor** — multiplier on VPIN for spread widening: `spread *= (1 + vpin_factor * vpin)`. Higher = more aggressive spread widening during toxic flow.
  - **Trade imbalance** — short-term buy/sell volume ratio. When buy_volume >> sell_volume over recent trades, the market is likely to move up → widen ask side to avoid adverse fills.

## Progress Tracking

- [x] Task 1: Add strategy_params to config and set reprice_bps=12
- [x] Task 2: Surface fee breakdown in bench output
- [x] Task 3: Add bench_vpin_factor_sweep test
- [x] Task 4: Fix Bybit recorder WS URL
- [x] Task 5: Add TradeImbalanceTracker to adaptive_mm
**Total Tasks:** 5 | **Completed:** 5 | **Remaining:** 0

## Implementation Tasks

### Task 1: Add strategy_params to TradingConfig and set reprice_bps=12

**Objective:** Wire strategy parameters from config file → engine → strategy creation. Set ETHUSDT optimal reprice_bps=12 in default config and K8s configmap.

**Dependencies:** None

**Files:**
- Modify: `crates/core/src/config.rs`
- Modify: `crates/trading/src/engine.rs`
- Modify: `config/default.toml`
- Create: `infra/deploy/k8s/trading/configmap-ethusdt.yaml`

**Key Decisions / Notes:**
- Add `strategy_params: String` to `TradingConfig` (config.rs:192). Use `#[serde(default)]` so it defaults to empty string.
- Add compiled-in default: `.set_default("trading.strategy_params", "")` at config.rs:288.
- In `engine.rs:189`, parse `config.trading.strategy_params`:
  ```rust
  let params = StrategyParams {
      params: if config.trading.strategy_params.is_empty() {
          serde_json::json!({})
      } else {
          serde_json::from_str(&config.trading.strategy_params)
              .unwrap_or_else(|e| {
                  tracing::warn!(error = %e, "invalid strategy_params JSON, using defaults");
                  serde_json::json!({})
              })
      },
  };
  ```
- In `config/default.toml`, add under `[trading]`:
  ```toml
  strategy_params = ''
  ```
  Keep default empty — ETHUSDT-specific value belongs only in K8s configmap to avoid affecting other symbols/deployments.
- K8s configmap: create `infra/deploy/k8s/trading/configmap-ethusdt.yaml` with `CM_HFT_TRADING__STRATEGY_PARAMS` env var.
- Overridable via env: `CM_HFT_TRADING__STRATEGY_PARAMS='{"reprice_threshold_bps": 12.0}'`

**Definition of Done:**
- [ ] `TradingConfig` has `strategy_params: String` field
- [ ] `config/default.toml` includes `strategy_params = ''` (empty default)
- [ ] `engine.rs` parses strategy_params from config into `StrategyParams`
- [ ] K8s configmap created with `CM_HFT_TRADING__STRATEGY_PARAMS='{"reprice_threshold_bps": 12.0}'`
- [ ] Invalid JSON gracefully falls back to empty params (unit test)
- [ ] `cargo test --lib -p cm-core` passes

**Verify:**
- `cargo test --lib -p cm-core -- config --nocapture 2>&1 | tail -10`

### Task 2: Surface fee breakdown in bench output

**Objective:** Display `fee_total` in bench_series and bench_ethusdt_optimization output tables so we can quantify fee impact vs adverse selection.

**Dependencies:** None

**Files:**
- Modify: `crates/trading/tests/replay_bench.rs`

**Key Decisions / Notes:**
- `ReplayResult` already has `fee_total: f64` (harness.rs:52) — no struct changes needed.
- Add `Fees$` column to both bench output tables:
  - `bench_series` result table (around line 1100-1140): add `result.fee_total` column
  - `bench_ethusdt_optimization` result table (around line 1870-1890): add `result.fee_total` column
- Compute `adverse_selection = realized_pnl_before_fees - m2m_pnl` where `realized_pnl_before_fees = result.total_pnl - result.fee_total` (since total_pnl already includes fees). Actually simpler: just show fees column and let user compute.
- Keep column format consistent: `{:>10.2}` for dollar amounts.

**Definition of Done:**
- [ ] `bench_series` output includes `Fees$` column
- [ ] `bench_ethusdt_optimization` output includes `Fees$` column
- [ ] `cargo test --test replay_bench -p cm-trading -- bench_ethusdt_optimization --nocapture` compiles and runs
- [ ] Existing bench_series output format is not broken

**Verify:**
- `cargo test --test replay_bench -p cm-trading -- bench_ethusdt_optimization --nocapture 2>&1 | head -30`

### Task 3: Add bench_vpin_factor_sweep test

**Objective:** Sweep vpin_factor × [4, 6, 8, 10] on 7-day ETHUSDT data with fixed reprice_bps=12, order_size=0.01, flush=0, to find if higher VPIN factor reduces adverse selection enough to go positive.

**Dependencies:** Task 2 (fee column needed in output)

**Files:**
- Modify: `crates/trading/tests/replay_bench.rs`

**Key Decisions / Notes:**
- New test function: `bench_vpin_factor_sweep()`
- Fixed params: reprice_bps=12, order_size=0.01, flush_interval_ticks=0, γ=0.3, κ=1.5, τ=1.0, decay=2.0, boost=0.5
- Sweep: vpin_factor × [0.0, 2.0, 4.0, 6.0, 8.0, 10.0] — include 0.0 (no VPIN) and 2.0 (current baseline)
- Use same `find_series_files()` + `load_events_multi()` pattern as `bench_ethusdt_optimization`
- Output table: Rank, vpin_factor, Fills, Orders, Realized$, M2M$, Fees$, MaxDD$, Calmar, $/fill
- Sort by Calmar descending
- Skip gracefully if no ETHUSDT data in testdata/

**Definition of Done:**
- [ ] `bench_vpin_factor_sweep` test exists and compiles
- [ ] Sweeps 6 vpin_factor values per exchange
- [ ] Output shows ranked results with Calmar ratio and Fees$
- [ ] Test skips gracefully if ETHUSDT data absent
- [ ] `cargo test --test replay_bench -p cm-trading -- bench_vpin_factor_sweep --nocapture` runs

**Verify:**
- `cargo test --test replay_bench -p cm-trading -- bench_vpin_factor_sweep --nocapture 2>&1 | head -30`

### Task 4: Fix Bybit recorder WS URL

**Objective:** Make the Bybit recorder use a configurable WS URL instead of hardcoded `BybitConfig::default()` so it can connect to linear perpetuals correctly and receive orderbook data.

**Dependencies:** None

**Files:**
- Modify: `crates/trading/src/bin/record.rs`

**Key Decisions / Notes:**
- Currently at record.rs:308: `BybitWsClient::new(BybitConfig::default(), symbols_for_ws)` — uses `testnet: false, ws_url: None` → resolves to `wss://stream.bybit.com/v5/public/linear`
- The Binance recorder (record.rs:126-133) builds a full `ExchangeConfig` with explicit URL. Bybit should follow same pattern.
- Add `--ws-url` CLI arg to `Args` struct (optional, overrides default based on exchange):
  - Bybit default: `wss://stream.bybit.com/v5/public/linear`
  - Also respect `RECORD_WS_URL` env var (for K8s configmap)
- Pass the resolved URL into `BybitConfig { ws_url: Some(url), ..Default::default() }`
- **Root cause:** The recorder bypasses `AppConfig` entirely — it creates `BybitConfig::default()` directly, so `CM_HFT_BYBIT__WS_URL` env var has no effect. `BybitConfig::default()` resolves to the correct production linear URL, so the issue is likely that ETHUSDT on Bybit linear requires a different subscription format, or the subscription silently fails. The fix must read `RECORD_WS_URL` env var (or `--ws-url` arg) and pass it directly to `BybitConfig { ws_url: Some(url), ..Default::default() }`.
- Add DoD: verify env var override works by setting `RECORD_WS_URL=wss://invalid.example.com` and confirming the connection error shows the custom URL.
- Update `infra/deploy/k8s/record/configmap-bybit.yaml` to include `RECORD_WS_URL` if needed.

**Definition of Done:**
- [ ] record.rs accepts `--ws-url` CLI arg for custom WS URL
- [ ] `RECORD_WS_URL` env var supported as fallback
- [ ] Bybit recorder passes configured URL to `BybitConfig`
- [ ] Default behavior unchanged (linear endpoint) when no override provided
- [ ] `cargo build --bin cm-record` compiles
- [ ] `cm-record --exchange bybit --symbols ETHUSDT --duration 1m --ws-url "wss://stream.bybit.com/v5/public/linear" --output /tmp/test` runs without error (manual test)

**Verify:**
- `cargo build --bin cm-record && echo "OK"`

### Task 5: Add TradeImbalanceTracker to adaptive_mm

**Objective:** Add a short-term trade imbalance signal that tracks rolling buy/sell volume ratio over recent N trades. When imbalance exceeds threshold, widen the vulnerable side's spread to reduce adverse selection fills.

**Dependencies:** None (can be implemented independently)

**Files:**
- Modify: `crates/strategy/src/strategies/signals.rs`
- Modify: `crates/strategy/src/strategies/adaptive_mm.rs`

**Key Decisions / Notes:**
- **New struct in signals.rs:** `TradeImbalanceTracker`
  ```rust
  pub struct TradeImbalanceTracker {
      window: VecDeque<(bool, f64, u64)>,  // (is_buy, notional, timestamp_ns) — timestamp stored for future time-based eviction
      max_size: usize,                // lookback window size (number of trades)
  }
  ```
  - `update(is_buy: bool, notional: f64)` — push to window, pop if > max_size
  - `imbalance(&self) -> f64` — returns (buy_vol - sell_vol) / (buy_vol + sell_vol), range [-1, 1]. 0 = balanced. +1 = all buys. -1 = all sells.
  - `buy_ratio(&self) -> f64` — returns buy_vol / total_vol, range [0, 1]

- **New params in adaptive_mm:**
  - `imbalance_window: usize` — number of recent trades to track. Default: 0 (disabled).
  - `imbalance_threshold: f64` — |imbalance| above which spread adjustment fires. Default: 0.5.
  - `imbalance_factor: f64` — spread multiplier on vulnerable side. Default: 0.0 (disabled).
  - When `imbalance_factor > 0` and `|imbalance| > imbalance_threshold`:
    - If imbalance > 0 (buy pressure): widen ask by `imbalance_factor * imbalance * half_spread`
    - If imbalance < 0 (sell pressure): widen bid by `imbalance_factor * |imbalance| * half_spread`

- **Integration in adaptive_mm:**
  - Add `imbalance_tracker: TradeImbalanceTracker` field
  - In `on_trade`: call `self.imbalance_tracker.update(is_buy, notional)` alongside existing trade_flow/vpin updates
  - In `on_book_update` spread calculation (around line 450): apply imbalance adjustment to bid/ask prices asymmetrically (not to mid like trade_flow, but to individual sides)
  - Pattern: after computing `bid_price` and `ask_price`, adjust:
    ```rust
    if self.imbalance_factor > 0.0 && self.imbalance_tracker.window_size() > 0 {
        let imb = self.imbalance_tracker.imbalance();
        if imb > self.imbalance_threshold {
            // Buy pressure: widen ask (raise it)
            ask_price += self.imbalance_factor * imb * half_spread;
        } else if imb < -self.imbalance_threshold {
            // Sell pressure: widen bid (lower it)
            bid_price -= self.imbalance_factor * imb.abs() * half_spread;
        }
    }
    ```

- **Backward compatible:** `imbalance_window=0` or `imbalance_factor=0.0` → no effect.

**Definition of Done:**
- [ ] `TradeImbalanceTracker` struct in signals.rs with unit tests
- [ ] Unit test: imbalance returns 0.0 for empty window
- [ ] Unit test: imbalance returns 1.0 for all-buy window, -1.0 for all-sell
- [ ] Unit test: window eviction works (oldest trades dropped when > max_size)
- [ ] adaptive_mm reads `imbalance_window`, `imbalance_threshold`, `imbalance_factor` from params
- [ ] `on_trade` feeds imbalance_tracker
- [ ] `on_book_update` applies asymmetric spread adjustment when imbalance exceeds threshold
- [ ] Default behavior unchanged (imbalance_window=0 → no imbalance tracking)
- [ ] Unit test: adaptive_mm with imbalance_factor=1.0 and all-buy window produces wider ask than bid
- [ ] Unit test: adaptive_mm with imbalance_factor=0.0 produces identical spread regardless of window contents
- [ ] `cargo test --lib -p cm-strategy` passes

**Verify:**
- `cargo test --lib -p cm-strategy -- --nocapture 2>&1 | tail -10`

## Assumptions

- Strategy params as JSON string in TOML is parseable by the `config` crate — supported by existing string params in TradingConfig — Tasks 1 depends on this
- `fee_total` in ReplayResult accurately reflects total fees applied during simulation — supported by harness.rs:420-422 fee calculation — Task 2 depends on this
- Higher vpin_factor will reduce fills (wider spread) but the remaining fills will be higher quality (less adverse selection) — hypothesis, Task 3 will validate
- Bybit linear endpoint subscribes to `orderbook.200.ETHUSDT` correctly — supported by client.rs:79 topic construction and test at client.rs:409 — Task 4 depends on this
- VecDeque-based rolling window is efficient enough for hot path (~100 trades/window, O(1) push/pop) — Task 5 depends on this

## Testing Strategy

- **Unit:** Config parsing tests (Task 1), TradeImbalanceTracker tests (Task 5), adaptive_mm param tests (Task 5)
- **Integration:** Bench output format verification (Tasks 2, 3), recorder compilation (Task 4)
- **Manual:** Bybit recorder 1-minute test run to verify book events arrive (Task 4)
- **Bench:** VPIN sweep on 7-day data to validate hypothesis (Task 3)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TOML config crate can't parse JSON string with nested quotes | Low | Medium | Test with `strategy_params = '{"key": 12.0}'` format. Fallback: env var override. |
| Higher vpin_factor reduces fills to near-zero | Medium | Low | Sweep includes vpin_factor=0.0 and 2.0 as baselines. If 4+ kills all fills, we know the range. |
| Bybit linear ETHUSDT doesn't exist (different symbol format) | Low | High | Verify via `curl "https://api.bybit.com/v5/market/instruments-info?category=linear&symbol=ETHUSDT"` before recording. |
| TradeImbalanceTracker adds latency to on_trade hot path | Low | Low | VecDeque push/pop is O(1). Window size ~100 trades = minimal memory. Measure with bench. |
| imbalance_factor too aggressive widens spread beyond market | Medium | Low | Default disabled (factor=0.0). Bench sweep will determine safe range. |

## Pre-Mortem

*Assume this plan failed. Most likely internal reasons:*

1. **Fee breakdown shows fees are negligible — adverse selection is the real problem** (Task 2) → Trigger: fee_total < $5/week while realized loss is -$41. This would mean VPIN and imbalance filter are the critical path, not fee optimization. Response: skip fee-related work, focus entirely on Tasks 3 and 5.

2. **vpin_factor=10 still shows negative Calmar** (Task 3) → Trigger: all 6 vpin_factor configs have negative realized PnL. This would mean spread widening alone cannot overcome adverse selection — the strategy may need fundamentally different fill logic (e.g., cancelling orders faster, or only quoting one side). Response: document finding, create follow-up task for fill model investigation.

3. **TradeImbalanceTracker fires too often on ETHUSDT (high-frequency trade data)** (Task 5) → Trigger: imbalance exceeds threshold >80% of the time with window=50-100 trades. This would make it effectively a constant spread widener rather than a directional filter. Response: increase default window size or use time-weighted window instead of trade-count window.

## Goal Verification

### Truths
1. Strategy params from config file are plumbed through to strategy creation (reprice_bps=12 is active)
2. Bench output shows fee breakdown alongside realized PnL for all configs
3. VPIN sweep produces ranked results showing Calmar at each vpin_factor level
4. Bybit recorder produces files with both book and trade events for ETHUSDT
5. TradeImbalanceTracker correctly computes directional imbalance and is disabled by default
6. All existing tests pass unchanged

### Artifacts
- `crates/core/src/config.rs` — strategy_params in TradingConfig
- `crates/trading/src/engine.rs` — strategy_params parsing
- `config/default.toml` — reprice_bps=12 default
- `infra/deploy/k8s/trading/configmap-ethusdt.yaml` — K8s strategy config
- `crates/trading/tests/replay_bench.rs` — fee column + vpin sweep
- `crates/trading/src/bin/record.rs` — configurable WS URL
- `crates/strategy/src/strategies/signals.rs` — TradeImbalanceTracker
- `crates/strategy/src/strategies/adaptive_mm.rs` — imbalance params + spread adjustment

### Key Links
- `config.rs:TradingConfig.strategy_params` → `engine.rs` strategy creation → `adaptive_mm.rs:from_params()`
- `replay_harness.rs:fee_total` → `replay_bench.rs` output tables
- `record.rs` → `BybitConfig.ws_url` → `bybit/client.rs:ws_url()`
- `signals.rs:TradeImbalanceTracker` → `adaptive_mm.rs:on_trade()` → `adaptive_mm.rs:on_book_update()` spread adjustment

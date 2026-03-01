"""Comprehensive tests for backtest engine modules.

Covers:
  - backtest.engine.orchestrator: BacktestParams, BacktestResult, BacktestOrchestrator
  - backtest.engine.analytics: MetricsCalculator, PerformanceMetrics,
    ReportGenerator, compare_results
  - backtest.engine.walk_forward: WalkForwardOptimizer, WalkForwardWindow, WalkForwardResult
  - backtest.engine.latency_analysis: LatencyAnalyzer, LatencyPoint, LatencySensitivityResult
"""

import json

import numpy as np
import polars as pl
import pytest

from backtest.engine.analytics import (
    MetricsCalculator,
    PerformanceMetrics,
    ReportGenerator,
    compare_results,
)
from backtest.engine.latency_analysis import (
    LatencyAnalyzer,
    LatencyPoint,
    LatencySensitivityResult,
)
from backtest.engine.orchestrator import (
    BacktestOrchestrator,
    BacktestParams,
    BacktestResult,
)
from backtest.engine.walk_forward import (
    WalkForwardOptimizer,
    WalkForwardResult,
    WalkForwardWindow,
)

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def orchestrator(tmp_path):
    """Orchestrator rooted in a temp directory."""
    return BacktestOrchestrator(output_dir=str(tmp_path / "bt_output"))


@pytest.fixture
def default_params():
    """Default backtest parameters."""
    return BacktestParams()


@pytest.fixture
def sample_result(orchestrator, default_params):
    """A small backtest result from the Python sim fallback."""
    return orchestrator.run_single(default_params, num_events=500)


@pytest.fixture
def sample_trades():
    """Four trades forming two buy/sell pairs."""
    return [
        {
            "timestamp_ns": 1_000_000,
            "side": "buy",
            "price": 50_000.0,
            "quantity": 0.001,
            "fee": 0.005,
        },
        {
            "timestamp_ns": 3_000_000,
            "side": "sell",
            "price": 50_010.0,
            "quantity": 0.001,
            "fee": 0.005,
        },
        {
            "timestamp_ns": 5_000_000,
            "side": "buy",
            "price": 50_020.0,
            "quantity": 0.001,
            "fee": 0.005,
        },
        {
            "timestamp_ns": 8_000_000,
            "side": "sell",
            "price": 50_005.0,
            "quantity": 0.001,
            "fee": 0.005,
        },
    ]


@pytest.fixture
def trades_with_pnl():
    """Trades carrying an explicit pnl field."""
    return [
        {"pnl": 10.0},
        {"pnl": -3.0},
        {"pnl": 5.0},
        {"pnl": -1.0},
        {"pnl": 8.0},
    ]


# ===========================================================================
# 1. orchestrator.py -- BacktestParams
# ===========================================================================


class TestBacktestParams:
    def test_default_values(self):
        """BacktestParams defaults are sensible market-making parameters."""
        p = BacktestParams()
        assert p.strategy == "market_making"
        assert p.maker_fee == -0.0001
        assert p.taker_fee == 0.0004
        assert p.latency_ns == 1_000_000
        assert "spread_bps" in p.params
        assert "order_size" in p.params
        assert "num_levels" in p.params
        assert "reprice_threshold_bps" in p.params
        assert "skew_factor" in p.params

    def test_custom_values(self):
        """BacktestParams with fully custom configuration."""
        custom = BacktestParams(
            strategy="arbitrage",
            params={"threshold": 0.01},
            maker_fee=-0.00025,
            taker_fee=0.0005,
            latency_ns=500_000,
        )
        assert custom.strategy == "arbitrage"
        assert custom.params == {"threshold": 0.01}
        assert custom.maker_fee == -0.00025
        assert custom.taker_fee == 0.0005
        assert custom.latency_ns == 500_000


# ===========================================================================
# 1. orchestrator.py -- BacktestResult
# ===========================================================================


class TestBacktestResult:
    def test_to_dict_keys(self, sample_result):
        """to_dict() returns all expected top-level keys."""
        d = sample_result.to_dict()
        expected_keys = {
            "strategy",
            "params",
            "maker_fee",
            "taker_fee",
            "latency_ns",
            "total_pnl",
            "total_fees",
            "trade_count",
            "fill_count",
            "max_position",
            "duration_seconds",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_params_expanded(self, sample_result):
        """Params dict is serialized inline."""
        d = sample_result.to_dict()
        assert isinstance(d["params"], dict)
        assert "spread_bps" in d["params"]

    def test_save_creates_pnl_parquet(self, sample_result, tmp_path):
        """save() writes pnl_series.parquet."""
        out = tmp_path / "result_dir"
        sample_result.save(out)
        assert (out / "pnl_series.parquet").exists()
        df = pl.read_parquet(out / "pnl_series.parquet")
        assert "timestamp_ns" in df.columns
        assert "pnl" in df.columns
        assert len(df) == len(sample_result.pnl_series)

    def test_save_creates_metadata_json(self, sample_result, tmp_path):
        """save() writes metadata.json with correct strategy."""
        out = tmp_path / "result_dir"
        sample_result.save(out)
        with open(out / "metadata.json") as f:
            meta = json.load(f)
        assert meta["strategy"] == sample_result.params.strategy
        assert "total_pnl" in meta

    def test_save_creates_trades_parquet(self, sample_result, tmp_path):
        """save() writes trades.parquet when trades exist."""
        out = tmp_path / "result_dir"
        sample_result.save(out)
        if sample_result.trades:
            assert (out / "trades.parquet").exists()

    def test_save_no_trades(self, tmp_path):
        """save() omits trades.parquet when trade list is empty."""
        result = BacktestResult(
            params=BacktestParams(),
            total_pnl=0.0,
            total_fees=0.0,
            trade_count=0,
            fill_count=0,
            max_position=0.0,
            pnl_series=[0.0, 0.0],
            timestamp_series=[1_000_000, 2_000_000],
            trades=[],
            duration_seconds=0.01,
        )
        out = tmp_path / "empty_trades"
        result.save(out)
        assert (out / "pnl_series.parquet").exists()
        assert not (out / "trades.parquet").exists()


# ===========================================================================
# 1. orchestrator.py -- BacktestOrchestrator
# ===========================================================================


class TestBacktestOrchestrator:
    def test_run_single_produces_result(self, orchestrator):
        """run_single() returns a BacktestResult with non-empty series."""
        result = orchestrator.run_single(BacktestParams(), num_events=500)
        assert isinstance(result, BacktestResult)
        assert len(result.pnl_series) == 499
        assert len(result.timestamp_series) == 499
        assert result.duration_seconds > 0

    def test_run_single_deterministic(self, orchestrator):
        """Two identical runs produce identical PnL (seeded RNG with seed=42)."""
        r1 = orchestrator.run_single(BacktestParams(), num_events=300)
        r2 = orchestrator.run_single(BacktestParams(), num_events=300)
        assert r1.total_pnl == r2.total_pnl
        assert r1.fill_count == r2.fill_count

    def test_sweep_params_cartesian(self, orchestrator):
        """sweep_params() returns one result per combination."""
        results = orchestrator.sweep_params(
            BacktestParams(),
            sweep={"spread_bps": [3.0, 5.0], "order_size": [0.001, 0.005]},
            num_events=200,
        )
        assert len(results) == 4
        combos = [(r.params.params["spread_bps"], r.params.params["order_size"]) for r in results]
        assert (3.0, 0.001) in combos
        assert (5.0, 0.005) in combos

    def test_save_results(self, orchestrator, tmp_path):
        """save_results() writes summary.parquet and per-run directories."""
        results = orchestrator.sweep_params(
            BacktestParams(),
            sweep={"spread_bps": [3.0, 5.0]},
            num_events=200,
        )
        out = orchestrator.save_results(results, "engine_test")
        assert (out / "summary.parquet").exists()
        df = pl.read_parquet(out / "summary.parquet")
        assert len(df) == 2
        assert "total_pnl" in df.columns
        assert "param_spread_bps" in df.columns
        assert (out / "run_0000" / "metadata.json").exists()
        assert (out / "run_0001" / "pnl_series.parquet").exists()


# ===========================================================================
# 2. analytics.py -- MetricsCalculator (sharpe_ratio)
# ===========================================================================


class TestSharpeRatio:
    def test_positive_returns(self):
        """Net-positive returns yield positive Sharpe."""
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01])
        assert MetricsCalculator.sharpe_ratio(returns) > 0

    def test_negative_returns(self):
        """Net-negative returns yield negative Sharpe."""
        returns = np.array([-0.02, -0.01, -0.03, 0.001, -0.015])
        assert MetricsCalculator.sharpe_ratio(returns) < 0

    def test_zero_std(self):
        """Constant returns produce zero Sharpe (division guard)."""
        returns = np.array([0.01, 0.01, 0.01])
        assert MetricsCalculator.sharpe_ratio(returns) == 0.0

    def test_empty_array(self):
        """Empty input yields zero."""
        assert MetricsCalculator.sharpe_ratio(np.array([])) == 0.0


# ===========================================================================
# 2. analytics.py -- MetricsCalculator (sortino_ratio)
# ===========================================================================


class TestSortinoRatio:
    def test_all_positive(self):
        """No downside returns yield inf Sortino."""
        returns = np.array([0.01, 0.02, 0.015, 0.005])
        assert MetricsCalculator.sortino_ratio(returns) == float("inf")

    def test_mixed_returns(self):
        """Mixed returns produce a finite positive Sortino."""
        returns = np.array([0.01, -0.005, 0.02, -0.003, 0.01])
        sortino = MetricsCalculator.sortino_ratio(returns)
        assert 0 < sortino < float("inf")

    def test_all_negative(self):
        """Entirely negative returns yield a negative Sortino."""
        returns = np.array([-0.01, -0.02, -0.015])
        assert MetricsCalculator.sortino_ratio(returns) < 0

    def test_empty(self):
        """Empty returns yield zero."""
        assert MetricsCalculator.sortino_ratio(np.array([])) == 0.0


# ===========================================================================
# 2. analytics.py -- MetricsCalculator (max_drawdown)
# ===========================================================================


class TestMaxDrawdown:
    def test_monotonic_increase(self):
        """No drawdown on a strictly increasing curve."""
        dd_abs, dd_pct = MetricsCalculator.max_drawdown([1, 2, 3, 4, 5])
        assert dd_abs == 0.0
        assert dd_pct == 0.0

    def test_monotonic_decrease(self):
        """Drawdown equals the full decline from first (peak) to last value."""
        dd_abs, dd_pct = MetricsCalculator.max_drawdown([100, 80, 60, 40, 20])
        assert dd_abs == pytest.approx(80.0)
        assert dd_pct == pytest.approx(80.0)

    def test_peak_valley(self):
        """Classic peak-then-valley-then-recovery pattern."""
        pnl = [0, 10, 20, 15, 5, 25, 30]
        dd_abs, dd_pct = MetricsCalculator.max_drawdown(pnl)
        assert dd_abs == pytest.approx(15.0)  # peak 20, trough 5
        assert dd_pct == pytest.approx(75.0)  # 15/20 * 100

    def test_empty_series(self):
        """Empty series returns (0, 0)."""
        assert MetricsCalculator.max_drawdown([]) == (0.0, 0.0)

    def test_single_element(self):
        """Single element returns (0, 0)."""
        assert MetricsCalculator.max_drawdown([42]) == (0.0, 0.0)


# ===========================================================================
# 2. analytics.py -- MetricsCalculator (profit_factor)
# ===========================================================================


class TestProfitFactor:
    def test_profitable(self):
        """Mix of wins/losses produces finite positive factor."""
        trades = [{"pnl": 10.0}, {"pnl": -3.0}, {"pnl": 5.0}, {"pnl": -2.0}]
        pf = MetricsCalculator.profit_factor(trades)
        # gross profit = 15, gross loss = 5 => pf = 3.0
        assert pf == pytest.approx(3.0)

    def test_unprofitable(self):
        """More losses than gains yields factor < 1."""
        trades = [{"pnl": 2.0}, {"pnl": -10.0}, {"pnl": 1.0}, {"pnl": -5.0}]
        pf = MetricsCalculator.profit_factor(trades)
        assert pf < 1.0

    def test_all_wins(self):
        """No losses yields inf."""
        trades = [{"pnl": 5.0}, {"pnl": 3.0}]
        assert MetricsCalculator.profit_factor(trades) == float("inf")

    def test_no_trades(self):
        """No trades yields 0."""
        assert MetricsCalculator.profit_factor([]) == 0.0


# ===========================================================================
# 2. analytics.py -- MetricsCalculator (win_rate)
# ===========================================================================


class TestWinRate:
    def test_mixed(self):
        """3 winners out of 5 trades."""
        trades = [{"pnl": 10.0}, {"pnl": -3.0}, {"pnl": 5.0}, {"pnl": -1.0}, {"pnl": 8.0}]
        assert MetricsCalculator.win_rate(trades) == pytest.approx(3 / 5)

    def test_all_wins(self):
        """Win rate of 1.0 when every trade is profitable."""
        trades = [{"pnl": 1.0}, {"pnl": 2.0}, {"pnl": 3.0}]
        assert MetricsCalculator.win_rate(trades) == pytest.approx(1.0)

    def test_all_losses(self):
        """Win rate of 0.0 when every trade loses."""
        trades = [{"pnl": -1.0}, {"pnl": -2.0}]
        assert MetricsCalculator.win_rate(trades) == pytest.approx(0.0)

    def test_no_trades(self):
        """Empty list gives 0."""
        assert MetricsCalculator.win_rate([]) == 0.0


# ===========================================================================
# 2. analytics.py -- MetricsCalculator.calculate()
# ===========================================================================


class TestMetricsCalculate:
    def test_basic_calculation(self, sample_trades):
        """Full pipeline returns PerformanceMetrics with expected totals."""
        pnl_series = [0.0, 5.0, 10.0, 8.0, 3.0, 12.0, 15.0]
        metrics = MetricsCalculator.calculate(
            pnl_series=pnl_series,
            trades=sample_trades,
            total_fees=0.02,
        )
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_pnl == pytest.approx(15.0)
        assert metrics.net_pnl == pytest.approx(15.0 - 0.02)
        assert metrics.trade_count == 4
        assert metrics.max_drawdown > 0

    def test_empty_inputs(self):
        """Empty series and trades return zero-filled metrics."""
        metrics = MetricsCalculator.calculate([], [])
        assert metrics.total_pnl == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.trade_count == 0


# ===========================================================================
# 2. analytics.py -- ReportGenerator
# ===========================================================================


class TestReportGenerator:
    def test_generate_single_html(self, tmp_path, sample_result):
        """generate_single() creates an HTML file with expected sections."""
        gen = ReportGenerator(output_dir=str(tmp_path / "reports"))
        metrics = MetricsCalculator.calculate(
            sample_result.pnl_series,
            sample_result.trades,
            sample_result.total_fees,
        )
        path = gen.generate_single(sample_result, metrics, name="single_test")
        assert path.exists()
        assert path.suffix == ".html"
        content = path.read_text()
        assert "Backtest Report" in content
        assert "Equity Curve" in content
        assert "Drawdown" in content
        assert "<table" in content

    def test_generate_sweep_html(self, tmp_path, orchestrator):
        """generate_sweep() creates an HTML file with a sweep summary table."""
        results = orchestrator.sweep_params(
            BacktestParams(),
            sweep={"spread_bps": [3.0, 5.0]},
            num_events=200,
        )
        gen = ReportGenerator(output_dir=str(tmp_path / "reports"))
        path = gen.generate_sweep(results, name="sweep_test")
        assert path.exists()
        content = path.read_text()
        assert "Sweep Report" in content
        assert "Sharpe" in content


# ===========================================================================
# 2. analytics.py -- compare_results()
# ===========================================================================


class TestCompareResults:
    def test_returns_polars_dataframe(self, orchestrator):
        """compare_results() produces a polars DataFrame with correct shape."""
        results = orchestrator.sweep_params(
            BacktestParams(),
            sweep={"spread_bps": [3.0, 5.0, 7.0]},
            num_events=200,
        )
        df = compare_results(results)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3
        assert "label" in df.columns
        assert "total_pnl" in df.columns
        assert "sharpe_ratio" in df.columns

    def test_custom_labels(self, orchestrator):
        """Custom labels are applied."""
        results = orchestrator.sweep_params(
            BacktestParams(),
            sweep={"spread_bps": [3.0, 5.0]},
            num_events=200,
        )
        df = compare_results(results, labels=["narrow", "wide"])
        assert df["label"].to_list() == ["narrow", "wide"]


# ===========================================================================
# 3. walk_forward.py -- WalkForwardOptimizer.create_windows
# ===========================================================================


class TestCreateWindows:
    def test_correct_splits(self):
        """Windows have contiguous, non-overlapping IS/OOS ranges."""
        opt = WalkForwardOptimizer(total_events=10000)
        windows = opt.create_windows(is_size=3000, oos_size=1000)
        assert len(windows) > 0
        for w in windows:
            assert w.oos_start_event == w.is_end_event
            assert w.oos_end_event - w.oos_start_event == 1000
            assert w.is_end_event - w.is_start_event == 3000

    def test_window_count(self):
        """Number of windows matches expected from total_events and step_size."""
        opt = WalkForwardOptimizer(total_events=10000)
        windows = opt.create_windows(is_size=3000, oos_size=1000)
        # default step_size = oos_size = 1000
        # windows start at 0, 1000, 2000, ... while start + 4000 <= 10000
        # => starts 0..6000 => 7 windows
        assert len(windows) == 7

    def test_custom_step_size(self):
        """Custom step_size controls overlap between windows."""
        opt = WalkForwardOptimizer(total_events=10000)
        windows = opt.create_windows(is_size=3000, oos_size=1000, step_size=2000)
        # starts: 0, 2000, 4000, 6000 (6000+3000+1000=10000 OK)
        assert len(windows) == 4

    def test_ids_are_sequential(self):
        """Window IDs count from 0."""
        opt = WalkForwardOptimizer(total_events=10000)
        windows = opt.create_windows(is_size=4000, oos_size=2000)
        ids = [w.window_id for w in windows]
        assert ids == list(range(len(windows)))


# ===========================================================================
# 3. walk_forward.py -- WalkForwardOptimizer.optimize
# ===========================================================================


class TestWalkForwardOptimize:
    def test_optimize_returns_result(self):
        """optimize() produces a WalkForwardResult with populated windows."""
        opt = WalkForwardOptimizer(total_events=5000, volatility=10.0)
        result = opt.optimize(
            strategy="market_making",
            param_grid={"spread_bps": [3.0, 5.0]},
            is_size=2000,
            oos_size=1000,
            base_params={
                "order_size": 0.001,
                "num_levels": 1,
                "reprice_threshold_bps": 2.0,
                "skew_factor": 0.5,
            },
        )
        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) > 0
        for w in result.windows:
            assert w.best_params is not None
            assert "spread_bps" in w.best_params


# ===========================================================================
# 3. walk_forward.py -- WalkForwardResult
# ===========================================================================


class TestWalkForwardResult:
    @pytest.fixture
    def wf_result(self):
        """A synthetic WalkForwardResult for unit-level property tests."""
        windows = [
            WalkForwardWindow(
                window_id=0,
                is_start_event=0,
                is_end_event=3000,
                oos_start_event=3000,
                oos_end_event=4000,
                best_params={"spread_bps": 5.0},
                is_sharpe=2.0,
                oos_sharpe=1.5,
                is_pnl=100.0,
                oos_pnl=60.0,
            ),
            WalkForwardWindow(
                window_id=1,
                is_start_event=1000,
                is_end_event=4000,
                oos_start_event=4000,
                oos_end_event=5000,
                best_params={"spread_bps": 3.0},
                is_sharpe=1.8,
                oos_sharpe=0.9,
                is_pnl=80.0,
                oos_pnl=30.0,
            ),
        ]
        return WalkForwardResult(
            windows=windows,
            param_grid={"spread_bps": [3.0, 5.0]},
            strategy="market_making",
        )

    def test_summary_keys(self, wf_result):
        """summary() returns expected keys."""
        s = wf_result.summary()
        expected = {
            "num_windows",
            "avg_is_sharpe",
            "avg_oos_sharpe",
            "overfitting_ratio",
            "total_oos_pnl",
        }
        assert expected == set(s.keys())

    def test_overfitting_ratio(self, wf_result):
        """Overfitting ratio = avg_oos_sharpe / avg_is_sharpe."""
        avg_is = (2.0 + 1.8) / 2
        avg_oos = (1.5 + 0.9) / 2
        expected = avg_oos / avg_is
        assert wf_result.overfitting_ratio == pytest.approx(expected, rel=1e-3)

    def test_to_json_is_valid(self, wf_result):
        """to_json() returns parseable JSON with correct structure."""
        raw = wf_result.to_json()
        data = json.loads(raw)
        assert data["strategy"] == "market_making"
        assert len(data["windows"]) == 2
        assert "summary" in data

    def test_save(self, wf_result, tmp_path):
        """save() writes the JSON to disk."""
        path = tmp_path / "wf_result.json"
        wf_result.save(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["strategy"] == "market_making"


# ===========================================================================
# 4. latency_analysis.py -- LatencyAnalyzer.analyze
# ===========================================================================


class TestLatencyAnalyzer:
    def test_analyze_returns_result(self):
        """analyze() returns LatencySensitivityResult with the right number of points."""
        analyzer = LatencyAnalyzer(num_events=500, volatility=10.0)
        latencies = [0, 1, 5]
        result = analyzer.analyze(
            strategy="market_making",
            params={
                "spread_bps": 5.0,
                "order_size": 0.001,
                "num_levels": 1,
                "reprice_threshold_bps": 2.0,
                "skew_factor": 0.5,
            },
            latencies_ms=latencies,
        )
        assert isinstance(result, LatencySensitivityResult)
        assert len(result.points) == len(latencies)
        for pt in result.points:
            assert isinstance(pt, LatencyPoint)
            assert pt.latency_ns == int(pt.latency_ms * 1_000_000)


# ===========================================================================
# 4. latency_analysis.py -- LatencySensitivityResult properties
# ===========================================================================


class TestLatencySensitivityResult:
    @pytest.fixture
    def sensitivity_result_negative_slope(self):
        """PnL decreases with latency (latency-sensitive strategy)."""
        points = [
            LatencyPoint(
                latency_ms=0,
                latency_ns=0,
                total_pnl=100.0,
                sharpe_ratio=2.0,
                fill_count=50,
                trade_count=50,
            ),
            LatencyPoint(
                latency_ms=5,
                latency_ns=5_000_000,
                total_pnl=60.0,
                sharpe_ratio=1.2,
                fill_count=40,
                trade_count=40,
            ),
            LatencyPoint(
                latency_ms=10,
                latency_ns=10_000_000,
                total_pnl=30.0,
                sharpe_ratio=0.5,
                fill_count=25,
                trade_count=25,
            ),
            LatencyPoint(
                latency_ms=20,
                latency_ns=20_000_000,
                total_pnl=10.0,
                sharpe_ratio=0.1,
                fill_count=15,
                trade_count=15,
            ),
        ]
        return LatencySensitivityResult(
            strategy="market_making",
            params={"spread_bps": 5.0},
            points=points,
        )

    @pytest.fixture
    def sensitivity_result_flat(self):
        """PnL stays roughly constant (not latency-sensitive)."""
        points = [
            LatencyPoint(
                latency_ms=0,
                latency_ns=0,
                total_pnl=50.0,
                sharpe_ratio=1.0,
                fill_count=30,
                trade_count=30,
            ),
            LatencyPoint(
                latency_ms=10,
                latency_ns=10_000_000,
                total_pnl=48.0,
                sharpe_ratio=0.95,
                fill_count=29,
                trade_count=29,
            ),
        ]
        return LatencySensitivityResult(
            strategy="market_making",
            params={"spread_bps": 5.0},
            points=points,
        )

    def test_sensitivity_coefficient_negative(self, sensitivity_result_negative_slope):
        """Negative slope when PnL decreases with latency."""
        coeff = sensitivity_result_negative_slope.sensitivity_coefficient
        assert coeff < 0

    def test_sensitivity_coefficient_single_point(self):
        """Single point => coefficient is 0."""
        result = LatencySensitivityResult(
            strategy="mm",
            params={},
            points=[
                LatencyPoint(
                    latency_ms=0,
                    latency_ns=0,
                    total_pnl=50.0,
                    sharpe_ratio=1.0,
                    fill_count=10,
                    trade_count=10,
                )
            ],
        )
        assert result.sensitivity_coefficient == 0.0

    def test_is_latency_sensitive_true(self, sensitivity_result_negative_slope):
        """PnL at 10ms is <50% of PnL at 0ms => latency sensitive."""
        # 30 / 100 = 0.3 < 0.5 => True
        assert sensitivity_result_negative_slope.is_latency_sensitive is True

    def test_is_latency_sensitive_false(self, sensitivity_result_flat):
        """PnL barely drops => not latency sensitive."""
        # 48 / 50 = 0.96 >= 0.5 => False
        assert sensitivity_result_flat.is_latency_sensitive is False

    def test_summary_structure(self, sensitivity_result_negative_slope):
        """summary() contains expected keys and point details."""
        s = sensitivity_result_negative_slope.summary()
        assert "strategy" in s
        assert "num_points" in s
        assert "sensitivity_coefficient" in s
        assert "is_latency_sensitive" in s
        assert "points" in s
        assert len(s["points"]) == 4

    def test_save(self, sensitivity_result_negative_slope, tmp_path):
        """save() writes a valid JSON file."""
        path = tmp_path / "latency.json"
        sensitivity_result_negative_slope.save(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["strategy"] == "market_making"
        assert data["num_points"] == 4

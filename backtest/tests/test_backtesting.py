"""Tests for backtesting orchestrator and analytics."""

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
from backtest.engine.orchestrator import (
    BacktestOrchestrator,
    BacktestParams,
    BacktestResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def orchestrator(tmp_path):
    """Create an orchestrator with a temporary output directory."""
    return BacktestOrchestrator(output_dir=str(tmp_path / "bt_output"))


@pytest.fixture
def default_params():
    """Default backtest parameters."""
    return BacktestParams()


@pytest.fixture
def sample_result(orchestrator, default_params):
    """Run a small backtest and return the result."""
    return orchestrator.run_single(default_params, num_events=500)


@pytest.fixture
def sample_trades():
    """Sample trade list for analytics tests."""
    return [
        {
            "timestamp_ns": 1_000_000,
            "side": "buy",
            "price": 50000.0,
            "quantity": 0.001,
            "fee": 0.005,
        },
        {
            "timestamp_ns": 3_000_000,
            "side": "sell",
            "price": 50010.0,
            "quantity": 0.001,
            "fee": 0.005,
        },
        {
            "timestamp_ns": 5_000_000,
            "side": "buy",
            "price": 50020.0,
            "quantity": 0.001,
            "fee": 0.005,
        },
        {
            "timestamp_ns": 8_000_000,
            "side": "sell",
            "price": 50005.0,
            "quantity": 0.001,
            "fee": 0.005,
        },
    ]


@pytest.fixture
def trades_with_pnl():
    """Trades with explicit pnl field."""
    return [
        {"pnl": 10.0},
        {"pnl": -3.0},
        {"pnl": 5.0},
        {"pnl": -1.0},
        {"pnl": 8.0},
    ]


# ---------------------------------------------------------------------------
# TestBacktestOrchestrator
# ---------------------------------------------------------------------------


class TestBacktestOrchestrator:
    def test_run_single_python_sim(self, orchestrator):
        """Test that pure Python simulation produces reasonable results."""
        result = orchestrator.run_single(BacktestParams(), num_events=1000)
        assert isinstance(result, BacktestResult)
        assert result.fill_count > 0
        assert len(result.pnl_series) > 0
        assert result.duration_seconds > 0

    def test_run_single_returns_trades(self, orchestrator):
        """Test that simulation returns trade records."""
        result = orchestrator.run_single(BacktestParams(), num_events=1000)
        assert isinstance(result.trades, list)
        assert len(result.trades) > 0
        trade = result.trades[0]
        assert "timestamp_ns" in trade
        assert "side" in trade
        assert "price" in trade
        assert "quantity" in trade
        assert "fee" in trade
        assert trade["side"] in ("buy", "sell")

    def test_run_single_pnl_series_length(self, orchestrator):
        """PnL series length should match num_events - 1 (starts from index 1)."""
        n = 500
        result = orchestrator.run_single(BacktestParams(), num_events=n)
        assert len(result.pnl_series) == n - 1
        assert len(result.timestamp_series) == n - 1

    def test_run_single_deterministic(self, orchestrator):
        """Two runs with same params should produce identical results (seeded RNG)."""
        r1 = orchestrator.run_single(BacktestParams(), num_events=500)
        r2 = orchestrator.run_single(BacktestParams(), num_events=500)
        assert r1.total_pnl == r2.total_pnl
        assert r1.fill_count == r2.fill_count
        assert r1.pnl_series == r2.pnl_series

    def test_sweep_params(self, orchestrator):
        """Test parameter sweep runs all combinations."""
        results = orchestrator.sweep_params(
            BacktestParams(),
            sweep={"spread_bps": [3.0, 5.0, 10.0]},
            num_events=500,
        )
        assert len(results) == 3
        # Each result should have its own spread_bps value
        spreads = [r.params.params["spread_bps"] for r in results]
        assert spreads == [3.0, 5.0, 10.0]

    def test_sweep_params_cartesian(self, orchestrator):
        """Sweep with two parameters produces Cartesian product."""
        results = orchestrator.sweep_params(
            BacktestParams(),
            sweep={"spread_bps": [3.0, 5.0], "order_size": [0.001, 0.005]},
            num_events=200,
        )
        assert len(results) == 4  # 2 * 2

    def test_save_results(self, orchestrator, tmp_path):
        """Test saving sweep results creates expected files."""
        results = orchestrator.sweep_params(
            BacktestParams(),
            sweep={"spread_bps": [3.0, 5.0]},
            num_events=200,
        )
        out_dir = orchestrator.save_results(results, "test_sweep")
        assert (out_dir / "summary.parquet").exists()
        assert (out_dir / "run_0000" / "metadata.json").exists()
        assert (out_dir / "run_0000" / "pnl_series.parquet").exists()

        # Verify summary parquet is readable
        df = pl.read_parquet(out_dir / "summary.parquet")
        assert len(df) == 2
        assert "total_pnl" in df.columns

    def test_backtest_params_defaults(self):
        """Test that default params have expected values."""
        p = BacktestParams()
        assert p.strategy == "market_making"
        assert p.maker_fee == -0.0001
        assert p.taker_fee == 0.0004
        assert p.latency_ns == 1_000_000
        assert "spread_bps" in p.params
        assert "order_size" in p.params

    def test_backtest_result_to_dict(self, sample_result):
        """Test result serialization to dict."""
        d = sample_result.to_dict()
        assert "strategy" in d
        assert "total_pnl" in d
        assert "params" in d
        assert isinstance(d["params"], dict)

    def test_backtest_result_save(self, sample_result, tmp_path):
        """Test saving individual result to disk."""
        out_path = tmp_path / "single_run"
        sample_result.save(out_path)
        assert (out_path / "pnl_series.parquet").exists()
        assert (out_path / "metadata.json").exists()
        if sample_result.trades:
            assert (out_path / "trades.parquet").exists()

        # Verify metadata JSON
        with open(out_path / "metadata.json") as f:
            meta = json.load(f)
        assert meta["strategy"] == "market_making"

    def test_run_single_max_position(self, orchestrator):
        """Max position should be non-negative."""
        result = orchestrator.run_single(BacktestParams(), num_events=500)
        assert result.max_position >= 0


# ---------------------------------------------------------------------------
# TestMetricsCalculator
# ---------------------------------------------------------------------------


class TestMetricsCalculator:
    def test_sharpe_ratio_positive(self):
        """Positive returns should yield positive Sharpe."""
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01])
        sharpe = MetricsCalculator.sharpe_ratio(returns)
        assert sharpe > 0

    def test_sharpe_ratio_zero_std(self):
        """Constant returns yield zero Sharpe (zero std)."""
        returns = np.array([0.01, 0.01, 0.01])
        sharpe = MetricsCalculator.sharpe_ratio(returns)
        assert sharpe == 0.0

    def test_sharpe_ratio_empty(self):
        """Empty returns yield zero Sharpe."""
        sharpe = MetricsCalculator.sharpe_ratio(np.array([]))
        assert sharpe == 0.0

    def test_sharpe_ratio_negative(self):
        """Mostly negative returns should yield negative Sharpe."""
        returns = np.array([-0.02, -0.01, -0.03, 0.001, -0.015])
        sharpe = MetricsCalculator.sharpe_ratio(returns)
        assert sharpe < 0

    def test_sortino_ratio_positive(self):
        """Positive returns with some downside."""
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01])
        sortino = MetricsCalculator.sortino_ratio(returns)
        assert sortino > 0

    def test_sortino_ratio_no_downside(self):
        """All positive returns should yield inf Sortino."""
        returns = np.array([0.01, 0.02, 0.015])
        sortino = MetricsCalculator.sortino_ratio(returns)
        assert sortino == float("inf")

    def test_sortino_ratio_empty(self):
        """Empty returns yield zero."""
        sortino = MetricsCalculator.sortino_ratio(np.array([]))
        assert sortino == 0.0

    def test_max_drawdown(self):
        """Standard drawdown case."""
        pnl = [0, 10, 20, 15, 5, 25, 30]
        dd_abs, dd_pct = MetricsCalculator.max_drawdown(pnl)
        assert dd_abs == 15  # from 20 to 5
        assert dd_pct > 0

    def test_max_drawdown_no_drawdown(self):
        """Monotonically increasing series has zero drawdown."""
        pnl = [0, 10, 20, 30]
        dd_abs, _ = MetricsCalculator.max_drawdown(pnl)
        assert dd_abs == 0

    def test_max_drawdown_empty(self):
        """Empty series returns zeros."""
        dd_abs, dd_pct = MetricsCalculator.max_drawdown([])
        assert dd_abs == 0.0
        assert dd_pct == 0.0

    def test_max_drawdown_single_element(self):
        """Single element series returns zeros."""
        dd_abs, dd_pct = MetricsCalculator.max_drawdown([100])
        assert dd_abs == 0.0

    def test_profit_factor(self, sample_trades):
        """Profit factor from trade pairs."""
        pf = MetricsCalculator.profit_factor(sample_trades)
        # First pair: sell 50010 - buy 50000 = +10 (per unit)
        # Second pair: sell 50005 - buy 50020 = -15 (per unit)
        assert isinstance(pf, float)
        assert pf >= 0

    def test_profit_factor_all_wins(self, trades_with_pnl):
        """All winning trades yield inf profit factor if no losses exist."""
        all_wins = [{"pnl": 10.0}, {"pnl": 5.0}]
        pf = MetricsCalculator.profit_factor(all_wins)
        assert pf == float("inf")

    def test_profit_factor_no_trades(self):
        """No trades yields zero."""
        pf = MetricsCalculator.profit_factor([])
        assert pf == 0.0

    def test_win_rate(self, trades_with_pnl):
        """Win rate from explicit pnl trades."""
        wr = MetricsCalculator.win_rate(trades_with_pnl)
        # 3 wins out of 5 trades
        assert wr == pytest.approx(3 / 5)

    def test_win_rate_empty(self):
        """No trades yields zero win rate."""
        wr = MetricsCalculator.win_rate([])
        assert wr == 0.0

    def test_calculate_full_metrics(self, sample_trades):
        """Test full metrics calculation pipeline."""
        pnl_series = [0, 5, 10, 8, 3, 12, 15]
        metrics = MetricsCalculator.calculate(
            pnl_series=pnl_series,
            trades=sample_trades,
            total_fees=0.02,
        )
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_pnl == 15.0  # last element of pnl_series
        assert metrics.total_fees == 0.02
        assert metrics.net_pnl == pytest.approx(15.0 - 0.02)
        assert metrics.trade_count == len(sample_trades)
        assert metrics.max_drawdown > 0  # there is a drawdown from 10 to 3

    def test_calculate_metrics_empty(self):
        """Calculate with empty inputs."""
        metrics = MetricsCalculator.calculate([], [])
        assert metrics.total_pnl == 0.0
        assert metrics.trade_count == 0

    def test_trade_pnls_with_explicit_pnl(self, trades_with_pnl):
        """Trade PnL extraction from explicit pnl field."""
        pnls = MetricsCalculator._trade_pnls(trades_with_pnl)
        assert pnls == [10.0, -3.0, 5.0, -1.0, 8.0]

    def test_trade_pnls_from_pairs(self, sample_trades):
        """Trade PnL pairing from buy/sell sides."""
        pnls = MetricsCalculator._trade_pnls(sample_trades)
        assert len(pnls) == 2  # 2 pairs from 4 trades

    def test_avg_trade_duration(self, sample_trades):
        """Average trade duration from paired buy/sell timestamps."""
        dur = MetricsCalculator._avg_trade_duration_ms(sample_trades)
        # Pair 1: 3_000_000 - 1_000_000 = 2ms
        # Pair 2: 8_000_000 - 5_000_000 = 3ms
        assert dur == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# TestReportGenerator
# ---------------------------------------------------------------------------


class TestReportGenerator:
    def test_equity_curve_svg(self, tmp_path):
        """Equity curve SVG contains expected elements."""
        gen = ReportGenerator(output_dir=str(tmp_path / "reports"))
        svg = gen._equity_curve_svg([0, 10, 5, 15, 20])
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "polyline" in svg

    def test_equity_curve_svg_empty(self, tmp_path):
        """Empty data produces fallback SVG."""
        gen = ReportGenerator(output_dir=str(tmp_path / "reports"))
        svg = gen._equity_curve_svg([])
        assert "<svg" in svg
        assert "No data" in svg

    def test_drawdown_svg(self, tmp_path):
        """Drawdown SVG contains expected elements."""
        gen = ReportGenerator(output_dir=str(tmp_path / "reports"))
        svg = gen._drawdown_svg([0, 10, 5, 15, 20])
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "polygon" in svg

    def test_generate_single_creates_file(self, tmp_path, sample_result):
        """Single report generates an HTML file."""
        gen = ReportGenerator(output_dir=str(tmp_path / "reports"))
        metrics = MetricsCalculator.calculate(
            sample_result.pnl_series,
            sample_result.trades,
            sample_result.total_fees,
        )
        path = gen.generate_single(sample_result, metrics, name="test_bt")
        assert path.exists()
        assert path.suffix == ".html"
        content = path.read_text()
        assert "Backtest Report" in content
        assert "Equity Curve" in content
        assert "<table" in content

    def test_generate_sweep_creates_file(self, tmp_path, orchestrator):
        """Sweep report generates an HTML file."""
        results = orchestrator.sweep_params(
            BacktestParams(),
            sweep={"spread_bps": [3.0, 5.0]},
            num_events=200,
        )
        gen = ReportGenerator(output_dir=str(tmp_path / "reports"))
        path = gen.generate_sweep(results, name="test_sweep")
        assert path.exists()
        content = path.read_text()
        assert "Sweep Report" in content
        assert "Sharpe" in content

    def test_metrics_html(self, tmp_path):
        """Metrics table contains expected rows."""
        gen = ReportGenerator(output_dir=str(tmp_path / "reports"))
        metrics = PerformanceMetrics(
            total_pnl=100.0,
            total_return_pct=5.0,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=20.0,
            max_drawdown_pct=10.0,
            profit_factor=2.0,
            win_rate=0.6,
            avg_win=10.0,
            avg_loss=-5.0,
            trade_count=50,
            avg_trade_duration_ms=100.0,
            total_fees=2.0,
            net_pnl=98.0,
        )
        html = gen._metrics_html(metrics)
        assert "Total PnL" in html
        assert "Sharpe Ratio" in html
        assert "Win Rate" in html
        assert "<table" in html


# ---------------------------------------------------------------------------
# TestCompareResults
# ---------------------------------------------------------------------------


class TestCompareResults:
    def test_compare_produces_dataframe(self, orchestrator):
        """compare_results returns a polars DataFrame with correct shape."""
        results = orchestrator.sweep_params(
            BacktestParams(),
            sweep={"spread_bps": [3.0, 5.0]},
            num_events=200,
        )
        df = compare_results(results)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert "label" in df.columns
        assert "total_pnl" in df.columns
        assert "sharpe_ratio" in df.columns

    def test_compare_with_labels(self, orchestrator):
        """Custom labels are applied correctly."""
        results = orchestrator.sweep_params(
            BacktestParams(),
            sweep={"spread_bps": [3.0, 5.0]},
            num_events=200,
        )
        df = compare_results(results, labels=["narrow", "wide"])
        labels = df["label"].to_list()
        assert labels == ["narrow", "wide"]

    def test_compare_default_labels(self, orchestrator):
        """Default labels are run_0, run_1, etc."""
        results = orchestrator.sweep_params(
            BacktestParams(),
            sweep={"spread_bps": [3.0, 5.0, 10.0]},
            num_events=200,
        )
        df = compare_results(results)
        labels = df["label"].to_list()
        assert labels == ["run_0", "run_1", "run_2"]

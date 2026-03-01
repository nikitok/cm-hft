"""Tests for walk-forward optimization and latency sensitivity analysis."""

import json
from pathlib import Path

from backtest.engine.latency_analysis import (
    LatencyAnalyzer,
    LatencyPoint,
    LatencySensitivityResult,
)
from backtest.engine.walk_forward import (
    WalkForwardOptimizer,
    WalkForwardResult,
    WalkForwardWindow,
)

# ---------------------------------------------------------------------------
# Walk-Forward tests
# ---------------------------------------------------------------------------


class TestWalkForward:
    """Tests for WalkForwardOptimizer and related dataclasses."""

    def test_create_windows(self) -> None:
        """Windows are created with correct boundaries and non-overlapping OOS."""
        wf = WalkForwardOptimizer(total_events=20000)
        windows = wf.create_windows(is_size=5000, oos_size=2000)
        assert len(windows) >= 2
        # Verify non-overlapping OOS periods
        for i in range(1, len(windows)):
            assert windows[i].oos_start_event >= windows[i - 1].oos_end_event

    def test_create_windows_boundaries(self) -> None:
        """Each window's IS/OOS boundaries are contiguous."""
        wf = WalkForwardOptimizer(total_events=10000)
        windows = wf.create_windows(is_size=3000, oos_size=1000)
        for w in windows:
            assert w.is_end_event == w.oos_start_event
            assert w.oos_end_event - w.oos_start_event == 1000
            assert w.is_end_event - w.is_start_event == 3000

    def test_create_windows_custom_step(self) -> None:
        """Custom step_size creates overlapping IS windows."""
        wf = WalkForwardOptimizer(total_events=10000)
        windows = wf.create_windows(is_size=3000, oos_size=1000, step_size=500)
        # With a small step, we should get many windows
        assert len(windows) > 5
        # Consecutive windows should start 500 events apart
        for i in range(1, len(windows)):
            assert windows[i].is_start_event - windows[i - 1].is_start_event == 500

    def test_create_windows_exact_fit(self) -> None:
        """Windows are created correctly when total events exactly fits."""
        wf = WalkForwardOptimizer(total_events=7000)
        windows = wf.create_windows(is_size=5000, oos_size=2000)
        assert len(windows) == 1
        assert windows[0].is_start_event == 0
        assert windows[0].oos_end_event == 7000

    def test_create_windows_too_small(self) -> None:
        """No windows when total events is too small."""
        wf = WalkForwardOptimizer(total_events=1000)
        windows = wf.create_windows(is_size=5000, oos_size=2000)
        assert len(windows) == 0

    def test_optimize_runs(self) -> None:
        """Optimization completes without error and returns valid results."""
        wf = WalkForwardOptimizer(total_events=5000)
        result = wf.optimize(
            "market_making",
            param_grid={"spread_bps": [3.0, 5.0]},
            is_size=2000,
            oos_size=1000,
        )
        assert len(result.windows) >= 1
        assert result.overfitting_ratio is not None
        # Each window should have best_params set
        for w in result.windows:
            assert w.best_params is not None
            assert "spread_bps" in w.best_params

    def test_optimize_multiple_params(self) -> None:
        """Optimization works with multi-dimensional param grids."""
        wf = WalkForwardOptimizer(total_events=5000)
        result = wf.optimize(
            "market_making",
            param_grid={"spread_bps": [3.0, 5.0], "order_size": [0.001, 0.002]},
            is_size=2000,
            oos_size=1000,
        )
        assert len(result.windows) >= 1
        for w in result.windows:
            assert w.best_params is not None
            assert "spread_bps" in w.best_params
            assert "order_size" in w.best_params

    def test_overfitting_ratio(self) -> None:
        """Overfitting ratio is computed correctly from window Sharpe ratios."""
        windows = [
            WalkForwardWindow(0, 0, 100, 100, 200, is_sharpe=2.0, oos_sharpe=1.0),
            WalkForwardWindow(1, 200, 300, 300, 400, is_sharpe=3.0, oos_sharpe=1.5),
        ]
        result = WalkForwardResult(windows=windows, param_grid={}, strategy="test")
        # avg IS = 2.5, avg OOS = 1.25, ratio = 0.5
        assert abs(result.overfitting_ratio - 0.5) < 1e-6

    def test_overfitting_ratio_zero_is(self) -> None:
        """Overfitting ratio is 0 when IS Sharpe is zero."""
        windows = [
            WalkForwardWindow(0, 0, 100, 100, 200, is_sharpe=0.0, oos_sharpe=1.0),
        ]
        result = WalkForwardResult(windows=windows, param_grid={}, strategy="test")
        assert result.overfitting_ratio == 0.0

    def test_walk_forward_result_summary(self) -> None:
        """Summary returns all expected keys with correct types."""
        windows = [
            WalkForwardWindow(
                0,
                0,
                100,
                100,
                200,
                best_params={"spread_bps": 5.0},
                is_sharpe=1.5,
                oos_sharpe=0.8,
                is_pnl=100.0,
                oos_pnl=50.0,
            ),
        ]
        result = WalkForwardResult(
            windows=windows, param_grid={"spread_bps": [5.0]}, strategy="test"
        )
        summary = result.summary()
        assert "num_windows" in summary
        assert "avg_is_sharpe" in summary
        assert "avg_oos_sharpe" in summary
        assert "overfitting_ratio" in summary
        assert "total_oos_pnl" in summary
        assert summary["num_windows"] == 1
        assert summary["total_oos_pnl"] == 50.0

    def test_walk_forward_result_to_json(self) -> None:
        """Result serializes to valid JSON."""
        windows = [
            WalkForwardWindow(
                0,
                0,
                100,
                100,
                200,
                best_params={"spread_bps": 5.0},
                is_sharpe=1.5,
                oos_sharpe=0.8,
            ),
        ]
        result = WalkForwardResult(
            windows=windows, param_grid={"spread_bps": [5.0]}, strategy="test"
        )
        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["strategy"] == "test"
        assert len(parsed["windows"]) == 1
        assert "summary" in parsed

    def test_walk_forward_result_save(self, tmp_path: Path) -> None:
        """Result saves to a JSON file on disk."""
        windows = [
            WalkForwardWindow(0, 0, 100, 100, 200, best_params={"a": 1}),
        ]
        result = WalkForwardResult(windows=windows, param_grid={"a": [1]}, strategy="test")
        out = tmp_path / "wf_result.json"
        result.save(out)
        assert out.exists()
        parsed = json.loads(out.read_text())
        assert parsed["strategy"] == "test"

    def test_total_oos_pnl(self) -> None:
        """Total OOS PnL sums across windows."""
        windows = [
            WalkForwardWindow(0, 0, 100, 100, 200, oos_pnl=10.0),
            WalkForwardWindow(1, 200, 300, 300, 400, oos_pnl=20.0),
            WalkForwardWindow(2, 400, 500, 500, 600, oos_pnl=-5.0),
        ]
        result = WalkForwardResult(windows=windows, param_grid={}, strategy="test")
        assert result.total_oos_pnl == 25.0

    def test_empty_windows(self) -> None:
        """Result handles empty window list gracefully."""
        result = WalkForwardResult(windows=[], param_grid={}, strategy="test")
        assert result.avg_is_sharpe == 0.0
        assert result.avg_oos_sharpe == 0.0
        assert result.overfitting_ratio == 0.0
        assert result.total_oos_pnl == 0.0


# ---------------------------------------------------------------------------
# Latency Analysis tests
# ---------------------------------------------------------------------------


class TestLatencyAnalysis:
    """Tests for LatencyAnalyzer and related dataclasses."""

    def test_analyze_runs(self) -> None:
        """Analysis completes and returns correct number of points."""
        la = LatencyAnalyzer(num_events=500)
        result = la.analyze("market_making", {"spread_bps": 5.0}, latencies_ms=[0, 1, 5])
        assert len(result.points) == 3
        assert result.points[0].latency_ms == 0
        assert result.points[1].latency_ms == 1
        assert result.points[2].latency_ms == 5

    def test_analyze_latency_ns_conversion(self) -> None:
        """Latency ms is correctly converted to ns in each point."""
        la = LatencyAnalyzer(num_events=500)
        result = la.analyze("market_making", {"spread_bps": 5.0}, latencies_ms=[0, 0.5, 2])
        assert result.points[0].latency_ns == 0
        assert result.points[1].latency_ns == 500_000
        assert result.points[2].latency_ns == 2_000_000

    def test_analyze_default_latencies(self) -> None:
        """Default latency list is used when none provided."""
        la = LatencyAnalyzer(num_events=500)
        result = la.analyze("market_making", {"spread_bps": 5.0})
        assert len(result.points) == len(LatencyAnalyzer.DEFAULT_LATENCIES_MS)

    def test_sensitivity_coefficient(self) -> None:
        """Sensitivity coefficient is computed for a simple linear case."""
        points = [
            LatencyPoint(
                latency_ms=0,
                latency_ns=0,
                total_pnl=100.0,
                sharpe_ratio=1.0,
                fill_count=10,
                trade_count=5,
            ),
            LatencyPoint(
                latency_ms=5,
                latency_ns=5_000_000,
                total_pnl=50.0,
                sharpe_ratio=0.5,
                fill_count=8,
                trade_count=4,
            ),
            LatencyPoint(
                latency_ms=10,
                latency_ns=10_000_000,
                total_pnl=0.0,
                sharpe_ratio=0.0,
                fill_count=6,
                trade_count=3,
            ),
        ]
        result = LatencySensitivityResult(strategy="test", params={}, points=points)
        # Linear decrease: 100 -> 50 -> 0 over 0 -> 5 -> 10ms. Slope = -10.0
        assert abs(result.sensitivity_coefficient - (-10.0)) < 0.01

    def test_sensitivity_coefficient_single_point(self) -> None:
        """Sensitivity coefficient is 0 with fewer than 2 points."""
        points = [
            LatencyPoint(
                latency_ms=0,
                latency_ns=0,
                total_pnl=100.0,
                sharpe_ratio=1.0,
                fill_count=10,
                trade_count=5,
            ),
        ]
        result = LatencySensitivityResult(strategy="test", params={}, points=points)
        assert result.sensitivity_coefficient == 0.0

    def test_is_latency_sensitive_true(self) -> None:
        """Strategy flagged as latency-sensitive when PnL drops >50% at 10ms."""
        points = [
            LatencyPoint(
                latency_ms=0,
                latency_ns=0,
                total_pnl=100.0,
                sharpe_ratio=1.0,
                fill_count=10,
                trade_count=5,
            ),
            LatencyPoint(
                latency_ms=10,
                latency_ns=10_000_000,
                total_pnl=30.0,
                sharpe_ratio=0.3,
                fill_count=6,
                trade_count=3,
            ),
        ]
        result = LatencySensitivityResult(strategy="test", params={}, points=points)
        # 30/100 = 0.3 < 0.5 -> sensitive
        assert result.is_latency_sensitive is True

    def test_is_latency_sensitive_false(self) -> None:
        """Strategy not flagged when PnL only drops slightly."""
        points = [
            LatencyPoint(
                latency_ms=0,
                latency_ns=0,
                total_pnl=100.0,
                sharpe_ratio=1.0,
                fill_count=10,
                trade_count=5,
            ),
            LatencyPoint(
                latency_ms=10,
                latency_ns=10_000_000,
                total_pnl=80.0,
                sharpe_ratio=0.8,
                fill_count=9,
                trade_count=5,
            ),
        ]
        result = LatencySensitivityResult(strategy="test", params={}, points=points)
        # 80/100 = 0.8 >= 0.5 -> not sensitive
        assert result.is_latency_sensitive is False

    def test_is_latency_sensitive_missing_points(self) -> None:
        """Not flagged as sensitive when 0ms or 10ms point is missing."""
        points = [
            LatencyPoint(
                latency_ms=1,
                latency_ns=1_000_000,
                total_pnl=100.0,
                sharpe_ratio=1.0,
                fill_count=10,
                trade_count=5,
            ),
            LatencyPoint(
                latency_ms=5,
                latency_ns=5_000_000,
                total_pnl=50.0,
                sharpe_ratio=0.5,
                fill_count=8,
                trade_count=4,
            ),
        ]
        result = LatencySensitivityResult(strategy="test", params={}, points=points)
        assert result.is_latency_sensitive is False

    def test_summary_has_expected_keys(self) -> None:
        """Summary dict contains all expected fields."""
        points = [
            LatencyPoint(
                latency_ms=0,
                latency_ns=0,
                total_pnl=100.0,
                sharpe_ratio=1.0,
                fill_count=10,
                trade_count=5,
            ),
        ]
        result = LatencySensitivityResult(strategy="test", params={"a": 1}, points=points)
        summary = result.summary()
        assert "strategy" in summary
        assert "num_points" in summary
        assert "sensitivity_coefficient" in summary
        assert "is_latency_sensitive" in summary
        assert "points" in summary
        assert summary["num_points"] == 1

    def test_save(self, tmp_path: Path) -> None:
        """Result saves to JSON file."""
        points = [
            LatencyPoint(
                latency_ms=0,
                latency_ns=0,
                total_pnl=100.0,
                sharpe_ratio=1.0,
                fill_count=10,
                trade_count=5,
            ),
        ]
        result = LatencySensitivityResult(strategy="test", params={}, points=points)
        out = tmp_path / "latency_result.json"
        result.save(out)
        assert out.exists()
        parsed = json.loads(out.read_text())
        assert parsed["strategy"] == "test"
        assert len(parsed["points"]) == 1

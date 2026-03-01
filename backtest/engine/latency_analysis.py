"""Latency sensitivity analysis for trading strategies.

Runs the same strategy with different added latencies to measure
how sensitive the strategy's PnL is to execution speed.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LatencyPoint:
    """Result at a specific latency level."""

    latency_ms: float
    latency_ns: int
    total_pnl: float
    sharpe_ratio: float
    fill_count: int
    trade_count: int


@dataclass
class LatencySensitivityResult:
    """Complete latency sensitivity analysis result."""

    strategy: str
    params: dict
    points: list[LatencyPoint]

    @property
    def sensitivity_coefficient(self) -> float:
        """Linear regression slope of PnL vs latency.

        Negative = PnL decreases with latency (latency-sensitive).
        Magnitude indicates how much PnL changes per ms of latency.
        """
        if len(self.points) < 2:
            return 0.0
        x = np.array([p.latency_ms for p in self.points])
        y = np.array([p.total_pnl for p in self.points])
        if np.std(x) == 0:
            return 0.0
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    @property
    def is_latency_sensitive(self) -> bool:
        """Strategy is considered latency-sensitive if PnL drops >50% at 10ms vs 0ms."""
        pnl_0 = next((p.total_pnl for p in self.points if p.latency_ms == 0), None)
        pnl_10 = next((p.total_pnl for p in self.points if p.latency_ms == 10), None)
        if pnl_0 is None or pnl_10 is None or pnl_0 == 0:
            return False
        return abs(pnl_10 / pnl_0) < 0.5

    def summary(self) -> dict:
        """Return a summary dict of the latency analysis."""
        return {
            "strategy": self.strategy,
            "num_points": len(self.points),
            "sensitivity_coefficient": round(self.sensitivity_coefficient, 4),
            "is_latency_sensitive": self.is_latency_sensitive,
            "min_pnl": round(min(p.total_pnl for p in self.points), 2) if self.points else 0.0,
            "max_pnl": round(max(p.total_pnl for p in self.points), 2) if self.points else 0.0,
            "points": [
                {
                    "latency_ms": p.latency_ms,
                    "total_pnl": round(p.total_pnl, 2),
                    "sharpe_ratio": round(p.sharpe_ratio, 4),
                    "fill_count": p.fill_count,
                    "trade_count": p.trade_count,
                }
                for p in self.points
            ],
        }

    def save(self, path: Path) -> None:
        """Save the analysis result to a JSON file.

        Args:
            path: File path to write the JSON output.
        """
        path.write_text(json.dumps(self.summary(), indent=2))


class LatencyAnalyzer:
    """Runs latency sensitivity analysis.

    Executes a strategy at multiple latency levels and records how
    performance metrics change as a function of execution delay.
    """

    DEFAULT_LATENCIES_MS = [0, 0.5, 1, 2, 5, 10, 20, 50]

    def __init__(
        self,
        num_events: int = 10000,
        initial_price: float = 50000.0,
        volatility: float = 10.0,
    ):
        """Initialize the analyzer.

        Args:
            num_events: Number of simulated events per run.
            initial_price: Starting price for synthetic data.
            volatility: Price volatility for synthetic data.
        """
        self.num_events = num_events
        self.initial_price = initial_price
        self.volatility = volatility

    def analyze(
        self,
        strategy: str,
        params: dict,
        latencies_ms: list[float] | None = None,
    ) -> LatencySensitivityResult:
        """Run strategy at each latency level and collect results.

        Args:
            strategy: Strategy name to test.
            params: Strategy parameters.
            latencies_ms: List of latencies in milliseconds to test.
                Defaults to DEFAULT_LATENCIES_MS.

        Returns:
            LatencySensitivityResult with a LatencyPoint per latency level.
        """
        from backtest.engine.analytics import MetricsCalculator
        from backtest.engine.orchestrator import BacktestOrchestrator, BacktestParams

        if latencies_ms is None:
            latencies_ms = self.DEFAULT_LATENCIES_MS

        orch = BacktestOrchestrator()
        points = []

        for lat_ms in latencies_ms:
            lat_ns = int(lat_ms * 1_000_000)
            bp = BacktestParams(strategy=strategy, params=params, latency_ns=lat_ns)
            result = orch.run_single(
                bp,
                num_events=self.num_events,
                initial_price=self.initial_price,
                volatility=self.volatility,
            )

            sharpe = 0.0
            if len(result.pnl_series) > 1:
                returns = np.diff(result.pnl_series)
                sharpe = MetricsCalculator.sharpe_ratio(returns)

            points.append(
                LatencyPoint(
                    latency_ms=lat_ms,
                    latency_ns=lat_ns,
                    total_pnl=result.total_pnl,
                    sharpe_ratio=sharpe,
                    fill_count=result.fill_count,
                    trade_count=result.trade_count,
                )
            )
            logger.info("Latency %.1fms: PnL=%.2f, Sharpe=%.2f", lat_ms, result.total_pnl, sharpe)

        return LatencySensitivityResult(strategy=strategy, params=params, points=points)

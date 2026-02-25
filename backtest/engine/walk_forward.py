"""Walk-forward optimization for strategy parameter validation.

Implements rolling in-sample/out-of-sample splits to detect overfitting
and validate strategy robustness across different time periods.
"""

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """A single in-sample/out-of-sample window."""

    window_id: int
    is_start_event: int  # start index of IS period
    is_end_event: int  # end index of IS period
    oos_start_event: int  # start index of OOS period
    oos_end_event: int  # end index of OOS period
    best_params: dict[str, Any] | None = None
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    is_pnl: float = 0.0
    oos_pnl: float = 0.0


@dataclass
class WalkForwardResult:
    """Result of a walk-forward optimization run."""

    windows: list[WalkForwardWindow]
    param_grid: dict[str, list]
    strategy: str

    @property
    def avg_is_sharpe(self) -> float:
        """Average in-sample Sharpe ratio across all windows."""
        return float(np.mean([w.is_sharpe for w in self.windows])) if self.windows else 0.0

    @property
    def avg_oos_sharpe(self) -> float:
        """Average out-of-sample Sharpe ratio across all windows."""
        return float(np.mean([w.oos_sharpe for w in self.windows])) if self.windows else 0.0

    @property
    def overfitting_ratio(self) -> float:
        """Ratio of OOS to IS Sharpe. < 0.5 suggests overfitting."""
        if self.avg_is_sharpe == 0:
            return 0.0
        return self.avg_oos_sharpe / self.avg_is_sharpe

    @property
    def total_oos_pnl(self) -> float:
        """Sum of out-of-sample PnL across all windows."""
        return sum(w.oos_pnl for w in self.windows)

    def summary(self) -> dict:
        """Return a summary dict of the walk-forward results."""
        return {
            "num_windows": len(self.windows),
            "avg_is_sharpe": round(self.avg_is_sharpe, 3),
            "avg_oos_sharpe": round(self.avg_oos_sharpe, 3),
            "overfitting_ratio": round(self.overfitting_ratio, 3),
            "total_oos_pnl": round(self.total_oos_pnl, 2),
        }

    def to_json(self) -> str:
        """Serialize the result to a JSON string."""
        data = {
            "strategy": self.strategy,
            "param_grid": self.param_grid,
            "summary": self.summary(),
            "windows": [
                {
                    "window_id": w.window_id,
                    "is_start_event": w.is_start_event,
                    "is_end_event": w.is_end_event,
                    "oos_start_event": w.oos_start_event,
                    "oos_end_event": w.oos_end_event,
                    "best_params": w.best_params,
                    "is_sharpe": round(w.is_sharpe, 4),
                    "oos_sharpe": round(w.oos_sharpe, 4),
                    "is_pnl": round(w.is_pnl, 2),
                    "oos_pnl": round(w.oos_pnl, 2),
                }
                for w in self.windows
            ],
        }
        return json.dumps(data, indent=2)

    def save(self, path: Path) -> None:
        """Save the result to a JSON file.

        Args:
            path: File path to write the JSON output.
        """
        path.write_text(self.to_json())


class WalkForwardOptimizer:
    """Performs walk-forward optimization on strategy parameters.

    Creates rolling in-sample/out-of-sample splits, optimizes parameters
    on the IS period, then validates on the OOS period to detect overfitting.
    """

    def __init__(
        self,
        total_events: int = 10000,
        initial_price: float = 50000.0,
        volatility: float = 10.0,
    ):
        """Initialize the optimizer.

        Args:
            total_events: Total number of events in the dataset.
            initial_price: Starting price for synthetic data.
            volatility: Price volatility for synthetic data.
        """
        self.total_events = total_events
        self.initial_price = initial_price
        self.volatility = volatility

    def create_windows(
        self,
        is_size: int,
        oos_size: int,
        step_size: int | None = None,
    ) -> list[WalkForwardWindow]:
        """Create IS/OOS window splits.

        Args:
            is_size: In-sample window size (number of events).
            oos_size: Out-of-sample window size.
            step_size: Step between windows (default = oos_size for non-overlapping OOS).

        Returns:
            List of WalkForwardWindow with event index ranges.
        """
        if step_size is None:
            step_size = oos_size

        windows = []
        start = 0
        window_id = 0
        while start + is_size + oos_size <= self.total_events:
            windows.append(
                WalkForwardWindow(
                    window_id=window_id,
                    is_start_event=start,
                    is_end_event=start + is_size,
                    oos_start_event=start + is_size,
                    oos_end_event=start + is_size + oos_size,
                )
            )
            start += step_size
            window_id += 1
        return windows

    def optimize(
        self,
        strategy: str,
        param_grid: dict[str, list],
        is_size: int = 5000,
        oos_size: int = 2000,
        base_params: dict | None = None,
    ) -> WalkForwardResult:
        """Run walk-forward optimization.

        For each window:
        1. Run all param combinations on IS period.
        2. Select best by Sharpe ratio.
        3. Test best params on OOS period.
        4. Record IS vs OOS performance.

        Args:
            strategy: Strategy name to test.
            param_grid: Dict mapping param names to lists of values to try.
            is_size: In-sample window size in events.
            oos_size: Out-of-sample window size in events.
            base_params: Base parameter dict merged with each grid combination.

        Returns:
            WalkForwardResult with per-window metrics.
        """
        from backtest.engine.analytics import MetricsCalculator
        from backtest.engine.orchestrator import BacktestOrchestrator, BacktestParams

        windows = self.create_windows(is_size, oos_size)
        orch = BacktestOrchestrator()

        for window in windows:
            best_sharpe = -float("inf")
            best_params: dict | None = None
            best_pnl = 0.0

            keys = list(param_grid.keys())
            values = list(param_grid.values())

            for combo in itertools.product(*values):
                params_dict = {**(base_params or {}), **dict(zip(keys, combo))}
                bp = BacktestParams(strategy=strategy, params=params_dict)
                result = orch.run_single(
                    bp, num_events=window.is_end_event - window.is_start_event
                )

                if len(result.pnl_series) > 1:
                    returns = np.diff(result.pnl_series)
                    sharpe = MetricsCalculator.sharpe_ratio(returns)
                else:
                    sharpe = 0.0

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params_dict
                    best_pnl = result.total_pnl

            window.best_params = best_params
            window.is_sharpe = best_sharpe
            window.is_pnl = best_pnl

            # Test on OOS with best params
            if best_params:
                bp = BacktestParams(strategy=strategy, params=best_params)
                oos_result = orch.run_single(
                    bp, num_events=window.oos_end_event - window.oos_start_event
                )

                if len(oos_result.pnl_series) > 1:
                    oos_returns = np.diff(oos_result.pnl_series)
                    window.oos_sharpe = MetricsCalculator.sharpe_ratio(oos_returns)
                window.oos_pnl = oos_result.total_pnl

            logger.info(
                "Window %d: IS Sharpe=%.3f, OOS Sharpe=%.3f, best_params=%s",
                window.window_id,
                window.is_sharpe,
                window.oos_sharpe,
                window.best_params,
            )

        return WalkForwardResult(windows=windows, param_grid=param_grid, strategy=strategy)

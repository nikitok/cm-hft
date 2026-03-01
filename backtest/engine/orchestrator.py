"""Backtesting orchestrator for running and managing backtests.

Provides high-level API for running single backtests, parameter sweeps,
and managing backtest results. Uses the cm_pybridge Rust module when available,
falls back to a pure Python simulation for testing.
"""

import itertools
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class BacktestParams:
    """Parameters for a single backtest run.

    Attributes:
        strategy: Strategy name to run (e.g. "market_making").
        params: Strategy-specific parameters dict.
        maker_fee: Maker fee rate (negative = rebate).
        taker_fee: Taker fee rate.
        latency_ns: Simulated latency in nanoseconds.
    """

    strategy: str = "market_making"
    params: dict[str, Any] = field(
        default_factory=lambda: {
            "spread_bps": 5.0,
            "order_size": 0.001,
            "num_levels": 1,
            "reprice_threshold_bps": 2.0,
            "skew_factor": 0.5,
        }
    )
    maker_fee: float = -0.0001
    taker_fee: float = 0.0004
    latency_ns: int = 1_000_000


@dataclass
class BacktestResult:
    """Result of a single backtest run.

    Attributes:
        params: The parameters used for this backtest.
        total_pnl: Total profit and loss including unrealized.
        total_fees: Total fees paid.
        trade_count: Number of trades executed.
        fill_count: Number of order fills.
        max_position: Maximum absolute position held.
        pnl_series: Time series of mark-to-market PnL.
        timestamp_series: Timestamps corresponding to pnl_series (nanoseconds).
        trades: List of trade dicts with timestamp_ns, side, price, quantity, fee.
        duration_seconds: Wall-clock time of the backtest run.
    """

    params: BacktestParams
    total_pnl: float
    total_fees: float
    trade_count: int
    fill_count: int
    max_position: float
    pnl_series: list[float]
    timestamp_series: list[int]
    trades: list[dict]
    duration_seconds: float

    def to_dict(self) -> dict:
        """Convert result to a plain dictionary for serialization.

        Returns:
            Dictionary with all result fields, params expanded inline.
        """
        return {
            "strategy": self.params.strategy,
            "params": self.params.params,
            "maker_fee": self.params.maker_fee,
            "taker_fee": self.params.taker_fee,
            "latency_ns": self.params.latency_ns,
            "total_pnl": self.total_pnl,
            "total_fees": self.total_fees,
            "trade_count": self.trade_count,
            "fill_count": self.fill_count,
            "max_position": self.max_position,
            "duration_seconds": self.duration_seconds,
        }

    def save(self, path: Path) -> None:
        """Save result to Parquet file.

        Saves the PnL series and trade list as separate Parquet files,
        plus a JSON metadata sidecar.

        Args:
            path: Directory path to save results into.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save PnL series
        pnl_df = pl.DataFrame(
            {
                "timestamp_ns": self.timestamp_series,
                "pnl": self.pnl_series,
            }
        )
        pnl_df.write_parquet(path / "pnl_series.parquet")

        # Save trades
        if self.trades:
            trades_df = pl.DataFrame(self.trades)
            trades_df.write_parquet(path / "trades.parquet")

        # Save metadata
        meta = self.to_dict()
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)


class BacktestOrchestrator:
    """High-level API for running backtests.

    Attributes:
        output_dir: Directory for saving backtest results.
    """

    def __init__(self, output_dir: str = "backtest_results") -> None:
        """Initialize orchestrator.

        Args:
            output_dir: Path for storing backtest output files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_single(
        self,
        params: BacktestParams,
        num_events: int = 10000,
        initial_price: float = 50000.0,
        volatility: float = 10.0,
    ) -> BacktestResult:
        """Run a single backtest.

        Tries to use cm_pybridge Rust module. Falls back to pure Python sim if not available.

        Args:
            params: Backtest parameters.
            num_events: Number of price events to simulate.
            initial_price: Starting price for synthetic data.
            volatility: Price volatility parameter.

        Returns:
            BacktestResult with full simulation output.
        """
        try:
            import cm_pybridge  # noqa: F401

            return self._run_rust(params, num_events, initial_price, volatility)
        except ImportError:
            logger.info("cm_pybridge not available, using pure Python simulation")
            return self._run_python_sim(params, num_events, initial_price, volatility)

    def _run_rust(
        self,
        params: BacktestParams,
        num_events: int,
        initial_price: float,
        volatility: float,
    ) -> BacktestResult:
        """Run backtest using Rust PyO3 module.

        Args:
            params: Backtest parameters.
            num_events: Number of price events.
            initial_price: Starting price.
            volatility: Volatility parameter.

        Returns:
            BacktestResult from the Rust engine.
        """
        import cm_pybridge

        config = cm_pybridge.BacktestConfig(params.strategy)
        config.params = json.dumps(params.params)
        config.maker_fee = params.maker_fee
        config.taker_fee = params.taker_fee
        config.latency_ns = params.latency_ns

        start = time.perf_counter()
        result = cm_pybridge.run_backtest_synthetic(config, num_events, initial_price, volatility)
        duration = time.perf_counter() - start

        # Convert trade records from Rust objects to dicts
        trades = []
        if hasattr(result, "trades"):
            for t in result.trades:
                trades.append(
                    {
                        "timestamp_ns": t.timestamp_ns,
                        "side": t.side,
                        "price": t.price,
                        "quantity": t.quantity,
                        "fee": t.fee,
                    }
                )

        return BacktestResult(
            params=params,
            total_pnl=result.total_pnl,
            total_fees=result.total_fees,
            trade_count=result.trade_count,
            fill_count=result.fill_count,
            max_position=result.max_position,
            pnl_series=list(result.pnl_series),
            timestamp_series=list(result.timestamp_series),
            trades=trades,
            duration_seconds=duration,
        )

    def _run_python_sim(
        self,
        params: BacktestParams,
        num_events: int,
        initial_price: float,
        volatility: float,
    ) -> BacktestResult:
        """Pure Python simulation fallback (simplified).

        Generates synthetic price data and simulates a basic market-making loop.
        NOT as accurate as the Rust sim -- used only for testing the Python layer.

        Args:
            params: Backtest parameters.
            num_events: Number of price events to generate.
            initial_price: Starting mid price.
            volatility: Standard deviation of per-step returns (in price units).

        Returns:
            BacktestResult with simulation output.
        """
        start = time.perf_counter()

        rng = np.random.default_rng(42)
        returns = rng.normal(0, volatility / initial_price, num_events)
        prices = initial_price * np.cumprod(1 + returns)

        spread = params.params.get("spread_bps", 5.0) / 10000
        order_size = params.params.get("order_size", 0.001)

        position = 0.0
        pnl = 0.0
        total_fees = 0.0
        pnl_series: list[float] = []
        timestamp_series: list[int] = []
        trades: list[dict] = []
        fill_count = 0
        max_pos = 0.0

        for i in range(1, num_events):
            mid = float(prices[i])
            prev_mid = float(prices[i - 1])

            # Place orders around previous mid price
            bid = prev_mid * (1 - spread / 2)
            ask = prev_mid * (1 + spread / 2)

            # Fill bid if current price dropped to or below our resting bid
            if mid <= bid and position < 0.1:
                fill_price = bid
                fee = abs(fill_price * order_size * params.maker_fee)
                position += order_size
                pnl -= fill_price * order_size + fee
                total_fees += fee
                fill_count += 1
                trades.append(
                    {
                        "timestamp_ns": i * 1_000_000,
                        "side": "buy",
                        "price": fill_price,
                        "quantity": order_size,
                        "fee": fee,
                    }
                )

            # Fill ask if current price rose to or above our resting ask
            if mid >= ask and position > -0.1:
                fill_price = ask
                fee = abs(fill_price * order_size * params.maker_fee)
                position -= order_size
                pnl += fill_price * order_size - fee
                total_fees += fee
                fill_count += 1
                trades.append(
                    {
                        "timestamp_ns": i * 1_000_000,
                        "side": "sell",
                        "price": fill_price,
                        "quantity": order_size,
                        "fee": fee,
                    }
                )

            # Mark-to-market
            mtm = pnl + position * mid
            pnl_series.append(mtm)
            timestamp_series.append(i * 1_000_000)
            max_pos = max(max_pos, abs(position))

        duration = time.perf_counter() - start

        return BacktestResult(
            params=params,
            total_pnl=pnl + position * float(prices[-1]),
            total_fees=total_fees,
            trade_count=len(trades),
            fill_count=fill_count,
            max_position=max_pos,
            pnl_series=pnl_series,
            timestamp_series=timestamp_series,
            trades=trades,
            duration_seconds=duration,
        )

    def sweep_params(
        self,
        base_params: BacktestParams,
        sweep: dict[str, list],
        num_events: int = 10000,
        max_workers: int = 4,
    ) -> list[BacktestResult]:
        """Run parameter sweep over all combinations (Cartesian product).

        Args:
            base_params: Base parameters; swept keys override params dict entries.
            sweep: Dict mapping param names to lists of values, e.g.
                   {"spread_bps": [3, 5, 7, 10], "order_size": [0.001, 0.005]}.
            num_events: Number of events per backtest.
            max_workers: Max parallel workers (reserved for future parallel impl).

        Returns:
            List of BacktestResult, one per parameter combination.
        """
        keys = list(sweep.keys())
        values = list(sweep.values())
        combos = list(itertools.product(*values))

        results: list[BacktestResult] = []
        for combo in combos:
            p = BacktestParams(
                strategy=base_params.strategy,
                params={**base_params.params, **dict(zip(keys, combo, strict=True))},
                maker_fee=base_params.maker_fee,
                taker_fee=base_params.taker_fee,
                latency_ns=base_params.latency_ns,
            )
            result = self.run_single(p, num_events=num_events)
            results.append(result)
            logger.info(f"Sweep {dict(zip(keys, combo, strict=True))}: PnL={result.total_pnl:.2f}")

        return results

    def save_results(self, results: list[BacktestResult], name: str) -> Path:
        """Save sweep results to Parquet and JSON.

        Creates a summary DataFrame with one row per result containing
        key metrics and parameter values.

        Args:
            results: List of backtest results to save.
            name: Name for the output subdirectory.

        Returns:
            Path to the output directory.
        """
        out_dir = self.output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for i, r in enumerate(results):
            row = r.to_dict()
            row["run_index"] = i
            rows.append(row)

        # Flatten params dict into columns for the summary
        flat_rows = []
        for row in rows:
            flat = {k: v for k, v in row.items() if k != "params"}
            if isinstance(row.get("params"), dict):
                for pk, pv in row["params"].items():
                    flat[f"param_{pk}"] = pv
            flat_rows.append(flat)

        df = pl.DataFrame(flat_rows)
        df.write_parquet(out_dir / "summary.parquet")

        # Also save individual results
        for i, r in enumerate(results):
            r.save(out_dir / f"run_{i:04d}")

        logger.info(f"Saved {len(results)} results to {out_dir}")
        return out_dir

"""Backtesting analytics: metrics, plots, and report generation.

Computes standard trading performance metrics and generates
visual reports for backtest analysis.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Standard trading performance metrics.

    Attributes:
        total_pnl: Gross PnL before fees.
        total_return_pct: Total return as percentage.
        sharpe_ratio: Annualized Sharpe ratio.
        sortino_ratio: Annualized Sortino ratio (downside deviation).
        max_drawdown: Maximum drawdown in absolute terms.
        max_drawdown_pct: Maximum drawdown as percentage of peak.
        profit_factor: Gross profit / gross loss.
        win_rate: Fraction of profitable trades.
        avg_win: Average profit on winning trades.
        avg_loss: Average loss on losing trades.
        trade_count: Total number of trades.
        avg_trade_duration_ms: Mean trade duration in milliseconds.
        total_fees: Total fees paid.
        net_pnl: PnL after fees (total_pnl - total_fees).
    """

    total_pnl: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    profit_factor: float
    win_rate: float
    avg_win: float
    avg_loss: float
    trade_count: int
    avg_trade_duration_ms: float
    total_fees: float
    net_pnl: float


class MetricsCalculator:
    """Calculates performance metrics from backtest results."""

    @staticmethod
    def calculate(
        pnl_series: list[float],
        trades: list[dict],
        total_fees: float = 0.0,
        risk_free_rate: float = 0.0,
        periods_per_year: float = 365 * 24 * 60,
    ) -> PerformanceMetrics:
        """Calculate all performance metrics from a PnL series and trade list.

        Args:
            pnl_series: Time series of mark-to-market PnL values.
            trades: List of trade dicts with keys: timestamp_ns, side, price,
                    quantity, fee. Each trade may also have a 'pnl' key.
            total_fees: Total fees paid during the backtest.
            risk_free_rate: Annual risk-free rate for ratio calculations.
            periods_per_year: Number of observation periods per year for annualization.

        Returns:
            PerformanceMetrics with all computed fields.
        """
        arr = np.array(pnl_series) if pnl_series else np.array([0.0])

        # PnL returns (differences)
        if len(arr) > 1:
            returns = np.diff(arr)
        else:
            returns = np.array([0.0])

        total_pnl = float(arr[-1]) if len(arr) > 0 else 0.0
        initial_value = float(arr[0]) if len(arr) > 0 and arr[0] != 0 else 1.0
        total_return_pct = (total_pnl / abs(initial_value)) * 100 if initial_value != 0 else 0.0

        sharpe = MetricsCalculator.sharpe_ratio(returns, risk_free_rate, periods_per_year)
        sortino = MetricsCalculator.sortino_ratio(returns, risk_free_rate, periods_per_year)
        dd_abs, dd_pct = MetricsCalculator.max_drawdown(pnl_series)
        pf = MetricsCalculator.profit_factor(trades)
        wr = MetricsCalculator.win_rate(trades)

        # Avg win/loss from trade PnL
        trade_pnls = MetricsCalculator._trade_pnls(trades)
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0

        # Average trade duration
        avg_dur = MetricsCalculator._avg_trade_duration_ms(trades)

        return PerformanceMetrics(
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=dd_abs,
            max_drawdown_pct=dd_pct,
            profit_factor=pf,
            win_rate=wr,
            avg_win=avg_win,
            avg_loss=avg_loss,
            trade_count=len(trades),
            avg_trade_duration_ms=avg_dur,
            total_fees=total_fees,
            net_pnl=total_pnl - total_fees,
        )

    @staticmethod
    def _trade_pnls(trades: list[dict]) -> list[float]:
        """Extract per-trade PnL from trade list.

        If trades have a 'pnl' key, use it directly.
        Otherwise, estimate from buy/sell pairs: sell_price - buy_price per unit.

        Args:
            trades: List of trade dicts.

        Returns:
            List of per-trade PnL values.
        """
        # If trades have explicit pnl
        if trades and "pnl" in trades[0]:
            return [t["pnl"] for t in trades]

        # Pair up buys and sells chronologically
        pnls: list[float] = []
        buys: list[dict] = []
        for t in trades:
            if t.get("side") == "buy":
                buys.append(t)
            elif t.get("side") == "sell" and buys:
                buy = buys.pop(0)
                trade_pnl = (t["price"] - buy["price"]) * t["quantity"] - t.get("fee", 0) - buy.get("fee", 0)
                pnls.append(trade_pnl)

        return pnls

    @staticmethod
    def _avg_trade_duration_ms(trades: list[dict]) -> float:
        """Calculate average duration between paired buy/sell trades.

        Args:
            trades: List of trade dicts with timestamp_ns.

        Returns:
            Average duration in milliseconds, or 0 if not enough data.
        """
        buys: list[dict] = []
        durations: list[float] = []
        for t in trades:
            if t.get("side") == "buy":
                buys.append(t)
            elif t.get("side") == "sell" and buys:
                buy = buys.pop(0)
                dur_ns = t.get("timestamp_ns", 0) - buy.get("timestamp_ns", 0)
                durations.append(dur_ns / 1_000_000)  # ns -> ms

        return float(np.mean(durations)) if durations else 0.0

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: float = 252,
    ) -> float:
        """Annualized Sharpe ratio.

        Args:
            returns: Array of period returns.
            risk_free_rate: Annual risk-free rate.
            periods_per_year: Number of periods per year for annualization.

        Returns:
            Annualized Sharpe ratio, or 0.0 if std is zero or no data.
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess = returns - risk_free_rate / periods_per_year
        return float(np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year))

    @staticmethod
    def sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: float = 252,
    ) -> float:
        """Annualized Sortino ratio (downside deviation only).

        Args:
            returns: Array of period returns.
            risk_free_rate: Annual risk-free rate.
            periods_per_year: Number of periods per year for annualization.

        Returns:
            Annualized Sortino ratio. Returns inf if positive mean with no downside,
            0.0 if no data.
        """
        if len(returns) == 0:
            return 0.0
        excess = returns - risk_free_rate / periods_per_year
        downside = returns[returns < 0]
        if len(downside) == 0 or np.std(downside) == 0:
            return float("inf") if np.mean(excess) > 0 else 0.0
        return float(np.mean(excess) / np.std(downside) * np.sqrt(periods_per_year))

    @staticmethod
    def max_drawdown(pnl_series: list[float]) -> tuple[float, float]:
        """Maximum drawdown (absolute and percentage).

        Args:
            pnl_series: Time series of cumulative PnL values.

        Returns:
            Tuple of (max_dd_absolute, max_dd_percentage). Percentage is relative
            to the peak value. Returns (0.0, 0.0) for empty or monotonically
            increasing series.
        """
        if not pnl_series or len(pnl_series) < 2:
            return 0.0, 0.0

        arr = np.array(pnl_series)
        running_max = np.maximum.accumulate(arr)
        drawdowns = running_max - arr

        max_dd = float(np.max(drawdowns))

        # Percentage drawdown relative to peak
        # Avoid division by zero: only compute where peak > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_drawdowns = np.where(running_max > 0, drawdowns / running_max, 0.0)
        max_dd_pct = float(np.max(pct_drawdowns)) * 100

        return max_dd, max_dd_pct

    @staticmethod
    def profit_factor(trades: list[dict]) -> float:
        """Gross profit / gross loss. > 1.0 is profitable.

        Args:
            trades: List of trade dicts. Uses 'pnl' key if available,
                    otherwise pairs buys and sells.

        Returns:
            Profit factor. Returns inf if no losses, 0.0 if no trades or no profits.
        """
        pnls = MetricsCalculator._trade_pnls(trades)
        if not pnls:
            return 0.0

        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def win_rate(trades: list[dict]) -> float:
        """Percentage of profitable trades.

        Args:
            trades: List of trade dicts.

        Returns:
            Win rate as a fraction (0.0 to 1.0). Returns 0.0 if no trades.
        """
        pnls = MetricsCalculator._trade_pnls(trades)
        if not pnls:
            return 0.0
        wins = sum(1 for p in pnls if p > 0)
        return wins / len(pnls)


class ReportGenerator:
    """Generates HTML reports for backtest results.

    Attributes:
        output_dir: Directory where reports are written.
    """

    def __init__(self, output_dir: str = "reports") -> None:
        """Initialize report generator.

        Args:
            output_dir: Path for saving HTML reports.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_single(
        self,
        result: "BacktestResult",  # noqa: F821 — avoid circular import
        metrics: PerformanceMetrics,
        name: str = "backtest",
    ) -> Path:
        """Generate HTML report for a single backtest.

        Includes: metrics table, equity curve, drawdown chart, trade distribution.
        Uses simple HTML + inline CSS (no external dependencies for report viewing).

        Args:
            result: BacktestResult from orchestrator.
            metrics: Computed PerformanceMetrics.
            name: Report file name (without extension).

        Returns:
            Path to the generated HTML file.
        """
        equity_svg = self._equity_curve_svg(result.pnl_series)
        drawdown_svg = self._drawdown_svg(result.pnl_series)
        metrics_table = self._metrics_html(metrics)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Backtest Report: {name}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
  .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  h1 {{ color: #333; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }}
  h2 {{ color: #555; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #ddd; }}
  th {{ background: #f8f8f8; font-weight: 600; }}
  .positive {{ color: #4CAF50; }}
  .negative {{ color: #f44336; }}
  .chart {{ margin: 20px 0; text-align: center; }}
  .params {{ background: #f9f9f9; padding: 15px; border-radius: 4px; font-family: monospace; font-size: 13px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Backtest Report: {name}</h1>

  <h2>Strategy Parameters</h2>
  <div class="params">
    Strategy: {result.params.strategy}<br>
    {self._format_params(result.params.params)}<br>
    Maker Fee: {result.params.maker_fee}, Taker Fee: {result.params.taker_fee}<br>
    Latency: {result.params.latency_ns} ns
  </div>

  <h2>Performance Metrics</h2>
  {metrics_table}

  <h2>Equity Curve</h2>
  <div class="chart">{equity_svg}</div>

  <h2>Drawdown</h2>
  <div class="chart">{drawdown_svg}</div>

  <p style="color: #999; font-size: 12px; text-align: center; margin-top: 40px;">
    Generated by CM.HFT Backtesting Engine | Duration: {result.duration_seconds:.3f}s
  </p>
</div>
</body>
</html>"""
        path = self.output_dir / f"{name}.html"
        path.write_text(html)
        logger.info(f"Report saved to {path}")
        return path

    def generate_sweep(
        self,
        results: list,
        name: str = "sweep",
    ) -> Path:
        """Generate HTML report for a parameter sweep.

        Includes: summary table of all runs sorted by Sharpe, best/worst configs.

        Args:
            results: List of BacktestResult objects.
            name: Report file name (without extension).

        Returns:
            Path to the generated HTML file.
        """
        rows_html = ""
        summaries = []
        for i, r in enumerate(results):
            pnl_arr = np.array(r.pnl_series) if r.pnl_series else np.array([0.0])
            returns = np.diff(pnl_arr) if len(pnl_arr) > 1 else np.array([0.0])
            sharpe = MetricsCalculator.sharpe_ratio(returns)
            dd_abs, _ = MetricsCalculator.max_drawdown(r.pnl_series)
            summaries.append((i, r, sharpe, dd_abs))

        # Sort by Sharpe descending
        summaries.sort(key=lambda x: x[2], reverse=True)

        for i, r, sharpe, dd_abs in summaries:
            pnl_class = "positive" if r.total_pnl >= 0 else "negative"
            param_str = ", ".join(f"{k}={v}" for k, v in r.params.params.items())
            rows_html += f"""<tr>
  <td>{i}</td>
  <td style="font-family:monospace;font-size:12px">{param_str}</td>
  <td class="{pnl_class}">{r.total_pnl:.4f}</td>
  <td>{sharpe:.2f}</td>
  <td>{dd_abs:.4f}</td>
  <td>{r.fill_count}</td>
</tr>
"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Sweep Report: {name}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
  .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  h1 {{ color: #333; border-bottom: 2px solid #FF9800; padding-bottom: 10px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
  th {{ background: #f8f8f8; font-weight: 600; }}
  .positive {{ color: #4CAF50; }}
  .negative {{ color: #f44336; }}
</style>
</head>
<body>
<div class="container">
  <h1>Parameter Sweep Report: {name}</h1>
  <p>Total runs: {len(results)} | Sorted by Sharpe ratio (descending)</p>
  <table>
    <tr><th>#</th><th>Parameters</th><th>PnL</th><th>Sharpe</th><th>Max DD</th><th>Fills</th></tr>
    {rows_html}
  </table>
</div>
</body>
</html>"""
        path = self.output_dir / f"{name}.html"
        path.write_text(html)
        logger.info(f"Sweep report saved to {path}")
        return path

    def _equity_curve_svg(
        self, pnl_series: list[float], width: int = 800, height: int = 300
    ) -> str:
        """Generate inline SVG for equity curve (no external dependencies).

        Args:
            pnl_series: Time series of PnL values.
            width: SVG width in pixels.
            height: SVG height in pixels.

        Returns:
            SVG markup string.
        """
        if not pnl_series or len(pnl_series) < 2:
            return '<svg width="800" height="300"><text x="400" y="150" text-anchor="middle" fill="#999">No data</text></svg>'

        padding = 50
        plot_w = width - 2 * padding
        plot_h = height - 2 * padding

        min_val = min(pnl_series)
        max_val = max(pnl_series)
        val_range = max_val - min_val if max_val != min_val else 1.0

        n = len(pnl_series)
        points = []
        for i, v in enumerate(pnl_series):
            x = padding + (i / (n - 1)) * plot_w
            y = padding + plot_h - ((v - min_val) / val_range) * plot_h
            points.append(f"{x:.1f},{y:.1f}")

        polyline = " ".join(points)

        # Zero line
        zero_y = padding + plot_h - ((0 - min_val) / val_range) * plot_h
        zero_line = ""
        if min_val <= 0 <= max_val:
            zero_line = f'<line x1="{padding}" y1="{zero_y:.1f}" x2="{width - padding}" y2="{zero_y:.1f}" stroke="#ccc" stroke-dasharray="4"/>'

        color = "#4CAF50" if pnl_series[-1] >= 0 else "#f44336"

        return f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{width}" height="{height}" fill="white" rx="4"/>
  {zero_line}
  <polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="1.5"/>
  <text x="{padding}" y="{padding - 10}" font-size="12" fill="#666">PnL: {pnl_series[-1]:.6f}</text>
  <text x="{padding}" y="{height - 10}" font-size="11" fill="#999">0</text>
  <text x="{width - padding}" y="{height - 10}" font-size="11" fill="#999" text-anchor="end">{n} ticks</text>
  <text x="{padding - 5}" y="{padding + 5}" font-size="10" fill="#999" text-anchor="end">{max_val:.6f}</text>
  <text x="{padding - 5}" y="{padding + plot_h}" font-size="10" fill="#999" text-anchor="end">{min_val:.6f}</text>
</svg>"""

    def _drawdown_svg(
        self, pnl_series: list[float], width: int = 800, height: int = 200
    ) -> str:
        """Generate inline SVG for drawdown chart.

        Args:
            pnl_series: Time series of PnL values.
            width: SVG width in pixels.
            height: SVG height in pixels.

        Returns:
            SVG markup string.
        """
        if not pnl_series or len(pnl_series) < 2:
            return f'<svg width="{width}" height="{height}"><text x="400" y="100" text-anchor="middle" fill="#999">No data</text></svg>'

        padding = 50
        plot_w = width - 2 * padding
        plot_h = height - 2 * padding

        arr = np.array(pnl_series)
        running_max = np.maximum.accumulate(arr)
        drawdowns = running_max - arr

        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 1.0
        if max_dd == 0:
            max_dd = 1.0

        n = len(drawdowns)
        points = [f"{padding},{padding}"]  # start at zero
        for i in range(n):
            x = padding + (i / (n - 1)) * plot_w
            y = padding + (float(drawdowns[i]) / max_dd) * plot_h
            points.append(f"{x:.1f},{y:.1f}")
        points.append(f"{width - padding},{padding}")  # close to zero at end

        polyline = " ".join(points)

        return f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{width}" height="{height}" fill="white" rx="4"/>
  <polygon points="{polyline}" fill="rgba(244,67,54,0.2)" stroke="#f44336" stroke-width="1"/>
  <text x="{padding}" y="{padding - 10}" font-size="12" fill="#666">Max Drawdown: {float(np.max(drawdowns)):.6f}</text>
  <text x="{padding - 5}" y="{padding + 5}" font-size="10" fill="#999" text-anchor="end">0</text>
  <text x="{padding - 5}" y="{padding + plot_h}" font-size="10" fill="#999" text-anchor="end">{float(np.max(drawdowns)):.6f}</text>
</svg>"""

    def _metrics_html(self, metrics: PerformanceMetrics) -> str:
        """Generate HTML table of performance metrics.

        Args:
            metrics: Computed performance metrics.

        Returns:
            HTML table string.
        """
        def _fmt(val: float, fmt: str = ".4f", is_pct: bool = False) -> str:
            """Format a value with color class."""
            suffix = "%" if is_pct else ""
            formatted = f"{val:{fmt}}{suffix}"
            if val > 0:
                return f'<span class="positive">{formatted}</span>'
            elif val < 0:
                return f'<span class="negative">{formatted}</span>'
            return formatted

        return f"""<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Total PnL</td><td>{_fmt(metrics.total_pnl)}</td></tr>
  <tr><td>Net PnL (after fees)</td><td>{_fmt(metrics.net_pnl)}</td></tr>
  <tr><td>Total Return</td><td>{_fmt(metrics.total_return_pct, '.2f', True)}</td></tr>
  <tr><td>Sharpe Ratio</td><td>{_fmt(metrics.sharpe_ratio, '.2f')}</td></tr>
  <tr><td>Sortino Ratio</td><td>{metrics.sortino_ratio:.2f}</td></tr>
  <tr><td>Max Drawdown</td><td>{metrics.max_drawdown:.6f}</td></tr>
  <tr><td>Max Drawdown %</td><td>{metrics.max_drawdown_pct:.2f}%</td></tr>
  <tr><td>Profit Factor</td><td>{metrics.profit_factor:.2f}</td></tr>
  <tr><td>Win Rate</td><td>{metrics.win_rate:.1%}</td></tr>
  <tr><td>Avg Win</td><td>{_fmt(metrics.avg_win, '.6f')}</td></tr>
  <tr><td>Avg Loss</td><td>{_fmt(metrics.avg_loss, '.6f')}</td></tr>
  <tr><td>Trade Count</td><td>{metrics.trade_count}</td></tr>
  <tr><td>Avg Trade Duration</td><td>{metrics.avg_trade_duration_ms:.1f} ms</td></tr>
  <tr><td>Total Fees</td><td>{metrics.total_fees:.6f}</td></tr>
</table>"""

    @staticmethod
    def _format_params(params: dict) -> str:
        """Format parameter dict for display.

        Args:
            params: Strategy parameters dict.

        Returns:
            Formatted string of key=value pairs.
        """
        return " | ".join(f"{k}={v}" for k, v in params.items())


def compare_results(
    results: list,
    labels: Optional[list[str]] = None,
) -> pl.DataFrame:
    """Compare multiple backtest results side by side.

    Returns a DataFrame with one row per result and columns for each metric.

    Args:
        results: List of BacktestResult objects.
        labels: Optional labels for each result. Defaults to "run_0", "run_1", etc.

    Returns:
        Polars DataFrame with comparison data.
    """
    if labels is None:
        labels = [f"run_{i}" for i in range(len(results))]

    rows = []
    for i, r in enumerate(results):
        pnl_arr = np.array(r.pnl_series) if r.pnl_series else np.array([0.0])
        returns = np.diff(pnl_arr) if len(pnl_arr) > 1 else np.array([0.0])

        sharpe = MetricsCalculator.sharpe_ratio(returns)
        sortino = MetricsCalculator.sortino_ratio(returns)
        dd_abs, dd_pct = MetricsCalculator.max_drawdown(r.pnl_series)

        rows.append(
            {
                "label": labels[i] if i < len(labels) else f"run_{i}",
                "strategy": r.params.strategy,
                "total_pnl": r.total_pnl,
                "total_fees": r.total_fees,
                "trade_count": r.trade_count,
                "fill_count": r.fill_count,
                "max_position": r.max_position,
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "max_drawdown": dd_abs,
                "max_drawdown_pct": dd_pct,
                "duration_seconds": r.duration_seconds,
            }
        )

    return pl.DataFrame(rows)

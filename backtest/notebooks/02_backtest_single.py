# %% [markdown]
# # Single Backtest Run
# Run a single backtest with fixed parameters and analyze the results.

# %%
from backtest.engine.orchestrator import BacktestOrchestrator, BacktestParams
from backtest.engine.analytics import MetricsCalculator
import numpy as np

# %%
orch = BacktestOrchestrator()
params = BacktestParams(strategy="market_making", params={"spread_bps": 5.0})
result = orch.run_single(params, num_events=10000)

# %% [markdown]
# ## PnL Curve

# %%
print(f"Total PnL: {result.total_pnl:.2f}")
print(f"Trades: {result.trade_count}")
# import matplotlib.pyplot as plt
# plt.plot(result.pnl_series)
# plt.title("Cumulative PnL")
# plt.xlabel("Event")
# plt.ylabel("PnL ($)")

# %% [markdown]
# ## Performance Metrics

# %%
if len(result.pnl_series) > 1:
    returns = np.diff(result.pnl_series)
    sharpe = MetricsCalculator.sharpe_ratio(returns)
    print(f"Sharpe Ratio: {sharpe:.3f}")

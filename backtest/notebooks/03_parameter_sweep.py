# %% [markdown]
# # Parameter Sweep
# Sweep strategy parameters and visualize as a heatmap.

# %%
from backtest.engine.orchestrator import BacktestOrchestrator, BacktestParams
from backtest.engine.analytics import MetricsCalculator
import numpy as np
import itertools

# %%
orch = BacktestOrchestrator()
spread_values = [2.0, 3.0, 5.0, 8.0, 10.0]
results = {}

for spread in spread_values:
    bp = BacktestParams(strategy="market_making", params={"spread_bps": spread})
    r = orch.run_single(bp, num_events=5000)
    sharpe = 0.0
    if len(r.pnl_series) > 1:
        sharpe = MetricsCalculator.sharpe_ratio(np.diff(r.pnl_series))
    results[spread] = {"pnl": r.total_pnl, "sharpe": sharpe}

# %% [markdown]
# ## Results Table

# %%
for spread, metrics in results.items():
    print(f"spread_bps={spread:5.1f}  PnL={metrics['pnl']:8.2f}  Sharpe={metrics['sharpe']:.3f}")

# %% [markdown]
# ## Heatmap (for 2D sweeps)

# %%
# import plotly.express as px
# For 2D sweep, build a matrix and use px.imshow(matrix, x=param1, y=param2)
print("Extend to 2D sweep for heatmap visualization")

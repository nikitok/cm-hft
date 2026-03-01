# %% [markdown]
# # Walk-Forward Optimization
# Run rolling IS/OOS optimization to detect overfitting.

# %%
from backtest.engine.walk_forward import WalkForwardOptimizer

# %%
wf = WalkForwardOptimizer(total_events=10000)
result = wf.optimize(
    strategy="market_making",
    param_grid={"spread_bps": [3.0, 5.0, 8.0]},
    is_size=3000,
    oos_size=1000,
)

# %% [markdown]
# ## Summary

# %%
summary = result.summary()
for key, val in summary.items():
    print(f"{key}: {val}")

# %% [markdown]
# ## Per-Window Results

# %%
for w in result.windows:
    print(
        f"Window {w.window_id}: IS Sharpe={w.is_sharpe:.3f}, OOS Sharpe={w.oos_sharpe:.3f}, "
        f"params={w.best_params}"
    )

# %% [markdown]
# ## Overfitting Check

# %%
ratio = result.overfitting_ratio
print(f"Overfitting ratio (OOS/IS Sharpe): {ratio:.3f}")
if ratio < 0.5:
    print("WARNING: Possible overfitting detected")

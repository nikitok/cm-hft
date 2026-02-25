# %% [markdown]
# # Latency Sensitivity Analysis
# Measure how strategy PnL degrades with increasing execution latency.

# %%
from backtest.engine.latency_analysis import LatencyAnalyzer

# %%
la = LatencyAnalyzer(num_events=5000)
result = la.analyze(
    strategy="market_making",
    params={"spread_bps": 5.0},
    latencies_ms=[0, 0.5, 1, 2, 5, 10, 20],
)

# %% [markdown]
# ## PnL vs Latency

# %%
for p in result.points:
    print(f"Latency {p.latency_ms:5.1f}ms: PnL={p.total_pnl:8.2f}, Sharpe={p.sharpe_ratio:.3f}")

# %% [markdown]
# ## Sensitivity

# %%
print(f"Sensitivity coefficient: {result.sensitivity_coefficient:.4f} $/ms")
print(f"Latency sensitive: {result.is_latency_sensitive}")

# %%
# import matplotlib.pyplot as plt
# plt.plot([p.latency_ms for p in result.points], [p.total_pnl for p in result.points], 'o-')
# plt.xlabel("Latency (ms)")
# plt.ylabel("Total PnL ($)")
# plt.title("PnL vs Execution Latency")
print("Uncomment above for visualization")

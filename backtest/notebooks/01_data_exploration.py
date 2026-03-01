# %% [markdown]
# # Data Exploration
# Load Parquet data and explore price, spread, volume, book depth.

# %%

# Load sample data
# data_path = Path("data/BTCUSDT/binance/")
# df = pl.read_parquet(data_path / "2025-01-15.trades.parquet")

# %% [markdown]
# ## Price Chart

# %%
# df.select("timestamp", "price").plot()  # placeholder
print("Load data above to explore")

# %% [markdown]
# ## Spread Analysis

# %%
# book = pl.read_parquet(data_path / "2025-01-15.book.parquet")
# book = book.with_columns((pl.col("ask_price") - pl.col("bid_price")).alias("spread"))
# book.select("timestamp", "spread").plot()
print("Load book data to analyze spreads")

# %% [markdown]
# ## Volume Distribution

# %%
# df["quantity"].describe()
print("Load trade data to see volume distribution")

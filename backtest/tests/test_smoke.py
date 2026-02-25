"""Smoke tests to verify the Python environment is set up correctly."""


def test_imports():
    """Verify core dependencies are importable."""
    import numpy
    import pandas
    import polars
    import pyarrow

    assert numpy.__version__
    assert pandas.__version__


def test_backtest_package():
    """Verify backtest package is importable."""
    import backtest
    import backtest.engine
    import backtest.data

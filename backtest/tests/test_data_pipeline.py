"""Tests for the data pipeline modules."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from backtest.data.download import BinanceDownloader, BybitDownloader
from backtest.data.export import QuestDBExporter
from backtest.data.validate import DataValidator, ValidationReport, ValidationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_book_df(
    n: int = 100,
    start_us: int = 1_000_000_000_000,
    step_us: int = 500_000,
    crossed: bool = False,
    negative_qty: bool = False,
) -> pl.DataFrame:
    """Create a synthetic book-update DataFrame for testing."""
    timestamps = [datetime.fromtimestamp((start_us + i * step_us) / 1e6, tz=UTC) for i in range(n)]
    bid_prices = [50000.0 + i * 0.1 for i in range(n)]
    ask_prices = [50000.5 + i * 0.1 for i in range(n)]
    if crossed:
        # Make the first entry crossed
        ask_prices[0] = bid_prices[0] - 1.0
    bid_qtys = [0.5] * n
    ask_qtys = [0.5] * n
    if negative_qty:
        bid_qtys[0] = -0.1
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "bid_price": bid_prices,
            "ask_price": ask_prices,
            "bid_qty": bid_qtys,
            "ask_qty": ask_qtys,
        }
    )


def _make_trades_df(
    n: int = 100,
    start_us: int = 1_000_000_000_000,
    step_us: int = 1_000_000,
    invalid_price: bool = False,
    invalid_side: bool = False,
) -> pl.DataFrame:
    """Create a synthetic trades DataFrame for testing."""
    timestamps = [datetime.fromtimestamp((start_us + i * step_us) / 1e6, tz=UTC) for i in range(n)]
    prices = [50000.0 + i * 0.5 for i in range(n)]
    if invalid_price:
        prices[0] = -1.0
    quantities = [0.01 + i * 0.001 for i in range(n)]
    sides = ["buy" if i % 2 == 0 else "sell" for i in range(n)]
    if invalid_side:
        sides[0] = "unknown"
    trade_ids = list(range(1, n + 1))
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "quantity": quantities,
            "side": sides,
            "trade_id": trade_ids,
        }
    )


# ---------------------------------------------------------------------------
# DataValidator tests
# ---------------------------------------------------------------------------

class TestDataValidator:
    """Tests for the DataValidator class."""

    def setup_method(self) -> None:
        self.validator = DataValidator()

    # --- gap_detection ---

    def test_gap_detection_no_gaps(self) -> None:
        """Timestamps with small, uniform gaps pass gap detection."""
        df = _make_book_df(n=50, step_us=500_000)  # 0.5s apart
        result = self.validator.gap_detection(df["timestamp"], max_gap_seconds=1.0)
        assert result.passed is True
        assert "No gaps" in result.message

    def test_gap_detection_with_gaps(self) -> None:
        """A 2-second gap in timestamps is detected."""
        ts = [
            datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 0, 0, 0, 500_000, tzinfo=UTC),
            datetime(2024, 1, 1, 0, 0, 3, tzinfo=UTC),  # 2.5s gap
            datetime(2024, 1, 1, 0, 0, 3, 500_000, tzinfo=UTC),
        ]
        series = pl.Series("timestamp", ts)
        result = self.validator.gap_detection(series, max_gap_seconds=1.0)
        assert result.passed is False
        assert result.details is not None
        assert result.details["gap_count"] == 1

    # --- duplicate_detection ---

    def test_duplicate_detection_no_duplicates(self) -> None:
        """Unique rows pass duplicate detection."""
        df = _make_trades_df(n=20)
        result = self.validator.duplicate_detection(df, ["trade_id"])
        assert result.passed is True

    def test_duplicate_detection_with_duplicates(self) -> None:
        """Duplicate trade IDs are detected."""
        df = _make_trades_df(n=10)
        # Duplicate the first row
        df = pl.concat([df, df.head(1)])
        result = self.validator.duplicate_detection(df, ["trade_id"])
        assert result.passed is False
        assert result.details is not None
        assert result.details["duplicate_count"] >= 2  # both copies are flagged

    # --- validate_book_updates ---

    def test_validate_book_updates_valid(self) -> None:
        """Valid book data passes all checks."""
        df = _make_book_df(n=50)
        results = self.validator.validate_book_updates(df)
        # All checks except statistics should pass
        for r in results:
            assert r.passed is True, f"Check '{r.check}' failed: {r.message}"

    def test_validate_book_updates_crossed_book(self) -> None:
        """Crossed book (bid >= ask) is detected."""
        df = _make_book_df(n=10, crossed=True)
        results = self.validator.validate_book_updates(df)
        crossed = [r for r in results if r.check == "crossed_book"]
        assert len(crossed) == 1
        assert crossed[0].passed is False
        assert crossed[0].details is not None
        assert crossed[0].details["crossed_count"] >= 1

    def test_validate_book_updates_negative_quantity(self) -> None:
        """Negative quantity in book data is detected."""
        df = _make_book_df(n=10, negative_qty=True)
        results = self.validator.validate_book_updates(df)
        qty_check = [r for r in results if r.check == "non_negative_quantities"]
        assert len(qty_check) == 1
        assert qty_check[0].passed is False

    # --- validate_trades ---

    def test_validate_trades_valid(self) -> None:
        """Valid trade data passes all checks."""
        df = _make_trades_df(n=50)
        results = self.validator.validate_trades(df)
        for r in results:
            assert r.passed is True, f"Check '{r.check}' failed: {r.message}"

    def test_validate_trades_invalid_price(self) -> None:
        """Negative price in trade data is detected."""
        df = _make_trades_df(n=10, invalid_price=True)
        results = self.validator.validate_trades(df)
        price_check = [r for r in results if r.check == "positive_prices"]
        assert len(price_check) == 1
        assert price_check[0].passed is False

    def test_validate_trades_invalid_side(self) -> None:
        """Invalid side value is detected."""
        df = _make_trades_df(n=10, invalid_side=True)
        results = self.validator.validate_trades(df)
        side_check = [r for r in results if r.check == "valid_side"]
        assert len(side_check) == 1
        assert side_check[0].passed is False
        assert side_check[0].details is not None
        assert "unknown" in side_check[0].details["invalid_sides"]

    # --- ValidationReport ---

    def test_validation_report_passed(self) -> None:
        """Report with all passing checks reports passed=True."""
        report = ValidationReport(
            file_path="test.parquet",
            symbol="BTCUSDT",
            exchange="binance",
            date_range="2024-01-01 - 2024-01-02",
            total_rows=100,
            results=[
                ValidationResult(check="a", passed=True, message="ok"),
                ValidationResult(check="b", passed=True, message="ok"),
            ],
        )
        assert report.passed is True

    def test_validation_report_failed(self) -> None:
        """Report with any failing check reports passed=False."""
        report = ValidationReport(
            file_path="test.parquet",
            symbol="BTCUSDT",
            exchange="binance",
            date_range="2024-01-01 - 2024-01-02",
            total_rows=100,
            results=[
                ValidationResult(check="a", passed=True, message="ok"),
                ValidationResult(check="b", passed=False, message="bad"),
            ],
        )
        assert report.passed is False

    def test_validation_report_to_json(self) -> None:
        """Report can be serialized to valid JSON."""
        report = ValidationReport(
            file_path="test.parquet",
            symbol="BTCUSDT",
            exchange="binance",
            date_range="2024-01-01",
            total_rows=50,
            results=[
                ValidationResult(check="test", passed=True, message="ok", details={"count": 1}),
            ],
        )
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["symbol"] == "BTCUSDT"
        assert parsed["total_rows"] == 50
        assert len(parsed["results"]) == 1
        assert parsed["results"][0]["passed"] is True

    def test_validation_report_save(self, tmp_path: Path) -> None:
        """Report can be saved to a JSON file."""
        report = ValidationReport(
            file_path="test.parquet",
            symbol="BTCUSDT",
            exchange="binance",
            date_range="2024-01-01",
            total_rows=10,
            results=[
                ValidationResult(check="test", passed=True, message="ok"),
            ],
        )
        out = tmp_path / "report.json"
        report.save(out)
        assert out.exists()
        parsed = json.loads(out.read_text())
        assert parsed["symbol"] == "BTCUSDT"


# ---------------------------------------------------------------------------
# BinanceDownloader tests
# ---------------------------------------------------------------------------

class TestBinanceDownloader:
    """Tests for the BinanceDownloader class."""

    def test_rate_limit_tracking(self) -> None:
        """Rate limit weight is tracked correctly."""
        dl = BinanceDownloader()
        dl._weight_reset_time = time.time() + 60  # ensure within window
        dl._request_weight = 0
        dl._rate_limit(weight=5)
        assert dl._request_weight == 5

    def test_rate_limit_threshold(self) -> None:
        """Rate limiter resets after exceeding threshold."""
        dl = BinanceDownloader()
        dl._weight_reset_time = time.time() + 0.01  # about to reset
        dl._request_weight = 999
        # This pushes over 1000
        time.sleep(0.02)  # let the window expire naturally
        dl._rate_limit(weight=5)
        # After window expiry, weight should be reset then incremented
        assert dl._request_weight == 5

    def test_klines_schema(self) -> None:
        """Downloaded klines DataFrame has the expected schema."""
        # Mock a single response page
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [
                1704067200000, "42000.00", "42100.00", "41900.00", "42050.00",
                "100.5", 1704067259999, "4225000.00", 500,
                "60.3", "2535000.00", "0",
            ]
        ]
        mock_response.raise_for_status = MagicMock()

        dl = BinanceDownloader()
        with patch.object(dl.client, "get", return_value=mock_response):
            df = dl.download_klines("BTCUSDT", "1m", 1704067200000, 1704067260000)

        assert "open_time" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert df.dtypes[df.columns.index("close")] == pl.Float64
        assert len(df) == 1

    def test_resume_point_no_existing_data(self, tmp_path: Path) -> None:
        """Resume returns None when no existing data file exists."""
        dl = BinanceDownloader(output_dir=str(tmp_path))
        assert dl._get_resume_point("BTCUSDT", "klines") is None

    def test_resume_point_with_existing_data(self, tmp_path: Path) -> None:
        """Resume returns the max timestamp from existing data."""
        dl = BinanceDownloader(output_dir=str(tmp_path))
        # Write a small parquet file
        df = pl.DataFrame({"open_time": [100, 200, 300], "close": [1.0, 2.0, 3.0]})
        path = tmp_path / "BTCUSDT_klines.parquet"
        df.write_parquet(path)
        assert dl._get_resume_point("BTCUSDT", "klines") == 300

    def test_download_to_parquet_invalid_type(self, tmp_path: Path) -> None:
        """Invalid data_type raises ValueError."""
        dl = BinanceDownloader(output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="Unknown data_type"):
            dl.download_to_parquet("BTCUSDT", "invalid", 0, 1000)

    def test_agg_trades_schema(self) -> None:
        """Downloaded agg_trades DataFrame has the expected schema."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "a": 123456,
                "p": "42000.00",
                "q": "0.5",
                "f": 100,
                "l": 105,
                "T": 1704067200000,
                "m": True,
            }
        ]
        mock_response.raise_for_status = MagicMock()

        dl = BinanceDownloader()
        with patch.object(dl.client, "get", return_value=mock_response):
            df = dl.download_agg_trades("BTCUSDT", 1704067200000, 1704067260000)

        assert "agg_trade_id" in df.columns
        assert "price" in df.columns
        assert "is_buyer_maker" in df.columns
        assert len(df) == 1
        assert df["price"][0] == 42000.0


# ---------------------------------------------------------------------------
# BybitDownloader tests
# ---------------------------------------------------------------------------

class TestBybitDownloader:
    """Tests for the BybitDownloader class."""

    def test_klines_schema(self) -> None:
        """Downloaded Bybit klines have the expected schema."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    [
                        "1704067200000", "42000.00", "42100.00", "41900.00",
                        "42050.00", "100.5", "4225000.00",
                    ],
                ]
            },
        }
        mock_response.raise_for_status = MagicMock()

        dl = BybitDownloader()
        with patch.object(dl.client, "get", return_value=mock_response):
            df = dl.download_klines("BTCUSDT", "1", 1704067200000, 1704067260000)

        assert "open_time" in df.columns
        assert "close" in df.columns
        assert "turnover" in df.columns
        assert df.dtypes[df.columns.index("open")] == pl.Float64
        assert len(df) == 1

    def test_download_to_parquet_invalid_type(self) -> None:
        """Invalid data_type raises ValueError."""
        dl = BybitDownloader()
        with pytest.raises(ValueError, match="Unknown data_type"):
            dl.download_to_parquet("BTCUSDT", "invalid", 0, 1000)

    def test_trades_schema(self) -> None:
        """Downloaded Bybit trades have the expected schema."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {
                        "execId": "abc-123",
                        "symbol": "BTCUSDT",
                        "side": "Buy",
                        "price": "42000.00",
                        "size": "0.5",
                        "time": "1704067200000",
                    }
                ]
            },
        }
        mock_response.raise_for_status = MagicMock()

        dl = BybitDownloader()
        with patch.object(dl.client, "get", return_value=mock_response):
            df = dl.download_trades("BTCUSDT", 0, 0)

        assert "trade_id" in df.columns
        assert "side" in df.columns
        assert df["side"][0] == "buy"  # lowercased
        assert df["price"][0] == 42000.0


# ---------------------------------------------------------------------------
# QuestDBExporter tests
# ---------------------------------------------------------------------------

class TestQuestDBExporter:
    """Tests for the QuestDBExporter class."""

    def test_export_path_format(self) -> None:
        """Output path follows the expected convention."""
        from datetime import date

        exporter = QuestDBExporter(output_dir="/tmp/test_export")
        path = exporter._output_path("BTCUSDT", "binance", date(2024, 1, 15), "book")
        assert str(path) == "/tmp/test_export/BTCUSDT/binance/2024-01-15.book.parquet"

    def test_create_manifest(self, tmp_path: Path) -> None:
        """Manifest lists Parquet files with correct metadata."""
        # Create directory structure: BTCUSDT/binance/2024-01-01.book.parquet
        data_dir = tmp_path / "BTCUSDT" / "binance"
        data_dir.mkdir(parents=True)
        df = pl.DataFrame({"a": [1, 2, 3]})
        df.write_parquet(data_dir / "2024-01-01.book.parquet")
        df.write_parquet(data_dir / "2024-01-02.trades.parquet")

        exporter = QuestDBExporter(output_dir=str(tmp_path))
        manifest_path = exporter.create_manifest()
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert len(manifest["files"]) == 2
        assert "BTCUSDT" in manifest["symbols"]
        sym = manifest["symbols"]["BTCUSDT"]
        assert "binance" in sym["exchanges"]
        assert "2024-01-01" in sym["dates"]
        assert "2024-01-02" in sym["dates"]

    def test_create_manifest_empty_dir(self, tmp_path: Path) -> None:
        """Manifest for an empty directory produces zero files."""
        exporter = QuestDBExporter(output_dir=str(tmp_path))
        manifest_path = exporter.create_manifest()
        manifest = json.loads(manifest_path.read_text())
        assert len(manifest["files"]) == 0

    def test_query_returns_dataframe(self) -> None:
        """query() parses CSV response into a polars DataFrame."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"col1,col2\n1,hello\n2,world\n"
        mock_response.raise_for_status = MagicMock()

        exporter = QuestDBExporter()
        with patch.object(exporter.client, "get", return_value=mock_response):
            df = exporter.query("SELECT * FROM test")

        assert len(df) == 2
        assert "col1" in df.columns
        assert "col2" in df.columns

    def test_query_empty_response(self) -> None:
        """query() returns empty DataFrame for empty response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b""
        mock_response.raise_for_status = MagicMock()

        exporter = QuestDBExporter()
        with patch.object(exporter.client, "get", return_value=mock_response):
            df = exporter.query("SELECT * FROM empty_table")

        assert len(df) == 0

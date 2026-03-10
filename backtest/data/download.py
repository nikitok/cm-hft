"""Download historical market data from Binance and Bybit exchanges.

Supports downloading kline/OHLCV data and aggregate trade data.
Handles pagination, rate limiting, and resumable downloads.
"""

import logging
import time
from pathlib import Path

import httpx
import polars as pl

logger = logging.getLogger(__name__)


class BinanceDownloader:
    """Download historical data from Binance REST API."""

    BASE_URL = "https://api.binance.com"
    TESTNET_URL = "https://testnet.binance.vision"

    def __init__(self, output_dir: str = "data/historical", testnet: bool = False) -> None:
        """Initialize the Binance downloader.

        Args:
            output_dir: Root directory for downloaded data.
            testnet: If True, use Binance testnet instead of production.
        """
        self.base_url = self.TESTNET_URL if testnet else self.BASE_URL
        self.output_dir = Path(output_dir)
        self.client = httpx.Client(timeout=30.0)
        self._request_weight = 0
        self._weight_reset_time = 0.0

    def _rate_limit(self, weight: int = 1) -> None:
        """Enforce Binance rate limits (1200 weight/min).

        Tracks cumulative request weight and sleeps when approaching the limit.

        Args:
            weight: Weight cost of the upcoming request.
        """
        now = time.time()
        if now > self._weight_reset_time:
            self._request_weight = 0
            self._weight_reset_time = now + 60
        self._request_weight += weight
        if self._request_weight > 1000:  # 80% threshold
            sleep_time = self._weight_reset_time - now
            if sleep_time > 0:
                logger.warning(
                    "Approaching rate limit (%d/1200), sleeping %.1fs",
                    self._request_weight,
                    sleep_time,
                )
                time.sleep(sleep_time)
            self._request_weight = 0
            self._weight_reset_time = time.time() + 60

    def download_klines(
        self, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> pl.DataFrame:
        """Download kline/OHLCV data with automatic pagination.

        GET /api/v3/klines?symbol={}&interval={}&startTime={}&endTime={}&limit=1000

        Args:
            symbol: Trading pair (e.g. BTCUSDT).
            interval: Kline interval (e.g. 1m, 5m, 1h, 1d).
            start_ms: Start time in milliseconds since epoch.
            end_ms: End time in milliseconds since epoch.

        Returns:
            polars DataFrame with columns: open_time, open, high, low, close,
            volume, close_time, quote_volume, trades, taker_buy_volume,
            taker_buy_quote_volume.
        """
        all_rows: list[list] = []
        current_start = start_ms

        while current_start < end_ms:
            self._rate_limit(weight=2)
            response = self.client.get(
                f"{self.base_url}/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "endTime": end_ms,
                    "limit": 1000,
                },
            )
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            all_rows.extend(data)
            # Next page starts after the last kline's close_time
            last_close_time = data[-1][6]
            if last_close_time >= end_ms or len(data) < 1000:
                break
            current_start = last_close_time + 1

        if not all_rows:
            return pl.DataFrame(
                schema={
                    "open_time": pl.Int64,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                    "close_time": pl.Int64,
                    "quote_volume": pl.Float64,
                    "trades": pl.Int64,
                    "taker_buy_volume": pl.Float64,
                    "taker_buy_quote_volume": pl.Float64,
                }
            )

        # Binance returns 12 fields per kline; we use the first 11
        rows = [r[:11] for r in all_rows]
        df = pl.DataFrame(
            rows,
            schema={
                "open_time": pl.Int64,
                "open": pl.Utf8,
                "high": pl.Utf8,
                "low": pl.Utf8,
                "close": pl.Utf8,
                "volume": pl.Utf8,
                "close_time": pl.Int64,
                "quote_volume": pl.Utf8,
                "trades": pl.Int64,
                "taker_buy_volume": pl.Utf8,
                "taker_buy_quote_volume": pl.Utf8,
            },
            orient="row",
        )
        # Cast string numeric fields to Float64
        float_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_volume",
            "taker_buy_quote_volume",
        ]
        df = df.with_columns([pl.col(c).cast(pl.Float64) for c in float_cols])
        return df

    def download_agg_trades(self, symbol: str, start_ms: int, end_ms: int) -> pl.DataFrame:
        """Download aggregate trade data with automatic pagination.

        GET /api/v3/aggTrades?symbol={}&startTime={}&endTime={}&limit=1000
        Paginates by fromId for efficiency.

        Args:
            symbol: Trading pair (e.g. BTCUSDT).
            start_ms: Start time in milliseconds since epoch.
            end_ms: End time in milliseconds since epoch.

        Returns:
            polars DataFrame with columns: agg_trade_id, price, quantity,
            first_trade_id, last_trade_id, timestamp, is_buyer_maker.
        """
        all_rows: list[dict] = []
        from_id: int | None = None

        # First request uses startTime/endTime
        self._rate_limit(weight=2)
        response = self.client.get(
            f"{self.base_url}/api/v3/aggTrades",
            params={
                "symbol": symbol,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        response.raise_for_status()
        data = response.json()

        if not data:
            return pl.DataFrame(
                schema={
                    "agg_trade_id": pl.Int64,
                    "price": pl.Float64,
                    "quantity": pl.Float64,
                    "first_trade_id": pl.Int64,
                    "last_trade_id": pl.Int64,
                    "timestamp": pl.Int64,
                    "is_buyer_maker": pl.Boolean,
                }
            )

        all_rows.extend(data)

        # Subsequent requests paginate by fromId
        while len(data) == 1000:
            from_id = data[-1]["a"] + 1
            self._rate_limit(weight=2)
            response = self.client.get(
                f"{self.base_url}/api/v3/aggTrades",
                params={
                    "symbol": symbol,
                    "fromId": from_id,
                    "limit": 1000,
                },
            )
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            # Filter out trades beyond end_ms
            data = [t for t in data if t["T"] <= end_ms]
            all_rows.extend(data)

            if not data or data[-1]["T"] >= end_ms:
                break

        df = pl.DataFrame(
            {
                "agg_trade_id": [r["a"] for r in all_rows],
                "price": [float(r["p"]) for r in all_rows],
                "quantity": [float(r["q"]) for r in all_rows],
                "first_trade_id": [r["f"] for r in all_rows],
                "last_trade_id": [r["l"] for r in all_rows],
                "timestamp": [r["T"] for r in all_rows],
                "is_buyer_maker": [r["m"] for r in all_rows],
            }
        )
        return df

    def _get_resume_point(self, symbol: str, data_type: str) -> int | None:
        """Check for existing data and return last timestamp for resume.

        Looks for an existing Parquet file and reads the maximum timestamp.

        Args:
            symbol: Trading pair.
            data_type: Data type (klines or agg_trades).

        Returns:
            Last timestamp in ms, or None if no existing data.
        """
        existing = self.output_dir / f"{symbol}_{data_type}.parquet"
        if not existing.exists():
            return None

        try:
            df = pl.read_parquet(existing)
            ts_col = "open_time" if data_type == "klines" else "timestamp"
            if ts_col in df.columns and len(df) > 0:
                return df[ts_col].max()
        except Exception:
            logger.exception("Failed to read existing file for resume: %s", existing)

        return None

    def download_to_parquet(self, symbol: str, data_type: str, start_ms: int, end_ms: int) -> Path:
        """Download data and save to Parquet file. Supports resuming.

        Args:
            symbol: Trading pair.
            data_type: Either 'klines' or 'agg_trades'.
            start_ms: Start time in milliseconds.
            end_ms: End time in milliseconds.

        Returns:
            Path to the written Parquet file.

        Raises:
            ValueError: If data_type is not recognized.
        """
        resume_point = self._get_resume_point(symbol, data_type)
        actual_start = resume_point + 1 if resume_point is not None else start_ms

        if actual_start != start_ms:
            logger.info("Resuming %s/%s from %d", symbol, data_type, actual_start)

        if data_type == "klines":
            df = self.download_klines(symbol, "1m", actual_start, end_ms)
        elif data_type == "agg_trades":
            df = self.download_agg_trades(symbol, actual_start, end_ms)
        else:
            msg = f"Unknown data_type: {data_type}. Use 'klines' or 'agg_trades'."
            raise ValueError(msg)

        out_path = self.output_dir / f"{symbol}_{data_type}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # If resuming, append to existing data
        if resume_point is not None and out_path.exists():
            existing_df = pl.read_parquet(out_path)
            df = pl.concat([existing_df, df])

        df.write_parquet(out_path)
        logger.info("Saved %d rows to %s", len(df), out_path)
        return out_path


class BybitDownloader:
    """Download historical data from Bybit REST API v5."""

    BASE_URL = "https://api.bybit.com"
    TESTNET_URL = "https://api-testnet.bybit.com"

    def __init__(self, output_dir: str = "data/historical", testnet: bool = False) -> None:
        """Initialize the Bybit downloader.

        Args:
            output_dir: Root directory for downloaded data.
            testnet: If True, use Bybit testnet instead of production.
        """
        self.base_url = self.TESTNET_URL if testnet else self.BASE_URL
        self.output_dir = Path(output_dir)
        self.client = httpx.Client(timeout=30.0)

    def download_klines(
        self, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> pl.DataFrame:
        """Download kline data with automatic pagination.

        GET /v5/market/kline?category=linear&symbol={}&interval={}&start={}&end={}&limit=200

        Args:
            symbol: Trading pair (e.g. BTCUSDT).
            interval: Kline interval (e.g. 1, 5, 60, D).
            start_ms: Start time in milliseconds since epoch.
            end_ms: End time in milliseconds since epoch.

        Returns:
            polars DataFrame with columns: open_time, open, high, low, close, volume, turnover.
        """
        all_rows: list[list] = []
        current_end = end_ms

        while current_end > start_ms:
            response = self.client.get(
                f"{self.base_url}/v5/market/kline",
                params={
                    "category": "linear",
                    "symbol": symbol,
                    "interval": interval,
                    "start": start_ms,
                    "end": current_end,
                    "limit": 200,
                },
            )
            response.raise_for_status()
            result = response.json()

            if result.get("retCode") != 0:
                logger.error("Bybit API error: %s", result.get("retMsg"))
                break

            rows = result.get("result", {}).get("list", [])
            if not rows:
                break

            all_rows.extend(rows)
            # Bybit returns newest first; paginate backwards
            oldest_time = int(rows[-1][0])
            if oldest_time <= start_ms or len(rows) < 200:
                break
            current_end = oldest_time - 1

        if not all_rows:
            return pl.DataFrame(
                schema={
                    "open_time": pl.Int64,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                    "turnover": pl.Float64,
                }
            )

        df = pl.DataFrame(
            all_rows,
            schema={
                "open_time": pl.Utf8,
                "open": pl.Utf8,
                "high": pl.Utf8,
                "low": pl.Utf8,
                "close": pl.Utf8,
                "volume": pl.Utf8,
                "turnover": pl.Utf8,
            },
            orient="row",
        )
        df = df.with_columns(
            [
                pl.col("open_time").cast(pl.Int64),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
                pl.col("turnover").cast(pl.Float64),
            ]
        )
        # Sort ascending by time (Bybit returns newest-first)
        df = df.sort("open_time")
        return df

    def download_trades(self, symbol: str, start_ms: int, end_ms: int) -> pl.DataFrame:
        """Download recent trade data.

        GET /v5/market/recent-trade?category=linear&symbol={}&limit=1000

        Note: Bybit's recent-trade endpoint only returns the most recent trades
        (up to 1000). It does NOT support historical trade retrieval by time range.
        The start_ms and end_ms parameters are accepted for API consistency but are
        not used in the request. For historical trade data, use Bybit's data download
        portal or a third-party provider.

        Args:
            symbol: Trading pair (e.g. BTCUSDT).
            start_ms: Not used (Bybit limitation). Kept for API consistency.
            end_ms: Not used (Bybit limitation). Kept for API consistency.

        Returns:
            polars DataFrame with columns: trade_id, symbol, side, price,
            quantity, timestamp.
        """
        response = self.client.get(
            f"{self.base_url}/v5/market/recent-trade",
            params={
                "category": "linear",
                "symbol": symbol,
                "limit": 1000,
            },
        )
        response.raise_for_status()
        result = response.json()

        if result.get("retCode") != 0:
            logger.error("Bybit API error: %s", result.get("retMsg"))
            return pl.DataFrame(
                schema={
                    "trade_id": pl.Utf8,
                    "symbol": pl.Utf8,
                    "side": pl.Utf8,
                    "price": pl.Float64,
                    "quantity": pl.Float64,
                    "timestamp": pl.Int64,
                }
            )

        trades = result.get("result", {}).get("list", [])
        if not trades:
            return pl.DataFrame(
                schema={
                    "trade_id": pl.Utf8,
                    "symbol": pl.Utf8,
                    "side": pl.Utf8,
                    "price": pl.Float64,
                    "quantity": pl.Float64,
                    "timestamp": pl.Int64,
                }
            )

        df = pl.DataFrame(
            {
                "trade_id": [t["execId"] for t in trades],
                "symbol": [t["symbol"] for t in trades],
                "side": [t["side"].lower() for t in trades],
                "price": [float(t["price"]) for t in trades],
                "quantity": [float(t["size"]) for t in trades],
                "timestamp": [int(t["time"]) for t in trades],
            }
        )
        return df.sort("timestamp")

    def download_to_parquet(self, symbol: str, data_type: str, start_ms: int, end_ms: int) -> Path:
        """Download and save to Parquet.

        Args:
            symbol: Trading pair.
            data_type: Either 'klines' or 'trades'.
            start_ms: Start time in milliseconds.
            end_ms: End time in milliseconds.

        Returns:
            Path to the written Parquet file.

        Raises:
            ValueError: If data_type is not recognized.
        """
        if data_type == "klines":
            df = self.download_klines(symbol, "1", start_ms, end_ms)
        elif data_type == "trades":
            df = self.download_trades(symbol, start_ms, end_ms)
        else:
            msg = f"Unknown data_type: {data_type}. Use 'klines' or 'trades'."
            raise ValueError(msg)

        out_path = self.output_dir / f"bybit_{symbol}_{data_type}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        logger.info("Saved %d rows to %s", len(df), out_path)
        return out_path

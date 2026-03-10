"""Export market data from QuestDB to Parquet files.

Queries QuestDB via REST SQL endpoint, converts results to Parquet format
using polars, and partitions by date and exchange.
"""

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import httpx
import polars as pl

logger = logging.getLogger(__name__)


class QuestDBExporter:
    """Exports market data from QuestDB to Parquet files."""

    def __init__(
        self,
        questdb_url: str = "http://localhost:9000",
        output_dir: str = "data",
    ) -> None:
        """Initialize the exporter.

        Args:
            questdb_url: Base URL of the QuestDB REST endpoint.
            output_dir: Root directory for exported Parquet files.
        """
        self.questdb_url = questdb_url.rstrip("/")
        self.output_dir = Path(output_dir)
        self.client = httpx.Client(timeout=60.0)

    def query(self, sql: str) -> pl.DataFrame:
        """Execute SQL query against QuestDB REST endpoint, return as polars DataFrame.

        Args:
            sql: SQL query string.

        Returns:
            polars DataFrame with query results.

        Raises:
            httpx.HTTPStatusError: If QuestDB returns an error status.
            ValueError: If the response cannot be parsed.
        """
        response = self.client.get(
            f"{self.questdb_url}/exec",
            params={"query": sql, "fmt": "csv"},
        )
        response.raise_for_status()

        # QuestDB returns CSV for fmt=csv
        from io import BytesIO

        csv_bytes = response.content
        if not csv_bytes.strip():
            return pl.DataFrame()

        return pl.read_csv(BytesIO(csv_bytes))

    def _output_path(self, symbol: str, exchange: str, day: date, data_type: str) -> Path:
        """Build the output file path.

        Format: {output_dir}/{symbol}/{exchange}/{date}.{data_type}.parquet

        Args:
            symbol: Trading pair symbol (e.g. BTCUSDT).
            exchange: Exchange name (e.g. binance).
            day: Date for the file.
            data_type: Either 'book' or 'trades'.

        Returns:
            Path to the output Parquet file.
        """
        return self.output_dir / symbol / exchange / f"{day.isoformat()}.{data_type}.parquet"

    def export_book_updates(self, symbol: str, exchange: str, start: date, end: date) -> list[Path]:
        """Export book updates for a date range to Parquet, one file per day.

        Output: {output_dir}/{symbol}/{exchange}/{date}.book.parquet

        Args:
            symbol: Trading pair symbol.
            exchange: Exchange name.
            start: Start date (inclusive).
            end: End date (exclusive).

        Returns:
            List of written file paths.
        """
        paths: list[Path] = []
        current = start
        while current < end:
            next_day = current + timedelta(days=1)
            sql = (
                f"SELECT * FROM book_updates "
                f"WHERE symbol = '{symbol}' AND exchange = '{exchange}' "
                f"AND timestamp >= '{current.isoformat()}' "
                f"AND timestamp < '{next_day.isoformat()}' "
                f"ORDER BY timestamp"
            )
            try:
                df = self.query(sql)
                if len(df) > 0:
                    out_path = self._output_path(symbol, exchange, current, "book")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    df.write_parquet(out_path)
                    logger.info("Exported %d book rows to %s", len(df), out_path)
                    paths.append(out_path)
                else:
                    logger.debug("No book data for %s/%s on %s", symbol, exchange, current)
            except Exception:
                logger.exception("Failed to export book data for %s", current)
            current = next_day
        return paths

    def export_trades(self, symbol: str, exchange: str, start: date, end: date) -> list[Path]:
        """Export trades for a date range to Parquet, one file per day.

        Output: {output_dir}/{symbol}/{exchange}/{date}.trades.parquet

        Args:
            symbol: Trading pair symbol.
            exchange: Exchange name.
            start: Start date (inclusive).
            end: End date (exclusive).

        Returns:
            List of written file paths.
        """
        paths: list[Path] = []
        current = start
        while current < end:
            next_day = current + timedelta(days=1)
            sql = (
                f"SELECT * FROM trades "
                f"WHERE symbol = '{symbol}' AND exchange = '{exchange}' "
                f"AND timestamp >= '{current.isoformat()}' "
                f"AND timestamp < '{next_day.isoformat()}' "
                f"ORDER BY timestamp"
            )
            try:
                df = self.query(sql)
                if len(df) > 0:
                    out_path = self._output_path(symbol, exchange, current, "trades")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    df.write_parquet(out_path)
                    logger.info("Exported %d trade rows to %s", len(df), out_path)
                    paths.append(out_path)
                else:
                    logger.debug("No trade data for %s/%s on %s", symbol, exchange, current)
            except Exception:
                logger.exception("Failed to export trade data for %s", current)
            current = next_day
        return paths

    def export_range(self, symbol: str, exchange: str, start: date, end: date) -> list[Path]:
        """Export all data types for a date range, one file per day.

        Args:
            symbol: Trading pair symbol.
            exchange: Exchange name.
            start: Start date (inclusive).
            end: End date (exclusive).

        Returns:
            Combined list of all written file paths.
        """
        paths: list[Path] = []
        paths.extend(self.export_book_updates(symbol, exchange, start, end))
        paths.extend(self.export_trades(symbol, exchange, start, end))
        return paths

    def create_manifest(self, output_dir: Path | None = None) -> Path:
        """Create manifest.json listing all available data files with date ranges.

        Scans the output directory for Parquet files and records their metadata.

        Args:
            output_dir: Directory to scan. Defaults to self.output_dir.

        Returns:
            Path to the written manifest.json.
        """
        scan_dir = Path(output_dir) if output_dir else self.output_dir
        manifest: dict = {"files": [], "symbols": {}}

        for parquet_file in sorted(scan_dir.rglob("*.parquet")):
            rel = parquet_file.relative_to(scan_dir)
            parts = rel.parts  # symbol/exchange/date.type.parquet

            entry: dict = {"path": str(rel)}
            if len(parts) >= 3:
                entry["symbol"] = parts[0]
                entry["exchange"] = parts[1]
                # Extract date from filename like 2024-01-15.book.parquet
                filename = parts[2]
                date_str = filename.split(".")[0]
                entry["date"] = date_str
                entry["type"] = filename.split(".")[1] if "." in filename else "unknown"

                # Aggregate by symbol
                sym = entry["symbol"]
                if sym not in manifest["symbols"]:
                    manifest["symbols"][sym] = {"exchanges": set(), "dates": set()}
                manifest["symbols"][sym]["exchanges"].add(entry.get("exchange", ""))
                manifest["symbols"][sym]["dates"].add(date_str)

            manifest["files"].append(entry)

        # Convert sets to sorted lists for JSON serialization
        for sym_info in manifest["symbols"].values():
            sym_info["exchanges"] = sorted(sym_info["exchanges"])
            sym_info["dates"] = sorted(sym_info["dates"])
            if sym_info["dates"]:
                sym_info["date_range"] = {
                    "start": sym_info["dates"][0],
                    "end": sym_info["dates"][-1],
                }

        manifest_path = scan_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        logger.info("Created manifest with %d files at %s", len(manifest["files"]), manifest_path)
        return manifest_path

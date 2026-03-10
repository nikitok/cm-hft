"""Validate market data quality in Parquet files.

Checks for gaps, duplicates, consistency issues, and generates validation reports.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    check: str
    passed: bool
    message: str
    details: dict | None = None


@dataclass
class ValidationReport:
    """Complete validation report for a data file."""

    file_path: str
    symbol: str
    exchange: str
    date_range: str
    total_rows: int
    results: list[ValidationResult]

    @property
    def passed(self) -> bool:
        """Return True if all validation checks passed."""
        return all(r.passed for r in self.results)

    def to_json(self) -> str:
        """Serialize report to JSON string."""
        return json.dumps(asdict(self), indent=2)

    def save(self, output_path: Path) -> None:
        """Write report to a JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json())


class DataValidator:
    """Validates market data quality."""

    def gap_detection(
        self, timestamps: pl.Series, max_gap_seconds: float = 1.0
    ) -> ValidationResult:
        """Detect time gaps larger than threshold.

        Args:
            timestamps: Series of datetime values (must be sorted).
            max_gap_seconds: Maximum allowed gap between consecutive timestamps.

        Returns:
            ValidationResult with gap details.
        """
        if len(timestamps) < 2:
            return ValidationResult(
                check="gap_detection",
                passed=True,
                message="Not enough rows to check for gaps",
            )

        diffs = timestamps.diff().drop_nulls()

        # Convert to microseconds for comparison (polars durations are in microseconds)
        threshold_us = int(max_gap_seconds * 1_000_000)
        gaps = diffs.filter(diffs.cast(pl.Int64) > threshold_us)
        gap_count = len(gaps)

        if gap_count == 0:
            return ValidationResult(
                check="gap_detection",
                passed=True,
                message=f"No gaps exceeding {max_gap_seconds}s found",
            )

        max_gap_us = gaps.cast(pl.Int64).max()
        return ValidationResult(
            check="gap_detection",
            passed=False,
            message=f"Found {gap_count} gap(s) exceeding {max_gap_seconds}s",
            details={
                "gap_count": gap_count,
                "max_gap_us": max_gap_us,
                "threshold_us": threshold_us,
            },
        )

    def duplicate_detection(self, df: pl.DataFrame, key_columns: list[str]) -> ValidationResult:
        """Detect duplicate rows based on key columns.

        Args:
            df: DataFrame to check.
            key_columns: Columns that should be unique together.

        Returns:
            ValidationResult with duplicate details.
        """
        available_cols = [c for c in key_columns if c in df.columns]
        if not available_cols:
            return ValidationResult(
                check="duplicate_detection",
                passed=True,
                message="No key columns found in DataFrame",
            )

        dup_count = df.select(available_cols).is_duplicated().sum()

        if dup_count == 0:
            return ValidationResult(
                check="duplicate_detection",
                passed=True,
                message=f"No duplicates found on columns {available_cols}",
            )

        return ValidationResult(
            check="duplicate_detection",
            passed=False,
            message=f"Found {dup_count} duplicate row(s) on columns {available_cols}",
            details={"duplicate_count": dup_count, "key_columns": available_cols},
        )

    def validate_book_updates(self, df: pl.DataFrame) -> list[ValidationResult]:
        """Validate L2 book update data.

        Expected columns: timestamp, bid_price, bid_qty, ask_price, ask_qty.

        Checks:
            1. Gap detection: time gaps > 1 second
            2. Duplicate timestamps
            3. Crossed book: bid >= ask
            4. Non-negative quantities
            5. Positive prices
            6. Statistics: message count, avg update rate
        """
        results: list[ValidationResult] = []

        # 1. Gap detection
        if "timestamp" in df.columns:
            results.append(self.gap_detection(df["timestamp"], max_gap_seconds=1.0))
        else:
            results.append(
                ValidationResult(
                    check="gap_detection",
                    passed=False,
                    message="Missing 'timestamp' column",
                )
            )

        # 2. Duplicate timestamps
        results.append(self.duplicate_detection(df, ["timestamp"]))

        # 3. Crossed book: bid should be < ask
        if "bid_price" in df.columns and "ask_price" in df.columns:
            crossed = df.filter(pl.col("bid_price") >= pl.col("ask_price"))
            crossed_count = len(crossed)
            results.append(
                ValidationResult(
                    check="crossed_book",
                    passed=crossed_count == 0,
                    message=(
                        "No crossed books found"
                        if crossed_count == 0
                        else f"Found {crossed_count} crossed book(s) (bid >= ask)"
                    ),
                    details={"crossed_count": crossed_count} if crossed_count > 0 else None,
                )
            )
        else:
            results.append(
                ValidationResult(
                    check="crossed_book",
                    passed=False,
                    message="Missing bid_price or ask_price column",
                )
            )

        # 4. Non-negative quantities
        qty_cols = [c for c in ["bid_qty", "ask_qty"] if c in df.columns]
        if qty_cols:
            neg_count = 0
            for col in qty_cols:
                neg_count += df.filter(pl.col(col) < 0).height
            results.append(
                ValidationResult(
                    check="non_negative_quantities",
                    passed=neg_count == 0,
                    message=(
                        "All quantities are non-negative"
                        if neg_count == 0
                        else f"Found {neg_count} negative quantity value(s)"
                    ),
                    details={"negative_count": neg_count} if neg_count > 0 else None,
                )
            )
        else:
            results.append(
                ValidationResult(
                    check="non_negative_quantities",
                    passed=False,
                    message="Missing quantity columns (bid_qty, ask_qty)",
                )
            )

        # 5. Positive prices
        price_cols = [c for c in ["bid_price", "ask_price"] if c in df.columns]
        if price_cols:
            non_pos_count = 0
            for col in price_cols:
                non_pos_count += df.filter(pl.col(col) <= 0).height
            results.append(
                ValidationResult(
                    check="positive_prices",
                    passed=non_pos_count == 0,
                    message=(
                        "All prices are positive"
                        if non_pos_count == 0
                        else f"Found {non_pos_count} non-positive price(s)"
                    ),
                    details={"non_positive_count": non_pos_count} if non_pos_count > 0 else None,
                )
            )
        else:
            results.append(
                ValidationResult(
                    check="positive_prices",
                    passed=False,
                    message="Missing price columns (bid_price, ask_price)",
                )
            )

        # 6. Statistics
        stats: dict = {"total_rows": len(df)}
        if "timestamp" in df.columns and len(df) >= 2:
            duration = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
            if duration > 0:
                stats["avg_updates_per_second"] = round(len(df) / duration, 2)
            stats["duration_seconds"] = round(duration, 2)
        results.append(
            ValidationResult(
                check="statistics",
                passed=True,
                message=f"Book update statistics: {len(df)} rows",
                details=stats,
            )
        )

        return results

    def validate_trades(self, df: pl.DataFrame) -> list[ValidationResult]:
        """Validate trade data.

        Expected columns: timestamp, price, quantity, side, (optionally trade_id).

        Checks:
            1. Gap detection: time gaps > 5 seconds
            2. Duplicate trade IDs
            3. Price > 0, quantity > 0
            4. Side is valid (buy/sell)
            5. Monotonic timestamps
        """
        results: list[ValidationResult] = []

        # 1. Gap detection
        if "timestamp" in df.columns:
            results.append(self.gap_detection(df["timestamp"], max_gap_seconds=5.0))
        else:
            results.append(
                ValidationResult(
                    check="gap_detection",
                    passed=False,
                    message="Missing 'timestamp' column",
                )
            )

        # 2. Duplicate trade IDs
        if "trade_id" in df.columns:
            results.append(self.duplicate_detection(df, ["trade_id"]))
        else:
            # Fall back to timestamp + price + quantity
            results.append(self.duplicate_detection(df, ["timestamp", "price", "quantity"]))

        # 3. Price > 0
        if "price" in df.columns:
            non_pos = df.filter(pl.col("price") <= 0).height
            results.append(
                ValidationResult(
                    check="positive_prices",
                    passed=non_pos == 0,
                    message=(
                        "All prices are positive"
                        if non_pos == 0
                        else f"Found {non_pos} non-positive price(s)"
                    ),
                    details={"non_positive_count": non_pos} if non_pos > 0 else None,
                )
            )
        else:
            results.append(
                ValidationResult(
                    check="positive_prices",
                    passed=False,
                    message="Missing 'price' column",
                )
            )

        # 3b. Quantity > 0
        if "quantity" in df.columns:
            non_pos = df.filter(pl.col("quantity") <= 0).height
            results.append(
                ValidationResult(
                    check="positive_quantities",
                    passed=non_pos == 0,
                    message=(
                        "All quantities are positive"
                        if non_pos == 0
                        else f"Found {non_pos} non-positive quantity/ies"
                    ),
                    details={"non_positive_count": non_pos} if non_pos > 0 else None,
                )
            )
        else:
            results.append(
                ValidationResult(
                    check="positive_quantities",
                    passed=False,
                    message="Missing 'quantity' column",
                )
            )

        # 4. Valid side
        if "side" in df.columns:
            valid_sides = {"buy", "sell"}
            unique_sides = set(df["side"].unique().to_list())
            invalid_sides = unique_sides - valid_sides
            results.append(
                ValidationResult(
                    check="valid_side",
                    passed=len(invalid_sides) == 0,
                    message=(
                        "All sides are valid (buy/sell)"
                        if len(invalid_sides) == 0
                        else f"Found invalid side values: {invalid_sides}"
                    ),
                    details={"invalid_sides": list(invalid_sides)} if invalid_sides else None,
                )
            )
        else:
            results.append(
                ValidationResult(
                    check="valid_side",
                    passed=False,
                    message="Missing 'side' column",
                )
            )

        # 5. Monotonic timestamps
        if "timestamp" in df.columns and len(df) >= 2:
            diffs = df["timestamp"].diff().drop_nulls()
            non_mono = diffs.filter(diffs.cast(pl.Int64) < 0)
            non_mono_count = len(non_mono)
            results.append(
                ValidationResult(
                    check="monotonic_timestamps",
                    passed=non_mono_count == 0,
                    message=(
                        "Timestamps are monotonically non-decreasing"
                        if non_mono_count == 0
                        else f"Found {non_mono_count} timestamp reversal(s)"
                    ),
                    details={"reversal_count": non_mono_count} if non_mono_count > 0 else None,
                )
            )
        elif "timestamp" in df.columns:
            results.append(
                ValidationResult(
                    check="monotonic_timestamps",
                    passed=True,
                    message="Not enough rows to check monotonicity",
                )
            )

        return results

    def validate_file(self, file_path: Path) -> ValidationReport:
        """Run all validations on a Parquet file.

        Infers data type from filename convention:
            - *.book.parquet -> book updates
            - *.trades.parquet -> trades

        Args:
            file_path: Path to the Parquet file.

        Returns:
            ValidationReport with all check results.
        """
        file_path = Path(file_path)
        df = pl.read_parquet(file_path)

        # Infer metadata from path: {output_dir}/{symbol}/{exchange}/{date}.{type}.parquet
        parts = file_path.parts
        symbol = parts[-3] if len(parts) >= 3 else "unknown"
        exchange = parts[-2] if len(parts) >= 2 else "unknown"

        if "timestamp" in df.columns and len(df) > 0:
            ts_min = str(df["timestamp"].min())
            ts_max = str(df["timestamp"].max())
            date_range = f"{ts_min} - {ts_max}"
        else:
            date_range = "unknown"

        name = file_path.name
        if ".book." in name:
            results = self.validate_book_updates(df)
        elif ".trades." in name:
            results = self.validate_trades(df)
        else:
            # Default: run basic checks
            results = self.validate_trades(df)

        return ValidationReport(
            file_path=str(file_path),
            symbol=symbol,
            exchange=exchange,
            date_range=date_range,
            total_rows=len(df),
            results=results,
        )

    def validate_directory(self, dir_path: Path) -> list[ValidationReport]:
        """Validate all Parquet files in a directory recursively.

        Args:
            dir_path: Root directory to scan for .parquet files.

        Returns:
            List of ValidationReports, one per file.
        """
        dir_path = Path(dir_path)
        reports: list[ValidationReport] = []

        for parquet_file in sorted(dir_path.rglob("*.parquet")):
            logger.info("Validating %s", parquet_file)
            try:
                report = self.validate_file(parquet_file)
                reports.append(report)
                status = "PASS" if report.passed else "FAIL"
                logger.info("  %s — %d rows, %s", status, report.total_rows, parquet_file.name)
            except Exception:
                logger.exception("  ERROR validating %s", parquet_file)

        return reports

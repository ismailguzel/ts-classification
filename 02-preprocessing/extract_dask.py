"""
Extract TSFresh features for hierarchical classification using Dask.

Run from the `02-preprocessing` directory.

Usage:
    python extract_dask.py --input ../data/raw/unified-90k --output ../data/features

    cd 02-preprocessing && python extract_dask.py --input ../data/raw/unified-20k --output ../data/features/test-dask --max-files 20 to smoke-test on a small subset.
Tune --n-workers, --threads-per-worker, or point --scheduler-address at your existing cluster once the basics look good.


This script reads the time series parquet files with dask, builds the feature
extraction graph via tsfresh, and materialises the result with `compute()`.

Outputs under the specified `--output` directory:
    - features.parquet
    - labels.parquet
    - feature_names.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import dask.dataframe as dd

warnings.filterwarnings("ignore")

try:
    from dask.distributed import Client  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Client = None  # type: ignore

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import (
    ComprehensiveFCParameters,
    EfficientFCParameters,
    MinimalFCParameters,
)


REQUIRED_COLUMNS = {
    "series_id",
    "time",
    "data",
    "is_stationary",
    "primary_category",
    "sub_category",
}


@dataclass
class DaskConfig:
    """Configuration for the optional local Dask client."""

    scheduler_address: Optional[str]
    n_workers: Optional[int]
    threads_per_worker: Optional[int]
    memory_limit: Optional[str]
    start_client: bool


class DaskTSFreshFeatureExtractor:
    """Extract TSFresh features using Dask-backed execution."""

    def __init__(self, feature_set: str = "efficient") -> None:
        self.feature_set = feature_set
        self.extraction_settings = self._select_feature_set(feature_set)
        self.client: Optional[Client] = None  # type: ignore[assignment]

    @staticmethod
    def _select_feature_set(feature_set: str):
        """Return TSFresh extraction parameters for the requested feature set."""
        feature_set = feature_set.lower()
        if feature_set == "minimal":
            return MinimalFCParameters()
        if feature_set == "comprehensive":
            return ComprehensiveFCParameters()
        if feature_set == "efficient":
            return EfficientFCParameters()
        raise ValueError(f"Unknown feature set: {feature_set}")

    def start_client(self, config: DaskConfig) -> None:
        """Create a Dask client based on the provided configuration."""
        if not config.start_client:
            return

        if Client is None:  # pragma: no cover - handled at runtime
            raise ImportError(
                "dask.distributed is not available. Install it or use --no-client."
            )

        if config.scheduler_address:
            self.client = Client(config.scheduler_address)  # type: ignore[arg-type]
            return

        client_kwargs = {}
        if config.n_workers is not None:
            client_kwargs["n_workers"] = config.n_workers
        if config.threads_per_worker is not None:
            client_kwargs["threads_per_worker"] = config.threads_per_worker
        if config.memory_limit is not None:
            client_kwargs["memory_limit"] = config.memory_limit

        # If no kwargs provided, rely on Dask defaults.
        self.client = Client(**client_kwargs)  # type: ignore[arg-type]

    def shutdown_client(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    @staticmethod
    def _collect_parquet_files(input_path: Path, max_files: Optional[int]) -> List[Path]:
        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")

        parquet_files = sorted(input_path.rglob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in: {input_path}")

        if max_files is not None:
            parquet_files = parquet_files[:max_files]
        return parquet_files

    def _read_dask_dataframe(
        self,
        parquet_files: Iterable[Path],
        columns: Optional[List[str]] = None,
        gather_statistics: bool = False,
    ) -> dd.DataFrame:
        string_paths = [str(p) for p in parquet_files]
        ddf = dd.read_parquet(
            string_paths,
            engine="pyarrow",
            gather_statistics=gather_statistics,
            columns=columns,
        )
        missing = REQUIRED_COLUMNS.difference({str(col) for col in ddf.columns})
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        return ddf

    @staticmethod
    def _clean_dataframe(ddf: dd.DataFrame) -> dd.DataFrame:
        """Basic sanity cleaning to avoid NaNs/Infs for TSFresh."""
        ddf = ddf.replace([np.inf, -np.inf], np.nan)
        ddf = ddf.dropna(subset=["data"])
        return ddf

    @staticmethod
    def _prepare_timeseries(ddf: dd.DataFrame) -> dd.DataFrame:
        """Select only the columns TSFresh needs for extraction."""
        return ddf[["series_id", "time", "data"]]

    @staticmethod
    def _prepare_labels(ddf: dd.DataFrame) -> pd.DataFrame:
        labels_ddf = ddf.groupby("series_id").agg(
            {
                "is_stationary": "first",
                "primary_category": "first",
                "sub_category": "first",
            }
        )
        labels_df = labels_ddf.compute()
        labels_df = labels_df.reset_index().sort_values("series_id").reset_index(drop=True)
        return labels_df

    def extract(self, ddf: dd.DataFrame, disable_progressbar: bool) -> pd.DataFrame:
        timeseries_ddf = self._prepare_timeseries(ddf)
        features_ddf = extract_features(
            timeseries_ddf,
            column_id="series_id",
            column_sort="time",
            column_value="data",
            default_fc_parameters=self.extraction_settings,
            disable_progressbar=disable_progressbar,
        )
        features_df = features_ddf.compute()
        features_df = impute(features_df)
        features_df.index.name = "series_id"
        # Ensure deterministic ordering by index
        features_df = features_df.sort_index()
        return features_df

    @staticmethod
    def save_results(features: pd.DataFrame, labels: pd.DataFrame, output_path: Path) -> None:
        output_path.mkdir(parents=True, exist_ok=True)

        features_file = output_path / "features.parquet"
        labels_file = output_path / "labels.parquet"
        feature_names_file = output_path / "feature_names.txt"

        features.reset_index().to_parquet(features_file, index=False)
        labels.to_parquet(labels_file, index=False)

        with feature_names_file.open("w", encoding="utf-8") as fh:
            for column in features.columns:
                fh.write(f"{column}\n")

    def run(
        self,
        input_path: Path,
        output_path: Path,
        max_files: Optional[int],
        disable_progressbar: bool,
        gather_statistics: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        parquet_files = self._collect_parquet_files(input_path, max_files)
        print(f"Found {len(parquet_files)} parquet files under {input_path}")

        ddf = self._read_dask_dataframe(
            parquet_files,
            columns=sorted(REQUIRED_COLUMNS),
            gather_statistics=gather_statistics,
        )
        print(f"Loaded Dask DataFrame with {ddf.npartitions} partitions")

        ddf = self._clean_dataframe(ddf)
        print("Applied basic NaN/Inf cleaning (lazy)")

        labels_df = self._prepare_labels(ddf)
        print(f"Computed labels dataframe with shape {labels_df.shape}")

        features_df = self.extract(ddf, disable_progressbar=disable_progressbar)
        print(f"Computed feature dataframe with shape {features_df.shape}")

        self.save_results(features_df, labels_df, output_path)
        print(f"Saved outputs to {output_path}")

        return features_df, labels_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract TSFresh features on parquet files using Dask"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../data/raw/unified-90k",
        help="Input directory containing parquet files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../data/features",
        help="Directory to store extracted features",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="efficient",
        choices=["minimal", "efficient", "comprehensive"],
        help="TSFresh feature set to use",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit the number of parquet files to process (debug/testing)",
    )
    parser.add_argument(
        "--no-client",
        action="store_true",
        help="Do not create a local Dask client (use the default scheduler)",
    )
    parser.add_argument(
        "--scheduler-address",
        type=str,
        default=None,
        help="Connect to an existing Dask scheduler at the given address",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of local Dask workers when creating a client",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=None,
        help="Threads per local Dask worker",
    )
    parser.add_argument(
        "--memory-limit",
        type=str,
        default=None,
        help="Per-worker memory limit (e.g. '16GB') when creating a client",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable TSFresh progress bars",
    )
    parser.add_argument(
        "--gather-statistics",
        action="store_true",
        help="Ask dask.read_parquet to gather metadata statistics (may slow startup)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("TSFresh Feature Extraction with Dask")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Feature set: {args.feature_set}")
    if args.max_files is not None:
        print(f"Max files: {args.max_files}")
    print(f"Using Dask client: {not args.no_client}")
    if args.scheduler_address:
        print(f"Scheduler address: {args.scheduler_address}")
    if args.n_workers is not None:
        print(f"n_workers: {args.n_workers}")
    if args.threads_per_worker is not None:
        print(f"threads_per_worker: {args.threads_per_worker}")
    if args.memory_limit is not None:
        print(f"memory_limit: {args.memory_limit}")
    print("=" * 80)

    extractor = DaskTSFreshFeatureExtractor(feature_set=args.feature_set)
    dask_config = DaskConfig(
        scheduler_address=args.scheduler_address,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        memory_limit=args.memory_limit,
        start_client=not args.no_client,
    )

    try:
        extractor.start_client(dask_config)
        features_df, labels_df = extractor.run(
            input_path=Path(args.input),
            output_path=Path(args.output),
            max_files=args.max_files,
            disable_progressbar=args.no_progress,
            gather_statistics=args.gather_statistics,
        )
    finally:
        extractor.shutdown_client()

    print("\n🎉 Feature extraction completed successfully!")
    print(f"   Final features: {features_df.shape}")
    print(f"   Final labels: {labels_df.shape}")


if __name__ == "__main__":
    main()

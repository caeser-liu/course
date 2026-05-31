"""Inspect the WM-811K LSWMD.pkl file.

This script does not modify data. It prints the dataframe schema, label
distribution, split labels, wafer-map shapes, and several sample summaries.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


class CompatUnpickler(pickle.Unpickler):
    """Load old pandas pickle files that reference pandas.indexes.* modules."""

    def find_class(self, module, name):
        if module.startswith("pandas.indexes"):
            module = module.replace("pandas.indexes", "pandas.core.indexes", 1)
        return super().find_class(module, name)


def extract_label(value):
    if isinstance(value, np.ndarray) and value.size > 0:
        return value[0][0]
    return "unknown"


def load_pickle(path: Path):
    for encoding in ("latin1", "bytes"):
        try:
            with path.open("rb") as file:
                data = CompatUnpickler(file, encoding=encoding).load()
            print(f"Loaded {path} with encoding={encoding!r}")
            return data
        except Exception as exc:
            print(f"encoding={encoding!r} failed: {exc}")
    raise RuntimeError(f"Could not load pickle file: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect WM-811K LSWMD.pkl")
    parser.add_argument(
        "--pkl-path",
        default=str(Path(__file__).resolve().parent / "archive" / "LSWMD.pkl"),
        help="Path to LSWMD.pkl",
    )
    parser.add_argument(
        "--shape-sample-size",
        type=int,
        default=1000,
        help="Number of rows sampled for waferMap shape statistics",
    )
    args = parser.parse_args()

    pkl_path = Path(args.pkl_path)
    if not pkl_path.exists():
        print(f"Missing file: {pkl_path}")
        print("Download it first with scripts/download_dataset.ps1 or scripts/download_dataset.sh")
        return 1

    print(f"File size: {os.path.getsize(pkl_path) / (1024 ** 3):.2f} GB")
    data = load_pickle(pkl_path)
    print(f"Data type: {type(data)}")

    if not isinstance(data, pd.DataFrame):
        print("The pickle did not contain a pandas DataFrame.")
        return 1

    print(f"DataFrame shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Dtypes:\n{data.dtypes}")
    print(f"Memory: {data.memory_usage(deep=True).sum() / (1024 ** 3):.2f} GB")
    print(f"Head:\n{data.head()}")

    print("\n" + "=" * 60)
    print("failureType distribution")
    print("=" * 60)
    failure_series = data["failureType"].apply(extract_label)
    print(failure_series.value_counts())

    if "trianTestLabel" in data.columns:
        print("\n" + "=" * 60)
        print("trianTestLabel distribution")
        print("=" * 60)
        split_series = data["trianTestLabel"].apply(extract_label)
        print(split_series.value_counts())

    print("\n" + "=" * 60)
    print("waferMap sample analysis")
    print("=" * 60)
    for i in range(min(10, len(data))):
        wafer_map = data["waferMap"].iloc[i]
        print(
            f"sample {i}: shape={wafer_map.shape}, "
            f"nonzero={np.count_nonzero(wafer_map)}, "
            f"max={wafer_map.max()}, unique={np.unique(wafer_map)}"
        )

    print("\n" + "=" * 60)
    print("labeled data distribution")
    print("=" * 60)
    labeled = data[failure_series != "unknown"]
    print(f"labeled samples: {len(labeled)}")
    print(labeled["failureType"].apply(extract_label).value_counts())

    print("\n" + "=" * 60)
    print(f"waferMap shape distribution, sample={args.shape_sample_size}")
    print("=" * 60)
    sample_data = data.sample(min(args.shape_sample_size, len(data)), random_state=42)
    shapes = sample_data["waferMap"].apply(lambda x: x.shape)
    print(shapes.value_counts())

    return 0


if __name__ == "__main__":
    sys.exit(main())

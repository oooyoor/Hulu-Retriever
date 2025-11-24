#!/usr/bin/env python3
"""
Aggregate query difficulty buckets and early/base/extreme method deltas.

Definitions:
    - extreme: SearchResults (earliest exit / theoretical best).
    - base:    StableResults (stable/default execution).
    - early:   EarlyResults  (proposed early-stop method).

The script:
    1) Uses extreme (SearchResults) iter_count values to build per-dataset
       quantile bins (difficulty levels).
    2) Assigns each query-id (array index) to one level.
    3) For each level aggregates iter_count statistics for the three methods.
    4) Computes improvement ratios:
           (base - early) / base
           (early - extreme) / early
    5) Dumps a JSON summary that also includes the raw query-id lists per level.

Example:
    python compare_query_difficulty.py \
        --results-root /home/zqf/Hulu-Retriever/Results \
        --output /home/zqf/Hulu-Retriever/Results/difficulty_summary.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class MethodPaths:
    """Holds per-method Offset_results roots."""

    base: Path      # StableResults
    early: Path     # EarlyResults
    extreme: Path   # SearchResults


@dataclass(frozen=True)
class IterData:
    """Container for per-method metric arrays plus extreme iter_counts for binning."""

    query_ids: List[int]
    base: List[float]
    early: List[float]
    extreme: List[float]
    extreme_iters: List[float]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def parse_config_dirname(name: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Parse 'threads_ef_iodepth' -> (threads, ef, iodepth)."""
    parts = name.split("_")
    if len(parts) < 3:
        return None, None, None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None, None, None


def load_metric_values(path: Path, key: str) -> Optional[List[float]]:
    """Return list of metric values (e.g., iter_count or dist_count) from JSON."""
    with open(path, "r") as f:
        data = json.load(f)

    values: List[float] = []
    for entry in data.get("entries", []):
        if key not in entry:
            return None
        values.append(float(entry[key]))
    return values


def locate_iter_file(config_dir: Path, candidate_names: Sequence[str]) -> Optional[Path]:
    """Traverse {queries}/{repeat}/file and return the first existing file."""
    if not config_dir.exists():
        return None
    for query_dir in sorted(p for p in config_dir.iterdir() if p.is_dir()):
        for repeat_dir in sorted(p for p in query_dir.iterdir() if p.is_dir()):
            for fname in candidate_names:
                candidate = repeat_dir / fname
                if candidate.exists():
                    return candidate
    return None


def find_config_with_matching_ef(dataset_dir: Path, target_ef: Optional[int]) -> Optional[Path]:
    """Pick the config directory whose EF matches target_ef."""
    if target_ef is None or not dataset_dir.exists():
        return None
    for cfg_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        _, ef, _ = parse_config_dirname(cfg_dir.name)
        if ef == target_ef:
            return cfg_dir
    return None


def make_quantile_bins(values: Sequence[float], num_bins: int) -> List[Tuple[int, int]]:
    """Return integer quantile bins."""
    if not values:
        return [(0, 0)] * num_bins
    arr = np.array(values, dtype=float)
    qs = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(arr, qs)
    edges = np.round(edges).astype(int)
    bins: List[Tuple[int, int]] = []
    for idx in range(num_bins):
        lo = int(edges[idx])
        hi = int(edges[idx + 1])
        if hi < lo:
            hi = lo
        bins.append((lo, hi))
    return bins


def assign_levels(values: Sequence[float], bins: Sequence[Tuple[int, int]]) -> List[int]:
    """Assign each iter_count to a level index (len == len(values))."""
    levels: List[int] = []
    for val in values:
        assigned = False
        for idx, (lo, hi) in enumerate(bins):
            if lo <= val <= hi:
                levels.append(idx)
                assigned = True
                break
        if not assigned:
            levels.append(len(bins) - 1)
    return levels


def compute_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Aggregated statistics for a bucket."""
    if not values:
        return {
            "avg_iter": None,
            "median_iter": None,
            "min_iter": None,
            "max_iter": None,
        }
    return {
        "avg_iter": float(sum(values) / len(values)),
        "median_iter": float(statistics.median(values)),
        "min_iter": float(min(values)),
        "max_iter": float(max(values)),
    }


def build_bucket_stats(ids: List[int], data: IterData) -> Dict[str, object]:
    """Aggregate per-difficulty statistics."""
    lvl_stats: Dict[str, Dict[str, Optional[float]]] = {}
    method_samples = {
        "base": [data.base[i] for i in ids],
        "early": [data.early[i] for i in ids],
        "extreme": [data.extreme[i] for i in ids],
    }
    for method, samples in method_samples.items():
        lvl_stats[method] = compute_stats(samples)

    base_avg = lvl_stats["base"]["avg_iter"]
    early_avg = lvl_stats["early"]["avg_iter"]
    extreme_avg = lvl_stats["extreme"]["avg_iter"]

    lvl_stats["ratios"] = {
        "base_vs_early": None
        if base_avg in (None, 0) or early_avg is None
        else float((base_avg - early_avg) / base_avg),
        "early_vs_extreme": None
        if early_avg in (None, 0) or extreme_avg is None
        else float((early_avg - extreme_avg) / early_avg),
    }

    return {
        "count": len(ids),
        "query_ids": ids,
        "stats": lvl_stats,
    }


def gather_iter_data(
    dataset: str,
    config_name: str,
    paths: MethodPaths,
    value_key: str,
) -> Optional[IterData]:
    """Load metric arrays for base/early/extreme plus iter_count for binning."""
    extreme_config_dir = paths.extreme / dataset / config_name
    extreme_file = locate_iter_file(
        extreme_config_dir,
        ["HNSWIO_IterDistCount.json", "HNSWIO_IterDistCounts.json"],
    )
    if extreme_file is None:
        return None

    extreme_iters = load_metric_values(extreme_file, "iter_count")
    if extreme_iters is None:
        return None

    extreme_metric = (
        extreme_iters
        if value_key == "iter_count"
        else load_metric_values(extreme_file, value_key)
    )
    if extreme_metric is None:
        return None

    threads, ef, _ = parse_config_dirname(config_name)
    dataset_base_root = paths.base / dataset
    dataset_early_root = paths.early / dataset

    base_config_dir = paths.base / dataset / config_name
    if not base_config_dir.exists():
        match = find_config_with_matching_ef(dataset_base_root, ef)
        if match:
            base_config_dir = match
    base_file = locate_iter_file(base_config_dir, ["HNSWIO_IterDistCount.json"])

    early_config_dir = paths.early / dataset / config_name
    if not early_config_dir.exists():
        match = find_config_with_matching_ef(dataset_early_root, ef)
        if match:
            early_config_dir = match
    early_file = locate_iter_file(
        early_config_dir,
        ["HNSWIO_IterDistCounts.json", "HNSWIO_IterDistCount.json"],
    )

    if not all([base_file, early_file]):
        return None

    base_vals = load_metric_values(base_file, value_key)
    early_vals = load_metric_values(early_file, value_key)
    if base_vals is None or early_vals is None:
        return None

    min_len = min(len(base_vals), len(early_vals), len(extreme_metric), len(extreme_iters))
    if min_len == 0:
        return None

    base_vals = base_vals[:min_len]
    early_vals = early_vals[:min_len]
    extreme_metric = extreme_metric[:min_len]
    extreme_iters = extreme_iters[:min_len]

    query_ids = list(range(min_len))
    return IterData(
        query_ids=query_ids,
        base=base_vals,
        early=early_vals,
        extreme=extreme_metric,
        extreme_iters=extreme_iters,
    )


# ----------------------------------------------------------------------
# Core pipeline
# ----------------------------------------------------------------------

def analyze_dataset_config(
    dataset: str,
    config_name: str,
    iter_data: IterData,
    num_bins: int,
) -> Dict[str, object]:
    """Build bucketed summary for (dataset, config)."""
    iter_bins = make_quantile_bins(iter_data.extreme_iters, num_bins)
    levels = assign_levels(iter_data.extreme_iters, iter_bins)

    buckets: Dict[int, List[int]] = defaultdict(list)
    for qid, lvl in zip(iter_data.query_ids, levels):
        buckets[lvl].append(qid)

    level_summaries = []
    for lvl in range(num_bins):
        level_summaries.append(
            {
                "level": lvl,
                "iter_range": list(iter_bins[lvl]),
                "details": build_bucket_stats(buckets.get(lvl, []), iter_data),
            }
        )

    # Overall summary
    overall = build_bucket_stats(iter_data.query_ids, iter_data)

    return {
        "dataset": dataset,
        "config": config_name,
        "num_queries": len(iter_data.query_ids),
        "iter_bins": [list(b) for b in iter_bins],
        "levels": level_summaries,
        "overall": overall,
    }


def run_pipeline(
    paths: MethodPaths,
    num_bins: int,
    datasets_filter: Optional[set[str]],
    value_key: str,
) -> Dict[str, object]:
    """Iterate over all dataset/config pairs found in base directory."""
    summary: Dict[str, object] = {}

    for dataset_dir in sorted(p for p in paths.extreme.iterdir() if p.is_dir()):
        dataset = dataset_dir.name
        if datasets_filter and dataset not in datasets_filter:
            continue

        for config_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            config_name = config_dir.name
            iter_data = gather_iter_data(dataset, config_name, paths, value_key)
            if not iter_data:
                print(f"[WARN] Missing iter data for {dataset}/{config_name}, skipping.")
                continue
            key = f"{dataset}_{config_name}"
            summary[key] = analyze_dataset_config(dataset, config_name, iter_data, num_bins)
            print(f"[OK] Aggregated {key}")

    return summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bucket query difficulty via SearchResults iter-counts "
        "and compare early/base/extreme methods."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("Results"),
        help="Root directory containing SearchResults/EarlyResults/StableResults.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Override base (StableResults) Offset_results directory.",
    )
    parser.add_argument(
        "--early-dir",
        type=Path,
        default=None,
        help="Override early Offset_results directory.",
    )
    parser.add_argument(
        "--extreme-dir",
        type=Path,
        default=None,
        help="Override extreme (SearchResults) Offset_results directory.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset name to include (can repeat). Default: all datasets.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=5,
        help="Number of difficulty bins (based on base iter-count quantiles).",
    )
    parser.add_argument(
        "--value-key",
        type=str,
        default="iter_count",
        choices=("iter_count", "dist_count"),
        help="Which JSON field to summarize for base/early/extreme methods.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("difficulty_summary.json"),
        help="Path to save the JSON summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    results_root = args.results_root.resolve()
    base_dir = (args.base_dir or results_root / "StableResults" / "Offset_results").resolve()
    early_dir = (args.early_dir or results_root / "EarlyResults" / "Offset_results").resolve()
    extreme_dir = (args.extreme_dir or results_root / "SearchResults" / "Offset_results").resolve()

    for path in [base_dir, early_dir, extreme_dir]:
        if not path.exists():
            print(f"[ERROR] Missing directory: {path}")
            return 1

    datasets_filter = set(args.dataset) if args.dataset else None
    paths = MethodPaths(base=base_dir, early=early_dir, extreme=extreme_dir)

    summary = run_pipeline(paths, args.num_bins, datasets_filter, args.value_key)
    if not summary:
        print("[WARN] No datasets processed. Nothing to write.")
        return 0

    summary["_meta"] = {
        "metric_key": args.value_key,
        "difficulty_source": "SearchResults.iter_count",
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Summary saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



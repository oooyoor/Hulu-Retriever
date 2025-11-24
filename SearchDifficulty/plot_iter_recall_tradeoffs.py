#!/usr/bin/env python3
"""
Render difficulty plots that compare early vs base and early vs extreme
for both iteration savings and recall differences.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class MethodPaths:
    base: Path
    early: Path
    extreme: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot iteration + recall deltas for difficulty buckets."
    )
    parser.add_argument("--summary", type=Path, required=True, help="difficulty summary JSON.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("Results"),
        help="Root containing StableResults/EarlyResults/SearchResults.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Override StableResults Offset_results directory.",
    )
    parser.add_argument(
        "--early-dir",
        type=Path,
        default=None,
        help="Override EarlyResults Offset_results directory.",
    )
    parser.add_argument(
        "--extreme-dir",
        type=Path,
        default=None,
        help="Override SearchResults Offset_results directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Results/difficulty_iter_recall_plots"),
        help="Directory for generated figures.",
    )
    parser.add_argument("--dataset", action="append", default=None, help="Datasets to include.")
    parser.add_argument(
        "--format",
        action="append",
        choices=("png", "pdf"),
        help="Figure format(s). Repeat flag to emit multiple formats. Default: png and pdf.",
    )
    return parser.parse_args()


def format_title(dataset: str, config: str) -> str:
    pretty_dataset = dataset.replace("_", " ").replace("-", " ").title()
    return f"{pretty_dataset} ({config})"


def parse_config_dirname(name: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    parts = name.split("_")
    if len(parts) < 3:
        return None, None, None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None, None, None


def locate_metric_file(config_dir: Path, candidate_names: Sequence[str]) -> Optional[Path]:
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
    if target_ef is None or not dataset_dir.exists():
        return None
    for cfg_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        _, ef, _ = parse_config_dirname(cfg_dir.name)
        if ef == target_ef:
            return cfg_dir
    return None


def load_recall_values(path: Path) -> Optional[List[float]]:
    if path is None:
        return None
    with open(path, "r") as f:
        data = json.load(f)
    values: List[float] = []
    for entry in data.get("entries", []):
        if "recall" not in entry:
            return None
        values.append(float(entry["recall"]))
    return values


def resolve_method_paths(args: argparse.Namespace) -> MethodPaths:
    base_dir = args.base_dir or (args.results_root / "StableResults" / "Offset_results")
    early_dir = args.early_dir or (args.results_root / "EarlyResults" / "Offset_results")
    extreme_dir = args.extreme_dir or (args.results_root / "SearchResults" / "Offset_results")
    return MethodPaths(base=base_dir, early=early_dir, extreme=extreme_dir)


def gather_recall_arrays(paths: MethodPaths, dataset: str, config: str) -> Optional[Dict[str, List[float]]]:
    _, ef, _ = parse_config_dirname(config)

    def load_method(method_root: Path, allow_fallback: bool) -> Optional[List[float]]:
        cfg_dir = method_root / dataset / config
        if allow_fallback and not cfg_dir.exists():
            match = find_config_with_matching_ef(method_root / dataset, ef)
            if match:
                cfg_dir = match
        recall_file = locate_metric_file(cfg_dir, ["HNSWIO_Recall.json"])
        return load_recall_values(recall_file) if recall_file else None

    base_vals = load_method(paths.base, allow_fallback=True)
    early_vals = load_method(paths.early, allow_fallback=True)
    extreme_vals = load_method(paths.extreme, allow_fallback=False)
    if not all([base_vals, early_vals, extreme_vals]):
        return None
    min_len = min(len(base_vals), len(early_vals), len(extreme_vals))
    return {
        "base": base_vals[:min_len],
        "early": early_vals[:min_len],
        "extreme": extreme_vals[:min_len],
    }


def average_subset(values: Sequence[float], indices: Iterable[int]) -> Optional[float]:
    subset = [values[i] for i in indices if 0 <= i < len(values)]
    if not subset:
        return None
    return float(sum(subset) / len(subset))


def prepare_recall_deltas(levels: List[dict], recalls: Dict[str, List[float]]) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    base_vs_early: List[Optional[float]] = []
    early_vs_extreme: List[Optional[float]] = []
    for lvl in levels:
        ids = lvl["details"]["query_ids"]
        avg_base = average_subset(recalls["base"], ids)
        avg_early = average_subset(recalls["early"], ids)
        avg_extreme = average_subset(recalls["extreme"], ids)
        base_vs_early.append(
            None if avg_base is None or avg_early is None else float(avg_early - avg_base)
        )
        early_vs_extreme.append(
            None if avg_early is None or avg_extreme is None else float(avg_early - avg_extreme)
        )
    return base_vs_early, early_vs_extreme


def plot_dual_metric(
    key: str,
    dataset: str,
    config: str,
    x_labels: List[str],
    iter_values: List[float],
    recall_deltas: List[Optional[float]],
    figure_title: str,
    iter_label: str,
    recall_label: str,
    color: str,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    y_iter = np.array(iter_values, dtype=float) * 100.0
    y_recall = np.array(
        [np.nan if val is None else val * 100.0 for val in recall_deltas], dtype=float
    )

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    fig.suptitle(f"{figure_title}\n{format_title(dataset, config)}")

    axes[0].bar(x_labels, y_iter, color=color, alpha=0.85)
    axes[0].axhline(0.0, color="#555555", linestyle="--", linewidth=1, alpha=0.5)
    axes[0].set_ylabel(iter_label)
    for idx, height in enumerate(y_iter):
        if np.isnan(height):
            continue
        axes[0].text(idx, height, f"{height:.1f}%", ha="center", va="bottom", fontsize=9)

    axes[1].bar(x_labels, y_recall, color="#6C5CE7", alpha=0.85)
    axes[1].axhline(0.0, color="#555555", linestyle="--", linewidth=1, alpha=0.5)
    axes[1].set_ylabel(recall_label)
    axes[1].set_xlabel("Difficulty Level (SearchResults quantiles)")
    for idx, height in enumerate(y_recall):
        if np.isnan(height):
            continue
        axes[1].text(idx, height, f"{height:.2f}pp", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for fmt in formats:
        out_path = output_dir / f"{key}.{fmt}"
        plt.savefig(out_path, format=fmt, dpi=300, bbox_inches="tight")
        saved_paths.append(out_path)
    plt.close(fig)
    for path in saved_paths:
        print(f"[OK] Saved plot: {path}")


def main() -> int:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    datasets_filter = set(args.dataset) if args.dataset else None
    paths = resolve_method_paths(args)

    formats = args.format or ["png", "pdf"]

    for key, info in summary.items():
        if key.startswith("_"):
            continue
        dataset = info.get("dataset")
        if datasets_filter and dataset not in datasets_filter:
            continue
        config = info.get("config", "")
        levels = info.get("levels", [])
        if not levels:
            continue

        recall_arrays = gather_recall_arrays(paths, dataset, config)
        if recall_arrays is None:
            print(f"[WARN] Missing recall data for {key}, skipping.")
            continue

        recall_be, recall_ee = prepare_recall_deltas(levels, recall_arrays)
        x_labels = [f"L{lvl['level']}" for lvl in levels]
        iter_base_vs_early = [
            np.nan
            if (lvl["details"]["stats"]["ratios"].get("base_vs_early") is None)
            else lvl["details"]["stats"]["ratios"]["base_vs_early"]
            for lvl in levels
        ]
        iter_early_vs_extreme = [
            np.nan
            if (lvl["details"]["stats"]["ratios"].get("early_vs_extreme") is None)
            else lvl["details"]["stats"]["ratios"]["early_vs_extreme"]
            for lvl in levels
        ]

        plot_dir = args.output_dir / key
        plot_dual_metric(
            key=f"{key}_base_vs_early",
            dataset=dataset,
            config=config,
            x_labels=x_labels,
            iter_values=iter_base_vs_early,
            recall_deltas=recall_be,
            figure_title="Early vs Base",
            iter_label="Iter Improvement (%)",
            recall_label="Recall Delta (pp)",
            color="#4FA3C0",
            output_dir=plot_dir,
            formats=formats,
        )
        plot_dual_metric(
            key=f"{key}_early_vs_extreme",
            dataset=dataset,
            config=config,
            x_labels=x_labels,
            iter_values=iter_early_vs_extreme,
            recall_deltas=recall_ee,
            figure_title="Early vs Extreme",
            iter_label="Iter Gap (%)",
            recall_label="Recall Delta (pp)",
            color="#E67E22",
            output_dir=plot_dir,
            formats=formats,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


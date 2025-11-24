#!/usr/bin/env python3
"""
Visualize improvement ratios per dataset/difficulty bucket.

Input JSON must be produced by compare_query_difficulty.py.
For every dataset_config entry the script plots:
    - (base - early) / base
    - (early - extreme) / early
as percentage bars for each difficulty level.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize per-difficulty improvement ratios from difficulty summary JSON."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to difficulty_summary JSON produced by compare_query_difficulty.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("difficulty_plots"),
        help="Directory to store the generated plots.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset name to include (can repeat). Default: all datasets.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=("png", "pdf"),
        help="Figure format.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> Dict[str, dict]:
    with open(path, "r") as f:
        return json.load(f)


def format_title(dataset: str, config: str) -> str:
    pretty_dataset = dataset.replace("_", " ").replace("-", " ").title()
    return f"{pretty_dataset} ({config})"


def plot_dataset(key: str, info: dict, out_dir: Path, fmt: str, metric_label: str):
    dataset = info.get("dataset", "unknown")
    config = info.get("config", "unknown")
    levels = info.get("levels", [])

    if not levels:
        return

    x_labels: List[str] = []
    base_vs_early: List[float] = []
    early_vs_extreme: List[float] = []

    for lvl in levels:
        lvl_idx = lvl.get("level")
        ratios = (lvl.get("details") or {}).get("stats", {}).get("ratios", {})
        base_ratio = ratios.get("base_vs_early")
        early_ratio = ratios.get("early_vs_extreme")

        x_labels.append(f"L{lvl_idx}")
        base_vs_early.append(100.0 * base_ratio if base_ratio is not None else np.nan)
        early_vs_extreme.append(100.0 * early_ratio if early_ratio is not None else np.nan)

    x = np.arange(len(x_labels))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.title(format_title(dataset, config))
    plt.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--", alpha=0.6)

    bars1 = plt.bar(
        x - width / 2,
        base_vs_early,
        width,
        label="(base - early) / base",
        color="#4FA3C0",
    )
    bars2 = plt.bar(
        x + width / 2,
        early_vs_extreme,
        width,
        label="(early - extreme) / early",
        color="#E67E22",
    )

    def annotate(bars):
        for bar in bars:
            height = bar.get_height()
            if np.isnan(height):
                continue
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=9,
                color="#2C3E50",
            )

    annotate(bars1)
    annotate(bars2)

    plt.xticks(x, x_labels)
    plt.ylabel(f"Improvement (%) on {metric_label}")
    plt.xlabel("Difficulty Level (SearchResults quantiles)")
    plt.legend()
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{key}.{fmt}"
    plt.savefig(out_path, format=fmt, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved plot: {out_path}")


def main() -> int:
    args = parse_args()
    summary = load_summary(args.summary)
    meta = summary.get("_meta", {})
    metric_label = meta.get("metric_key", "iter_count")
    datasets_filter = set(args.dataset) if args.dataset else None

    for key, info in summary.items():
        if key.startswith("_"):
            continue
        dataset = info.get("dataset")
        if datasets_filter and dataset not in datasets_filter:
            continue
        plot_dataset(key, info, args.output_dir, args.format, metric_label)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



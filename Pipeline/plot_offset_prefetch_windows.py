#!/usr/bin/env python3
"""
Plot the optimization space captured in Offset_results prefetch.json files.

The script produces:
1. A stacked bar chart comparing avg_start_prefetch, optimization window, and
   remaining iterations for every dataset under Offset_results.
2. One histogram per dataset showing the per-query ratio of optimization window
   to iter_count (i.e., (last_prefetch - start_prefetch) / iter_count).
3. A bar chart comparing avg_io_cnt across all datasets.
4. One histogram per dataset showing the per-query io_cnt distribution.

Usage:
    python plot_offset_prefetch_windows.py \
        --root /home/zqf/Hulu-Retriever/Pipeline/prefetch_results/raw_results/Offset_results \
        --output /home/zqf/Hulu-Retriever/Pipeline/prefetch_results/plots
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot optimization window summaries from Offset_results prefetch.json files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing dataset folders under Offset_results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where plots will be written.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of bins for histogram plots (default: 20).",
    )
    return parser.parse_args()


def find_prefetch_files(root: Path) -> Dict[str, Path]:
    dataset_files: Dict[str, Path] = {}
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        matches = sorted(dataset_dir.rglob("prefetch.json"))
        if matches:
            dataset_files[dataset_dir.name] = matches[0]
    return dataset_files


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return math.nan
    return numerator / denominator


def load_dataset_metrics(prefetch_path: Path) -> Dict[str, float]:
    with prefetch_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    avg = payload["avgcost"]
    entries = payload.get("entries", [])

    avg_iter = float(avg["avg_iter_count"])
    avg_start = float(avg["avg_start_prefetch"])
    avg_last = float(avg["avg_last_prefetch"])
    avg_io_cnt = float(avg.get("avg_io_cnt", 0.0))

    # Optimization window: the period where prefetch is active (waste window)
    # Longer window = more IO waste
    avg_window = max(avg_last - avg_start, 0.0)
    avg_ratio = safe_ratio(avg_window, avg_iter)
    
    # Performance ratio: avg_io_cnt / 100 (theoretical optimal is 100)
    # Higher ratio = worse performance (more IO overhead)
    performance_ratio = safe_ratio(avg_io_cnt, 100.0)
    io_overhead = max(avg_io_cnt - 100.0, 0.0)  # Extra IO beyond theoretical optimal

    per_query_ratios: List[float] = []
    per_query_io_cnts: List[float] = []
    for entry in entries:
        iter_count = float(entry.get("iter_count", 0.0))
        start = float(entry.get("start_prefetch", 0.0))
        last = float(entry.get("last_prefetch", 0.0))
        span = max(last - start, 0.0)
        ratio = safe_ratio(span, iter_count)
        if not math.isnan(ratio):
            per_query_ratios.append(max(0.0, min(1.0, ratio)))
        
        io_cnt = float(entry.get("io_cnt", 0.0))
        if io_cnt > 0:
            per_query_io_cnts.append(io_cnt)

    # Calculate per-query performance ratios
    per_query_performance_ratios: List[float] = []
    per_query_windows: List[float] = []
    for entry in entries:
        io_cnt = float(entry.get("io_cnt", 0.0))
        if io_cnt > 0:
            perf_ratio = safe_ratio(io_cnt, 100.0)
            if not math.isnan(perf_ratio):
                per_query_performance_ratios.append(perf_ratio)
        
        start = float(entry.get("start_prefetch", 0.0))
        last = float(entry.get("last_prefetch", 0.0))
        window = max(last - start, 0.0)
        per_query_windows.append(window)

    return {
        "avg_iter": avg_iter,
        "avg_start": avg_start,
        "avg_last": avg_last,
        "avg_window": avg_window,
        "avg_ratio": avg_ratio,
        "ratios": per_query_ratios,
        "avg_io_cnt": avg_io_cnt,
        "io_cnts": per_query_io_cnts,
        "performance_ratio": performance_ratio,
        "io_overhead": io_overhead,
        "per_query_performance_ratios": per_query_performance_ratios,
        "per_query_windows": per_query_windows,
    }


def plot_dataset_summary(dataset_metrics: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    """Plot prefetch window analysis showing overlap execution space and theoretical optimization space."""
    if not dataset_metrics:
        raise ValueError("No dataset metrics found to plot.")

    # Use professional color scheme for paper (blue-gray palette)
    fig, ax = plt.subplots(figsize=(max(10, len(dataset_metrics) * 1.6), 6))

    datasets = list(dataset_metrics.keys())
    indices = np.arange(len(datasets))

    starts = np.array([dataset_metrics[d]["avg_start"] for d in datasets])
    windows = np.array([dataset_metrics[d]["avg_window"] for d in datasets])  # Overlap execution space
    tails = np.array([max(dataset_metrics[d]["avg_iter"] - dataset_metrics[d]["avg_last"], 0.0) for d in datasets])
    ratios = np.array([dataset_metrics[d]["avg_ratio"] for d in datasets])

    # Professional color scheme: light blue, orange, dark blue
    color_before = "#6C9BD2"  # Light blue
    color_overlap = "#F4A460"  # Sandy brown / orange
    color_optimization = "#4A90A4"  # Teal blue

    # Before prefetch starts
    ax.bar(indices, starts, label="Before Prefetch Start", color=color_before, alpha=0.85, edgecolor="white", linewidth=0.5)
    # Overlap execution space (start to last) - where prefetch and search overlap
    ax.bar(indices, windows, bottom=starts, label="Overlap Execution Space", color=color_overlap, alpha=0.85, edgecolor="white", linewidth=0.5)
    # After prefetch ends (last to end) - remaining iterations after prefetch should end
    ax.bar(indices, tails, bottom=starts + windows, label="After Prefetch End", color=color_optimization, alpha=0.85, edgecolor="white", linewidth=0.5)

    # Add overlap space percentage labels
    for idx, (window, ratio) in enumerate(zip(windows, ratios)):
        if math.isnan(ratio) or window <= 0:
            continue
        ax.text(
            idx,
            starts[idx] + windows[idx] / 2,
            f"{ratio:.1%}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )

    ax.set_xticks(indices)
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=11, fontweight="bold")
    ax.set_ylabel("Iterations", fontsize=12, fontweight="bold")
    ax.set_title("Prefetch Execution Timeline Analysis", fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
    
    # Set background to white for paper
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    fig.tight_layout()

    # Save as PNG
    output_path_png = output_dir / "offset_dataset_stack.png"
    fig.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Save as PDF for paper insertion
    output_path_pdf = output_dir / "offset_dataset_stack.pdf"
    fig.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    
    plt.close(fig)


def plot_histograms(
    dataset_metrics: Dict[str, Dict[str, float]], output_dir: Path, bins: int
) -> List[Tuple[str, Path]]:
    saved_paths: List[Tuple[str, Path]] = []
    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    for dataset, metrics in dataset_metrics.items():
        ratios = metrics["ratios"]
        if not ratios:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(ratios, bins=bin_edges, color="#5ca4a9", edgecolor="white", alpha=0.9)
        ax.set_xlabel("Optimization window / iter_count")
        ax.set_ylabel("Query count")
        ax.set_title(f"{dataset}: query-level optimization ratio")
        ax.set_xlim(0, 1)
        ax.grid(axis="y", alpha=0.4)
        fig.tight_layout()

        output_path = output_dir / f"{dataset}_optimization_ratio_hist.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

        saved_paths.append((dataset, output_path))

    return saved_paths


def plot_io_cnt_comparison(dataset_metrics: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    """Plot average IO count comparison across all datasets."""
    if not dataset_metrics:
        raise ValueError("No dataset metrics found to plot.")

    fig, ax = plt.subplots(figsize=(max(10, len(dataset_metrics) * 1.6), 6))

    datasets = list(dataset_metrics.keys())
    indices = np.arange(len(datasets))
    io_cnts = np.array([dataset_metrics[d]["avg_io_cnt"] for d in datasets])

    bars = ax.bar(indices, io_cnts, color="#9b59b6", alpha=0.8, edgecolor="black", linewidth=1.2)
    
    # Add theoretical optimal line at 100
    ax.axhline(y=100, color="#e74c3c", linestyle="--", linewidth=2, label="Theoretical Optimal (100)", zorder=0)
    
    # Add value labels on top of bars
    for idx, (bar, io_cnt) in enumerate(zip(bars, io_cnts)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{io_cnt:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(indices)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("Average IO Count", fontsize=12)
    ax.set_title("Average IO Count Comparison Across Datasets (Optimal = 100)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()

    output_path = output_dir / "io_cnt_comparison.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_performance_ratio_comparison(dataset_metrics: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    """Plot performance ratio (avg_io_cnt / 100) comparison across all datasets."""
    if not dataset_metrics:
        raise ValueError("No dataset metrics found to plot.")

    fig, ax = plt.subplots(figsize=(max(10, len(dataset_metrics) * 1.6), 6))

    datasets = list(dataset_metrics.keys())
    indices = np.arange(len(datasets))
    performance_ratios = np.array([dataset_metrics[d]["performance_ratio"] for d in datasets])
    io_overheads = np.array([dataset_metrics[d]["io_overhead"] for d in datasets])

    # Create bars with different colors based on performance
    colors = ["#2ecc71" if ratio <= 1.0 else "#e74c3c" for ratio in performance_ratios]
    bars = ax.bar(indices, performance_ratios, color=colors, alpha=0.8, edgecolor="black", linewidth=1.2)
    
    # Add theoretical optimal line at 1.0 (100%)
    ax.axhline(y=1.0, color="#34495e", linestyle="--", linewidth=2, label="Theoretical Optimal (1.0)", zorder=0)
    
    # Add value labels on top of bars
    for idx, (bar, ratio, overhead) in enumerate(zip(bars, performance_ratios, io_overheads)):
        height = bar.get_height()
        label_text = f"{ratio:.3f}\n(+{overhead:.1f})" if overhead > 0 else f"{ratio:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            label_text,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(indices)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("Performance Ratio (avg_io_cnt / 100)", fontsize=12)
    ax.set_title("Prefetch Performance Ratio: Lower is Better (Optimal = 1.0)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()

    output_path = output_dir / "performance_ratio_comparison.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_io_cnt_histograms(
    dataset_metrics: Dict[str, Dict[str, float]], output_dir: Path, bins: int
) -> List[Tuple[str, Path]]:
    """Plot IO count distribution histogram for each dataset."""
    saved_paths: List[Tuple[str, Path]] = []

    for dataset, metrics in dataset_metrics.items():
        io_cnts = metrics["io_cnts"]
        if not io_cnts:
            continue

        # Determine appropriate bin range based on data
        min_io = min(io_cnts)
        max_io = max(io_cnts)
        # Use integer bins if range is reasonable
        if max_io - min_io < 200:
            bin_edges = np.arange(int(min_io), int(max_io) + 2, max(1, int((max_io - min_io) / bins)))
        else:
            bin_edges = np.linspace(min_io, max_io, bins + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        n, bins_used, patches = ax.hist(io_cnts, bins=bin_edges, color="#e74c3c", edgecolor="white", alpha=0.8)
        
        # Add vertical line for average
        avg_io = metrics["avg_io_cnt"]
        ax.axvline(avg_io, color="#2c3e50", linestyle="--", linewidth=2, label=f"Average: {avg_io:.2f}")
        
        ax.set_xlabel("IO Count", fontsize=11)
        ax.set_ylabel("Query Count", fontsize=11)
        ax.set_title(f"{dataset}: IO Count Distribution", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.4, linestyle="--")
        fig.tight_layout()

        output_path = output_dir / f"{dataset}_io_cnt_hist.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

        saved_paths.append((dataset, output_path))

    return saved_paths


def plot_waste_window_histograms(
    dataset_metrics: Dict[str, Dict[str, float]], output_dir: Path, bins: int
) -> List[Tuple[str, Path]]:
    """Plot waste window (start to last) distribution histogram for each dataset."""
    saved_paths: List[Tuple[str, Path]] = []

    for dataset, metrics in dataset_metrics.items():
        windows = metrics["per_query_windows"]
        if not windows:
            continue

        # Determine appropriate bin range
        min_window = min(windows)
        max_window = max(windows)
        if max_window - min_window < 50:
            bin_edges = np.arange(int(min_window), int(max_window) + 2, max(1, int((max_window - min_window) / bins)))
        else:
            bin_edges = np.linspace(min_window, max_window, bins + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        n, bins_used, patches = ax.hist(windows, bins=bin_edges, color="#e74c3c", edgecolor="white", alpha=0.8)
        
        # Add vertical line for average
        avg_window = metrics["avg_window"]
        ax.axvline(avg_window, color="#2c3e50", linestyle="--", linewidth=2, label=f"Average: {avg_window:.2f}")
        
        ax.set_xlabel("Waste Window Size (iterations)", fontsize=11)
        ax.set_ylabel("Query Count", fontsize=11)
        ax.set_title(f"{dataset}: Waste Window Distribution (Longer = More IO Waste)", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.4, linestyle="--")
        fig.tight_layout()

        output_path = output_dir / f"{dataset}_waste_window_hist.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

        saved_paths.append((dataset, output_path))

    return saved_paths


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_files = find_prefetch_files(root)
    if not dataset_files:
        raise FileNotFoundError(f"No prefetch.json files found under {root}")

    dataset_metrics: Dict[str, Dict[str, float]] = {}
    for dataset, path in dataset_files.items():
        dataset_metrics[dataset] = load_dataset_metrics(path)

    plot_dataset_summary(dataset_metrics, output_dir)
    plot_histograms(dataset_metrics, output_dir, args.bins)
    plot_io_cnt_comparison(dataset_metrics, output_dir)
    plot_io_cnt_histograms(dataset_metrics, output_dir, args.bins)
    plot_performance_ratio_comparison(dataset_metrics, output_dir)
    plot_waste_window_histograms(dataset_metrics, output_dir, args.bins)


if __name__ == "__main__":
    main()


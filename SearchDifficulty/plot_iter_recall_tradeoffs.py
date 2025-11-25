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


# def plot_dual_metric(
#     key: str,
#     dataset: str,
#     config: str,
#     x_labels: List[str],
#     iter_values: List[float],
#     recall_deltas: List[Optional[float]],
#     figure_title: str,
#     iter_label: str,
#     recall_label: str,
#     color: str,
#     output_dir: Path,
#     formats: Sequence[str],
# ) -> None:
#     y_iter = np.array(iter_values, dtype=float) * 100.0
#     y_recall = np.array(
#         [np.nan if val is None else val * 100.0 for val in recall_deltas], dtype=float
#     )

#     fig, ax_iter = plt.subplots(figsize=(6.0, 3.4))
#     ax_recall = ax_iter.twinx()
#     fig.suptitle(f"{figure_title} • {format_title(dataset, config)}", fontsize=11, y=1.02)

#     bar_positions = np.arange(len(x_labels))
#     bars = ax_iter.bar(
#         bar_positions,
#         y_iter,
#         color=color,
#         alpha=0.85,
#         width=0.35,
#         label=iter_label,
#         edgecolor="#1B4F72",
#         linewidth=0.5,
#     )
#     ax_iter.axhline(0.0, color="#777777", linestyle="--", linewidth=0.9, alpha=0.6)
#     ax_iter.set_ylabel(iter_label, color=color, fontsize=10)
#     ax_iter.tick_params(axis="y", colors=color)
#     ax_iter.set_xticks(bar_positions)
#     ax_iter.set_xticklabels(x_labels, fontsize=9)

#     recall_line = ax_recall.plot(
#         bar_positions,
#         y_recall,
#         color="#6C5CE7",
#         marker="D",
#         markersize=5,
#         linewidth=1.8,
#         label=recall_label,
#     )[0]
#     ax_recall.axhline(0.0, color="#999999", linestyle=":", linewidth=0.8, alpha=0.6)
#     ax_recall.set_ylabel(recall_label, color="#6C5CE7", fontsize=10)
#     ax_recall.tick_params(axis="y", colors="#6C5CE7")

#     for bar in bars:
#         height = bar.get_height()
#         if np.isnan(height):
#             continue
#         ax_iter.text(
#             bar.get_x() + bar.get_width() / 2,
#             height,
#             f"{height:.1f}%",
#             ha="center",
#             va="bottom" if height >= 0 else "top",
#             fontsize=8,
#             color="#0B2545",
#         )
#     for idx, val in enumerate(y_recall):
#         if np.isnan(val):
#             continue
#         ax_recall.text(
#             bar_positions[idx],
#             val,
#             f"{val:.2f}pp",
#             ha="center",
#             va="bottom" if val >= 0 else "top",
#             fontsize=8,
#             color="#4A3F8C",
#             rotation=0,
#         )

#     ax_iter.set_xlabel("Difficulty Level (SearchResults quantiles)", fontsize=10)
#     ax_iter.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

#     handles = [bars, recall_line]
#     labels = [iter_label, recall_label]
#     ax_iter.legend(
#         handles,
#         labels,
#         loc="upper left",
#         fontsize=9,
#         frameon=False,
#         bbox_to_anchor=(0.0, 1.15),
#         ncol=2,
#         columnspacing=1.5,
#     )
#     fig.tight_layout()

#     output_dir.mkdir(parents=True, exist_ok=True)
#     saved_paths = []
#     for fmt in formats:
#         out_path = output_dir / f"{key}.{fmt}"
#         fig.savefig(out_path, format=fmt, dpi=300, bbox_inches="tight")
#         saved_paths.append(out_path)
#     plt.close(fig)
#     for path in saved_paths:
#         print(f"[OK] Saved plot: {path}")

def compute_optimal_figsize(num_buckets: int) -> tuple:
    """
    自动计算最佳图像尺寸，使得无论 bucket 个数多少、注释多少，都不会重叠。
    """
    # 每个 bucket 分配宽度（经验值）
    per_bucket_width = 0.9  # 每个柱子大约需要 0.9 inch 才不会挤
    base_width = 2.8        # 图像基础宽度

    width = base_width + num_buckets * per_bucket_width

    # 高度可以根据是否有 recall 双轴动态调节
    height = 2.8  # 学术图默认比较扁平，2.6~3.0 都很合理

    # 限制最大宽度，避免横向太夸张
    width = min(width, 6.8)  # 单栏 3.3、双栏 6.9，这里留一点 margin

    return (width, height)


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
):
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })
    W, H = compute_optimal_figsize(len(x_labels))
    fig, ax_iter = plt.subplots(figsize=(W, H))
    # ---- convert to % ----
    y_iter = np.array(iter_values) * 100.0
    y_recall = np.array([np.nan if v is None else v * 100.0 for v in recall_deltas])

    # ---- Dynamic figure size (auto) ----
    # W = max(3.6, 0.6 * len(x_labels))  # 避免 x 轴挤爆；随 bucket 增加自动加宽
    # fig, ax_iter = plt.subplots(figsize=(W, 2.6))
    ax_recall = ax_iter.twinx()

    fig.suptitle(figure_title, y=1.03, fontsize=9)

    pos = np.arange(len(x_labels))

    # === BAR ===
    bars = ax_iter.bar(
        pos,
        y_iter,
        width=0.55,
        color=color,
        alpha=0.88,
        edgecolor="#1A374D",
        linewidth=0.4,
        label=iter_label,
    )

    ax_iter.set_ylabel(iter_label)
    ax_iter.set_xticks(pos)
    ax_iter.set_xticklabels(x_labels)

    # === LINE ===
    recall_line = ax_recall.plot(
        pos,
        y_recall,
        "-o",
        markersize=3,
        linewidth=1.2,
        color="#6C5CE7",
        label=recall_label,
    )[0]

    ax_recall.set_ylabel(recall_label)

    # === GRID (iter axis only) ===
    ax_iter.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)

    # ---------------------------------------------------------
    #  SMART LABEL POSITIONS (绝不遮挡)
    # ---------------------------------------------------------

    # 1) annotate bars above/below with dynamic offset
    ylim_iter = ax_iter.get_ylim()
    iter_range = ylim_iter[1] - ylim_iter[0]
    bar_offset = iter_range * 0.03   # 3% of y-range

    for bar in bars:
        h = bar.get_height()
        if np.isnan(h): 
            continue
        y = h + (bar_offset if h >= 0 else -bar_offset)
        ax_iter.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{h:.1f}%",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=7,
        )

    # 2) annotate recall with dynamic offset (avoids bar area)
    ylim_recall = ax_recall.get_ylim()
    recall_range = ylim_recall[1] - ylim_recall[0]
    rec_offset = recall_range * 0.04  # 4% of y-range

    for i, val in enumerate(y_recall):
        if np.isnan(val): 
            continue

        # 如果 recall 值落在 bar 的“视觉区域”，自适应偏移
        # bar_top/bottom in recall-scale coordinates
        bar_height = y_iter[i]
        bar_y_min = ax_recall.transData.inverted().transform(
            ax_iter.transData.transform((0, min(0, bar_height)))
        )[1]
        bar_y_max = ax_recall.transData.inverted().transform(
            ax_iter.transData.transform((0, max(0, bar_height)))
        )[1]

        # 如果点落入柱子区域 → 自动往上/下移动
        if bar_y_min <= val <= bar_y_max:
            if val >= 0:
                val_adj = val + rec_offset * 2.5
            else:
                val_adj = val - rec_offset * 2.5
        else:
            val_adj = val + (rec_offset if val >= 0 else -rec_offset)

        ax_recall.text(
            pos[i],
            val_adj,
            f"{val:.2f}",
            ha="center",
            va="bottom" if val_adj >= val else "top",
            fontsize=7,
            color="#6C5CE7",
        )

    # === Legend ===
    ax_iter.legend(
        [bars, recall_line],
        [iter_label, recall_label],
        loc="upper right",
        frameon=False,
        handlelength=1.8,
    )

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{key}.{fmt}", dpi=320, bbox_inches="tight")

    plt.close(fig)

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


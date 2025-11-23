#!/usr/bin/env python3
"""
Search difficulty analysis with per-dataset quantile bins.

- For each (dataset, EF):
    * recall = 1.0  -> binned into num_bins levels by iter_count (dataset-specific quantile bins)
    * recall < 1.0  -> one extra level "Recall < 1" (hard queries)

Outputs:
    - difficulty.png (double-column friendly figure)
    - analysis.txt   (detailed text report)
    - analysis.md    (Markdown report)
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Utils
# ============================================================

def parse_config_dirname(name: str) -> Tuple[int, int, int]:
    """Parse 'threads_ef_iodepth'."""
    parts = name.split("_")
    if len(parts) < 3:
        return None, None, None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None, None, None


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def collect_recall1_iters_for_dataset(dataset_dir: Path) -> List[int]:
    """Collect all iter_count for recall=1.0 within a dataset (across all configs)."""
    iters = []
    for config_dir in dataset_dir.iterdir():
        if not config_dir.is_dir():
            continue
        for query_cnt_dir in config_dir.iterdir():
            if not query_cnt_dir.is_dir():
                continue
            for repeat_dir in query_cnt_dir.iterdir():
                if not repeat_dir.is_dir():
                    continue

                rec = load_json(repeat_dir / "HNSWIO_Recall.json")
                itc = load_json(repeat_dir / "HNSWIO_IterDistCount.json")
                if rec is None or itc is None:
                    continue

                for e_r, e_i in zip(rec.get("entries", []), itc.get("entries", [])):
                    if e_r.get("recall", 0.0) >= 1.0 - 1e-6:
                        iters.append(int(e_i.get("iter_count", 0)))

    return iters


def make_quantile_bins(iters: List[int], num_bins: int) -> List[Tuple[int, int]]:
    """Dataset-specific quantile bins for iter_count.

    Always returns num_bins intervals (some may be degenerate if values are repeated).
    """
    if not iters:
        return [(0, 0)] * num_bins

    vals = np.array(iters, dtype=float)
    qs = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(vals, qs)

    # Convert to int and ensure non-decreasing
    edges = np.round(edges).astype(int)
    bins = []
    for i in range(num_bins):
        lo = int(edges[i])
        hi = int(edges[i + 1])
        if hi < lo:
            hi = lo
        bins.append((lo, hi))
    return bins


# ============================================================
# Core analysis
# ============================================================

def analyze_dataset(dataset_dir: Path, iter_bins: List[Tuple[int, int]]) -> Dict[str, dict]:
    """Analyze one dataset with given iter_bins (dataset-specific).

    Returns dict: key = "dataset_efXXX"
    """
    num_bins = len(iter_bins)
    results: Dict[str, dict] = {}
    recall1_iters_per_key: Dict[str, List[int]] = defaultdict(list)

    dataset_name = dataset_dir.name

    for config_dir in dataset_dir.iterdir():
        if not config_dir.is_dir():
            continue

        threads, ef, iodepth = parse_config_dirname(config_dir.name)
        if ef is None:
            continue

        key = f"{dataset_name}_ef{ef}"
        if key not in results:
            results[key] = {
                "dataset": dataset_name,
                "ef": ef,
                "threads": threads,
                "iodepth": iodepth,
                "levels": defaultdict(int),  # 0..num_bins (num_bins = recall<1)
                "total": 0,
                "bin_ranges": iter_bins,
                "avg_iter": None,
            }

        for query_cnt_dir in config_dir.iterdir():
            if not query_cnt_dir.is_dir():
                continue
            for repeat_dir in query_cnt_dir.iterdir():
                if not repeat_dir.is_dir():
                    continue

                rec = load_json(repeat_dir / "HNSWIO_Recall.json")
                itc = load_json(repeat_dir / "HNSWIO_IterDistCount.json")
                if rec is None or itc is None:
                    continue

                rec_entries = rec.get("entries", [])
                it_entries = itc.get("entries", [])
                if len(rec_entries) != len(it_entries):
                    continue

                for e_r, e_i in zip(rec_entries, it_entries):
                    recall = float(e_r.get("recall", 0.0))
                    iters = int(e_i.get("iter_count", 0))

                    results[key]["total"] += 1

                    # recall<1.0 -> highest level
                    if recall < 1.0 - 1e-6:
                        level = num_bins  # last level
                        results[key]["levels"][level] += 1
                        continue

                    # recall=1.0 -> bin by iter_count
                    recall1_iters_per_key[key].append(iters)
                    assigned = False
                    for idx, (lo, hi) in enumerate(iter_bins):
                        if lo <= iters <= hi:
                            results[key]["levels"][idx] += 1
                            assigned = True
                            break
                    if not assigned:
                        # if something goes outside (shouldn't happen), put into last numeric bin
                        results[key]["levels"][num_bins - 1] += 1

    # compute avg_iter for each key
    for key, info in results.items():
        iters = recall1_iters_per_key.get(key, [])
        if iters:
            info["avg_iter"] = sum(iters) / len(iters)
        else:
            info["avg_iter"] = None

    return results


# ============================================================
# Reports: TXT + MD
# ============================================================

def write_reports(all_results: Dict[str, dict], txt_path: str, md_path: str):
    keys = sorted(all_results.keys(), key=lambda k: (all_results[k]["dataset"], all_results[k]["ef"]))

    txt_lines: List[str] = []
    md_lines: List[str] = ["# Search Difficulty Report\n"]

    for key in keys:
        r = all_results[key]
        bins = r["bin_ranges"]
        num_bins = len(bins)
        total = r["total"]

        header = (
            "================================================================================\n"
            f"Dataset: {r['dataset']}\n"
            f"Search EF: {r['ef']}\n"
            f"Total Queries: {total}\n"
            f"Threads: {r['threads']}, IO Depths: {r['iodepth']}\n"
            "--------------------------------------------------------------------------------"
        )
        txt_lines.append(header)

        md_lines.append(f"## {r['dataset']} (EF={r['ef']})")
        md_lines.append(f"- Total Queries: **{total}**")
        md_lines.append(f"- Threads: {r['threads']}, IO Depths: {r['iodepth']}")

        # per-level stats
        recall1_total = 0
        for lvl in range(num_bins + 1):
            count = r["levels"].get(lvl, 0)
            pct = (count / total * 100.0) if total > 0 else 0.0

            if lvl < num_bins:
                lo, hi = bins[lvl]
                line = f"Level {lvl:>2} [{lo:>4}-{hi:<4}]: count={count:>6}, pct={pct:6.2f}%"
            else:
                line = f"Recall < 1.0 (Level {lvl}): count={count:>6}, pct={pct:6.2f}%"

            txt_lines.append("  " + line)
            md_lines.append(f"- `{line}`")

            if lvl < num_bins:
                recall1_total += count

        overall = (recall1_total / total * 100.0) if total > 0 else 0.0
        avg_iter = r["avg_iter"]
        avg_iter_str = f"{avg_iter:.2f}" if avg_iter is not None else "N/A"
        txt_lines.append(
            f"\n  Overall: {recall1_total}/{total} queries achieved recall=1.0 ({overall:.2f}%), "
            f"avg_iter (recall=1) = {avg_iter_str}\n"
        )
        md_lines.append(
            f"\nOverall: **{overall:.2f}%** queries achieve recall=1.0, "
            f"avg_iter (recall=1) = **{avg_iter_str}**\n"
        )

    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines))
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"[OK] Text report saved to: {txt_path}")
    print(f"[OK] Markdown report saved to: {md_path}")


# ============================================================
# Plotting
# ============================================================

def compute_ef_thresholds_for_levels(all_results: Dict[str, dict], num_bins: int) -> Dict[str, List[int]]:
    """Compute EF threshold for each difficulty level per dataset.
    
    For each dataset, determine the EF value for each level based on dataset characteristics
    and the actual EF value used in the experiment.
    """
    # Group by dataset
    by_dataset: Dict[str, List[dict]] = defaultdict(list)
    for key, result in all_results.items():
        by_dataset[result["dataset"]].append(result)
    
    ef_thresholds: Dict[str, List[int]] = {}
    
    # Dataset-specific EF threshold patterns (based on reference image)
    dataset_patterns = {
        "sift": [20, 30, 40, 80, 300],
        "fashion-mnist": [20, 300],
        "fashion_mnist": [20, 300],
        "mnist": [20, 300],
        "deep": [20, 40, 80, 170, 300],
        "deep1b": [20, 40, 80, 170, 300],
        "glove": [20, 60, 230, 300],
        "glove1m": [30, 80, 300],
        "glove2m": [30, 80, 300],
        "gist": [20, 40, 80, 170, 300],
        "wikipedia": [30, 60, 140, 300],
    }
    
    for dataset, results_list in by_dataset.items():
        # Sort by EF value
        results_list = sorted(results_list, key=lambda r: r["ef"])
        base_ef = results_list[0]["ef"]
        dataset_lower = dataset.lower()
        
        # Check if we have a predefined pattern
        if dataset_lower in dataset_patterns:
            pattern = dataset_patterns[dataset_lower]
            # Use the pattern, but ensure it has num_bins elements
            if len(pattern) >= num_bins:
                thresholds = pattern[:num_bins] + [300]
            else:
                # Extend pattern if needed
                thresholds = pattern + [300] * (num_bins - len(pattern) + 1)
        else:
            # Generate thresholds based on base_ef and dataset characteristics
            # For datasets with low base_ef (easy datasets)
            if base_ef <= 200:
                if "mnist" in dataset_lower or "fashion" in dataset_lower:
                    thresholds = [20, 300] + [300] * (num_bins - 2) + [300]
                else:
                    thresholds = [20, 40, 80, min(170, base_ef), base_ef] + [300]
                    thresholds = thresholds[:num_bins] + [300]
            # For datasets with medium base_ef
            elif base_ef <= 400:
                thresholds = [20, 40, 80, max(140, base_ef - 100), base_ef] + [300]
                thresholds = thresholds[:num_bins] + [300]
            # For datasets with high base_ef
            else:
                thresholds = [20, 40, 80, 170, 300] + [300]
                thresholds = thresholds[:num_bins] + [300]
        
        # Ensure we have exactly num_bins + 1 thresholds
        if len(thresholds) < num_bins + 1:
            # Pad with the last value
            last_val = thresholds[-1] if thresholds else 300
            thresholds.extend([last_val] * (num_bins + 1 - len(thresholds)))
        elif len(thresholds) > num_bins + 1:
            thresholds = thresholds[:num_bins + 1]
        
        # Ensure thresholds are non-decreasing
        for i in range(1, len(thresholds)):
            if thresholds[i] < thresholds[i-1]:
                thresholds[i] = thresholds[i-1]
        
        ef_thresholds[dataset] = thresholds
    
    return ef_thresholds


def plot_difficulty(all_results: Dict[str, dict], out_path: str, num_bins: int):
    # Group results by dataset (one bar per dataset)
    by_dataset: Dict[str, dict] = {}
    for key, result in all_results.items():
        dataset = result["dataset"]
        # Use the result with the largest total, or first one if equal
        if dataset not in by_dataset or result["total"] > by_dataset[dataset]["total"]:
            by_dataset[dataset] = result
    
    if not by_dataset:
        print("[WARN] No results to plot.")
        return
    
    # Sort datasets
    datasets = sorted(by_dataset.keys())
    n = len(datasets)
    
    # Compute EF thresholds for each level
    ef_thresholds = compute_ef_thresholds_for_levels(all_results, num_bins)
    
    # Prepare matrix: levels 0..num_bins (num_bins = recall<1)
    max_level = num_bins  # 0..num_bins, length num_bins+1
    matrix = np.zeros((max_level + 1, n), dtype=float)
    level_efs = np.zeros((max_level + 1, n), dtype=int)  # EF threshold for recall<1.0 level
    bin_ranges_per_dataset = {}  # Store iter_count ranges for each dataset
    
    for j, dataset in enumerate(datasets):
        r = by_dataset[dataset]
        total = r["total"]
        thresholds = ef_thresholds.get(dataset, [20, 40, 80, 170, 300, 300])
        bin_ranges_per_dataset[dataset] = r["bin_ranges"]  # Store iter_count ranges
        
        for lvl in range(max_level + 1):
            cnt = r["levels"].get(lvl, 0)
            matrix[lvl, j] = cnt / total * 100.0 if total > 0 else 0.0
            # Only store EF for recall<1.0 level (last level)
            if lvl == num_bins:
                level_efs[lvl, j] = r["ef"]  # Use the EF value from the result
    
    # --- Professional figure style for paper ---
    plt.figure(figsize=(10, 5.5))
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 100,
    })
    
    # Elegant academic color palette - classic scientific journal style
    # Option 1: Sophisticated blue gradient (current - elegant and professional)
    COLORS = [
        "#E8F4F8",  # Level 0: Very light blue (easiest)
        "#B3D9E6",  # Level 1: Light blue
        "#7FC4D3",  # Level 2: Medium blue
        "#4FA3C0",  # Level 3: Medium-dark blue
        "#2E7D9A",  # Level 4: Dark blue (harder)
        "#1A5F7A",  # Recall < 1: Darker blue-gray (harmonious with gradient, distinct)
    ]
    
    # Option 2: Classic grayscale (most conservative, uncomment to use)
    # COLORS = [
    #     "#F5F5F5",  # Level 0: Very light gray
    #     "#D9D9D9",  # Level 1: Light gray
    #     "#BDBDBD",  # Level 2: Medium-light gray
    #     "#969696",  # Level 3: Medium gray
    #     "#737373",  # Level 4: Dark gray
    #     "#525252",  # Recall < 1: Very dark gray
    # ]
    
    # Option 3: Soft pastel academic (Nature/Science style, uncomment to use)
    # COLORS = [
    #     "#DEEBF7",  # Level 0: Very light blue
    #     "#C6DBEF",  # Level 1: Light blue
    #     "#9ECAE1",  # Level 2: Medium-light blue
    #     "#6BAED6",  # Level 3: Medium blue
    #     "#4292C6",  # Level 4: Dark blue
    #     "#6B6B6B",  # Recall < 1: Medium gray
    # ]
    while len(COLORS) < max_level + 1:
        COLORS.append("#CCCCCC")
    
    x = np.arange(n)
    bar_width = 0.7
    bottoms = np.zeros(n)
    
    # Draw stacked bars
    bars = []
    for lvl in range(max_level + 1):
        bar = plt.bar(
            x,
            matrix[lvl],
            bottom=bottoms,
            width=bar_width,
            color=COLORS[lvl],
            edgecolor="white",
            linewidth=1.2,
            label=f"Level {lvl}" if lvl < num_bins else "Recall < 1.0",
        )
        bars.append(bar)
        bottoms += matrix[lvl]
    
    # Add labels inside each segment (only once per segment)
    # Use a set to track which labels have been drawn to avoid duplicates
    drawn_labels = set()
    for j, dataset in enumerate(datasets):
        current_bottom = 0
        bin_ranges = bin_ranges_per_dataset.get(dataset, [])
        
        for lvl in range(max_level + 1):
            height = matrix[lvl, j]
            if height > 3.0:  # Only label if segment is large enough to be readable
                text_y = current_bottom + height / 2
                
                if lvl < num_bins:
                    # For Level 0-4: show only the range [lo-hi]
                    if lvl < len(bin_ranges):
                        lo, hi = bin_ranges[lvl]
                        label_text = f"[{lo}-{hi}]"
                    else:
                        label_text = f"L{lvl}"
                    
                    # Create unique key for this label position to avoid duplicates
                    label_key = (j, lvl, label_text)
                    if label_key not in drawn_labels:
                        drawn_labels.add(label_key)
                        
                        # Determine text color based on background brightness
                        # Light backgrounds (Level 0-2) use dark text, darker backgrounds use white text
                        if lvl <= 2:
                            text_color = "#2C3E50"  # Dark blue-gray for readability
                        else:
                            text_color = "white"
                        
                        # Add subtle background box for light segments to ensure readability
                        bbox_props = None
                        if lvl <= 1:
                            # Very light backgrounds need subtle box for contrast
                            bbox_props = dict(
                                boxstyle="round,pad=0.3",
                                facecolor="white",
                                alpha=0.6,
                                edgecolor="none"
                            )
                        
                        # Draw label only once
                        plt.text(
                            x[j],
                            text_y,
                            label_text,
                            ha="center",
                            va="center",
                            fontsize=9,
                            fontweight="bold",
                            color=text_color,
                            bbox=bbox_props,
                        )
                # For recall<1.0 level: no label (EF will be shown below x-axis)
                
            current_bottom += height
    
    # Format dataset labels (capitalize properly)
    def format_dataset_name(name):
        name = name.replace("_", " ").replace("-", " ")
        # Handle special cases
        if "mnist" in name.lower():
            parts = name.split()
            if len(parts) > 1:
                return f"{parts[0].title()} {parts[1].upper()}"
        return name.title()
    
    labels = [format_dataset_name(ds) for ds in datasets]
    
    # Add EF values below dataset names
    labels_with_ef = []
    for j, (label, dataset) in enumerate(zip(labels, datasets)):
        ef_val = level_efs[num_bins, j]  # Get EF from recall<1.0 level
        labels_with_ef.append(f"{label}\nEF={ef_val}")
    
    plt.ylabel("Percentage of Queries (%)", fontweight="bold", fontsize=13)
    plt.title("Query Difficulty Distribution by Iteration Count Ranges", 
              fontweight="bold", fontsize=15, pad=15)
    plt.xticks(x, labels_with_ef, rotation=0, fontsize=10, fontweight="normal")
    plt.yticks(fontsize=11)
    plt.ylim(0, 105)
    plt.xlim(-0.5, n - 0.5)
    
    # Add grid for better readability
    plt.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
    plt.gca().set_axisbelow(True)
    
    # Legend
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10,
        title="Difficulty Levels",
        title_fontsize=11,
    )
    
    plt.tight_layout()
    
    # Save as both PNG and PDF
    # Determine base path and extension
    if out_path.endswith('.png'):
        base_path = out_path[:-4]
        png_path = out_path
        pdf_path = base_path + '.pdf'
    elif out_path.endswith('.pdf'):
        base_path = out_path[:-4]
        pdf_path = out_path
        png_path = base_path + '.png'
    else:
        # No extension, save both
        base_path = out_path
        png_path = base_path + '.png'
        pdf_path = base_path + '.pdf'
    
    # Save PNG (high resolution for preview)
    plt.savefig(png_path, dpi=350, bbox_inches="tight", facecolor="white")
    print(f"[OK] PNG figure saved to: {png_path}")
    
    # Save PDF (vector format for paper insertion)
    plt.savefig(pdf_path, format='pdf', bbox_inches="tight", facecolor="white")
    print(f"[OK] PDF figure saved to: {pdf_path}")


def plot_individual_datasets_cumulative(all_results: Dict[str, dict], out_dir: str, num_bins: int):
    """Generate individual cumulative percentage plots for each dataset.
    
    Each dataset gets its own figure showing cumulative percentage distribution.
    """
    # Group results by dataset
    by_dataset: Dict[str, dict] = {}
    for key, result in all_results.items():
        dataset = result["dataset"]
        if dataset not in by_dataset or result["total"] > by_dataset[dataset]["total"]:
            by_dataset[dataset] = result
    
    if not by_dataset:
        print("[WARN] No results to plot.")
        return
    
    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Color palette (same as main plot)
    COLORS = [
        "#E8F4F8",  # Level 0: Very light blue
        "#B3D9E6",  # Level 1: Light blue
        "#7FC4D3",  # Level 2: Medium blue
        "#4FA3C0",  # Level 3: Medium-dark blue
        "#2E7D9A",  # Level 4: Dark blue
        "#1A5F7A",  # Recall < 1: Darker blue-gray
    ]
    
    # Format dataset name
    def format_dataset_name(name):
        name = name.replace("_", " ").replace("-", " ")
        if "mnist" in name.lower():
            parts = name.split()
            if len(parts) > 1:
                return f"{parts[0].title()} {parts[1].upper()}"
        return name.title()
    
    for dataset, r in by_dataset.items():
        total = r["total"]
        bin_ranges = r["bin_ranges"]
        ef_val = r["ef"]
        max_level = num_bins
        
        # Calculate individual percentages
        individual_pcts = []
        colors_list = []
        
        for lvl in range(max_level + 1):
            cnt = r["levels"].get(lvl, 0)
            pct = (cnt / total * 100.0) if total > 0 else 0.0
            individual_pcts.append(pct)
            colors_list.append(COLORS[lvl] if lvl < len(COLORS) else "#CCCCCC")
        
        # Create figure - single y-axis only
        plt.rcParams.update({
            "font.size": 11,
            "font.family": "serif",
            "axes.labelsize": 12,
            "axes.labelweight": "normal",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        })
        
        # Create single figure with bars only (no dual y-axis)
        fig, ax = plt.subplots(figsize=(9, 7))
        
        # Prepare x-axis labels and positions
        x_labels = []
        x_positions = []
        for i in range(max_level + 1):
            if i < num_bins:
                lo, hi = bin_ranges[i] if i < len(bin_ranges) else (0, 0)
                x_labels.append(f"Level {i}\n[{lo}-{hi}]")
            else:
                x_labels.append("Recall\n< 1.0")
            x_positions.append(i)
        
        # Draw line plot with markers
        dataset_display = format_dataset_name(dataset)
        max_individual = max(individual_pcts) if max(individual_pcts) > 0 else 50
        
        # Plot line with markers
        line = ax.plot(x_positions, individual_pcts, 
                      marker='o', markersize=10, linewidth=2.5,
                      color='#2E7D9A', markerfacecolor='#2E7D9A',
                      markeredgecolor='white', markeredgewidth=2.0,
                      zorder=3)
        
        # Formatting axis
        # Increase y-axis range to accommodate labels
        ax.set_ylim(0, max(max_individual * 1.3, 60))
        ax.set_xlim(-0.5, max_level + 0.5)
        
        # Add percentage labels on data points
        for i, (x_pos, pct) in enumerate(zip(x_positions, individual_pcts)):
            if pct > 0.5:  # Only label if value is significant
                label_y = pct + max_individual * 0.05  # Spacing above point
                text_color = '#2C3E50'
                
                ax.text(x_pos, label_y,
                       f'{pct:.1f}%',
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold', zorder=4,
                       color=text_color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                alpha=0.9, edgecolor='#2E7D9A', linewidth=1.0))
        
        # Complete axis formatting
        ax.set_ylabel("Percentage of Queries (%)", fontweight="normal", fontsize=12)
        ax.set_xlabel("Difficulty Level", fontweight="normal", fontsize=12)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=10)
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        
        # Title
        ax.set_title(f"{dataset_display} - Difficulty Distribution (EF={ef_val})", 
                    fontweight="bold", fontsize=13, pad=15)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.15)
        
        # Save as PNG and PDF
        safe_dataset_name = dataset.replace(" ", "_").replace("-", "_")
        png_path = out_path / f"{safe_dataset_name}_difficulty.png"
        pdf_path = out_path / f"{safe_dataset_name}_difficulty.pdf"
        
        plt.savefig(png_path, dpi=350, bbox_inches="tight", facecolor="white")
        plt.savefig(pdf_path, format='pdf', bbox_inches="tight", facecolor="white")
        plt.close()
        
        print(f"[OK] Individual plot saved: {png_path} and {pdf_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze search difficulty by iter_count bins (per-dataset quantiles) and recall."
    )
    parser.add_argument("results_dir", type=str, help="Root directory of results (Offset_results)")
    parser.add_argument("--num-bins", type=int, default=5, help="Number of iter_count bins for recall=1")
    parser.add_argument("--plot", type=str, default="difficulty.png", help="Output image path")
    parser.add_argument("--txt", type=str, default="analysis.txt", help="Output text report path")
    parser.add_argument("--md", type=str, default="analysis.md", help="Output markdown report path")
    parser.add_argument("--individual-dir", type=str, default="individual_plots", 
                        help="Directory to save individual dataset plots")
    args = parser.parse_args()

    root = Path(args.results_dir)
    if not root.exists():
        print(f"[ERROR] results_dir not found: {root}")
        return 1

    num_bins = args.num_bins
    all_results: Dict[str, dict] = {}

    # For each dataset, compute its own iter_count quantile bins and analysis
    for dataset_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if not dataset_dir.is_dir():
            continue

        print(f"[INFO] Processing dataset: {dataset_dir.name}")
        iters = collect_recall1_iters_for_dataset(dataset_dir)
        if not iters:
            print(f"[WARN] No recall=1 iters found in dataset {dataset_dir.name}, skipping.")
            continue

        iter_bins = make_quantile_bins(iters, num_bins)
        print(f"       iter_bins (quantiles): {iter_bins}")

        dataset_results = analyze_dataset(dataset_dir, iter_bins)
        all_results.update(dataset_results)

    if not all_results:
        print("[WARN] No results aggregated, nothing to output.")
        return 0

    write_reports(all_results, args.txt, args.md)
    plot_difficulty(all_results, args.plot, num_bins)
    plot_individual_datasets_cumulative(all_results, args.individual_dir, num_bins)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

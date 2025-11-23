#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys

# 检查必要的依赖
try:
    import numpy as np
except ImportError:
    print("Error: numpy is not installed. Please install it with: pip install numpy")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is not installed. Please install it with: pip install pandas")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is not installed. Please install it with: pip install matplotlib")
    sys.exit(1)

from pathlib import Path


# 计算 JSON 文件中的平均值（与原脚本一致）
def calculate_avg(json_file, percent=0.95):
    """从JSON文件中计算平均值，默认使用最后95%的数据"""
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)

        entries = data["entries"]
        if len(entries) == 0:
            return None, None, None

        entries_to_consider = max(1, int(len(entries) * percent))
        filtered_entries = entries[-entries_to_consider:]

        avg_hnsw = np.mean([entry["hnsw"] for entry in filtered_entries])
        avg_hnswio = np.mean([entry["hnswio"] for entry in filtered_entries])
        avg_io = np.mean([entry["io"] for entry in filtered_entries])

        return avg_hnsw, avg_hnswio, avg_io
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return None, None, None


# 从目录结构中收集 multi-level 的结果
def collect_results_from_dir(base_dir):
    """
    从目录结构中收集 multi-level 实验结果
    目录结构: {base_dir}/{dataset}/{threads}_{ef}_{io_depths}/{query_cnt}/{repeat_id}/HNSWIO.json
    """
    results = []

    if not os.path.exists(base_dir):
        print(f"Warning: Directory {base_dir} does not exist")
        return results

    # 遍历数据集文件夹
    for dataset_name in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        # 遍历配置文件夹 (如 1_300_256)
        for config_dir in os.listdir(dataset_path):
            config_path = os.path.join(dataset_path, config_dir)
            if not os.path.isdir(config_path):
                continue

            # 遍历 query_cnt 文件夹 (如 10000)
            for query_cnt_dir in os.listdir(config_path):
                query_cnt_path = os.path.join(config_path, query_cnt_dir)
                if not os.path.isdir(query_cnt_path):
                    continue

                # 遍历 repeat_id 文件夹 (如 1, 2, 3)
                for repeat_id_dir in os.listdir(query_cnt_path):
                    repeat_path = os.path.join(query_cnt_path, repeat_id_dir)
                    if not os.path.isdir(repeat_path):
                        continue

                    json_file = os.path.join(repeat_path, "HNSWIO.json")
                    if os.path.exists(json_file):
                        avg_hnsw, avg_hnswio, avg_io = calculate_avg(json_file)
                        if avg_hnsw is not None and avg_hnswio > 0:
                            results.append(
                                {
                                    "dataset": dataset_name,
                                    "config": config_dir,
                                    "query_cnt": query_cnt_dir,
                                    "repeat_id": repeat_id_dir,
                                    "avg_hnsw": avg_hnsw,
                                    "avg_hnswio": avg_hnswio,
                                    "avg_io": avg_io,
                                }
                            )

    return results


# 聚合并计算 IO / HNSW 百分比
def aggregate_results(all_results):
    """按数据集聚合，计算 IO 和 HNSW 的百分比（纵坐标 0-100%）"""
    df = pd.DataFrame(all_results)

    if df.empty:
        print("Warning: No results collected")
        return pd.DataFrame()

    # 这里只关心数据集维度，跨 config / repeat 求平均
    df_agg = df.groupby(["dataset"], as_index=False).agg(
        {
            "avg_hnsw": "mean",
            "avg_hnswio": "mean",
            "avg_io": "mean",
        }
    )

    # 计算百分比
    df_agg["io_ratio"] = (df_agg["avg_io"] / df_agg["avg_hnswio"]) * 100.0
    df_agg["hnsw_ratio"] = (df_agg["avg_hnsw"] / df_agg["avg_hnswio"]) * 100.0

    # 为了保证纵轴 0-100%，可以做一次裁剪（防止浮点误差）
    df_agg["io_ratio"] = df_agg["io_ratio"].clip(lower=0, upper=100)
    df_agg["hnsw_ratio"] = df_agg["hnsw_ratio"].clip(lower=0, upper=100)

    return df_agg


# 绘制 Io-hold（HNSW vs IO）占比图
def plot_io_hold_ratio(df_agg, results_dir):
    """
    绘制 multi-level 实验的 Io-hold 占比图：
    - 纵坐标 0-100%
    - 堆叠柱状图展示 HNSW 和 IO 的占比
    - 只展示 multi-level（MultiLayer_Fs_results）的数据
    """
    if df_agg.empty:
        print("Warning: No data to plot")
        return

    # 按 IO 占比从大到小排序
    df_agg = df_agg.sort_values(by="io_ratio", ascending=False).reset_index(drop=True)

    # 白色背景
    plt.style.use("default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 按 io_ratio 排序后的顺序直接作为横坐标顺序
    datasets = df_agg["dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.6

    # 直接按当前顺序取比例数组，保证与排序一致
    io_ratios = df_agg["io_ratio"].to_numpy()
    hnsw_ratios = df_agg["hnsw_ratio"].to_numpy()

    # 颜色（更加偏论文风格：柔和蓝 + 中性灰）
    # 参考常见学术配色：IO 用蓝色，HNSW 用中性灰
    color_io = "#4F81BD"      # 柔和蓝：IO
    color_hnsw = "#C0C0C0"    # 中性浅灰：HNSW / Compute

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("white")

    # 堆叠柱状图
    ax.bar(
        x,
        io_ratios,
        width,
        label="IO Time Ratio",
        color=color_io,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.bar(
        x,
        hnsw_ratios,
        width,
        bottom=io_ratios,
        label="HNSW Time Ratio",
        color=color_hnsw,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.8,
    )

    # 坐标轴和标题（纵轴 0-100%）
    ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax.set_ylabel("Time Ratio (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Multi-Level Directory: IO vs HNSW Time Ratio",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    # 横坐标数据集名称加粗
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=11, fontweight="bold")
    # 为柱顶留一点空间，避免顶死在 100% 上
    ax.set_ylim(0, 105)
    # 增加 50% 参考线
    ax.axhline(
        y=50,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="50% Threshold",
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--", color="gray")

    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    # 添加数值标签（考虑顶部留白后微调位置）
    for i, (io_r, hnsw_r) in enumerate(zip(io_ratios, hnsw_ratios)):
        if io_r > 0:
            ax.text(
                i,
                io_r / 2,
                f"{io_r:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
            )
        if hnsw_r > 0:
            ax.text(
                i,
                io_r + hnsw_r / 2,
                f"{hnsw_r:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
            )

    plt.tight_layout()

    # 输出目录与文件
    os.makedirs(results_dir, exist_ok=True)
    png_file = os.path.join(results_dir, "multi_level_io_hold_ratio.png")
    pdf_file = os.path.join(results_dir, "multi_level_io_hold_ratio.pdf")
    plt.savefig(png_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_file, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Io-hold 占比图已保存到: {png_file} 和 {pdf_file}")


def main():
    # 只分析 multi-level 的结果目录
    multi_base_dir = (
        "/home/zqf/Hulu-Retriever/MakeFsWorkSpace/results/MultiLayer_Fs_results"
    )

    print("正在收集 multi-level 实验结果...")
    all_results = collect_results_from_dir(multi_base_dir)
    print(f"  找到 {len(all_results)} 个结果文件")

    if not all_results:
        print("错误: 未找到任何 multi-level 实验结果文件")
        return

    print("\n正在聚合结果并计算占比...")
    df_agg = aggregate_results(all_results)

    if df_agg.empty:
        print("错误: 聚合后结果为空")
        return

    # 输出目录
    output_dir = "/home/zqf/Hulu-Retriever/MakeFsWorkSpace/results/io_hold_analysis"

    # 保存 CSV（可选）
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "multi_level_io_hold_ratio.csv")
    df_agg.to_csv(csv_file, index=False)
    print(f"CSV 结果已保存到: {csv_file}")

    # 绘图
    plot_io_hold_ratio(df_agg, output_dir)

    # 控制台打印简要汇总
    print("\n=== Multi-Level IO vs HNSW Time Ratio (0-100%) ===")
    df_show = df_agg[["dataset", "io_ratio", "hnsw_ratio"]].copy()
    df_show["io_ratio"] = df_show["io_ratio"].round(2)
    df_show["hnsw_ratio"] = df_show["hnsw_ratio"].round(2)
    print(df_show.to_string(index=False))
    print(f"\n所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
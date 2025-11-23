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


# 与 analyze.py 一致的平均值计算
def calculate_avg(json_file, percent=0.95):
    """从JSON文件中计算平均值，默认使用最后95%的数据"""
    try:
        with open(json_file, "r") as file:
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


def collect_results_from_dir(base_dir, experiment_name):
    """
    从目录结构中收集结果
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
                        if avg_hnsw is not None:
                            results.append(
                                {
                                    "experiment": experiment_name,
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


def aggregate_results(all_results):
    """对每个实验方法 / 数据集聚合，只保留 IO 信息"""
    df = pd.DataFrame(all_results)

    if df.empty:
        print("Warning: No results collected")
        return pd.DataFrame()

    df_agg = df.groupby(["experiment", "dataset"], as_index=False).agg(
        {
            "avg_io": ["mean", "std"],
        }
    )

    # 展平列名
    df_agg.columns = [
        "experiment",
        "dataset",
        "avg_io",
        "avg_io_std",
    ]

    return df_agg


def plot_io_only(df_agg, results_dir):
    """
    只绘制 IO 部分的对比（Offset / Single / Multi），适合论文插图：
    - 每个数据集一组，组内三根柱（Raw, Single, Multi）
    - 坐标轴 / 字体加粗放大
    """
    if df_agg.empty:
        print("Warning: No data to plot")
        return

    # 论文友好风格设置：白底、向量友好
    plt.style.use("default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 统一实验顺序
    experiments = ["offset", "single", "multi"]
    exp_labels = {
        "offset": "Raw Disk Offset",
        "single": "Single-Layer Directory",
        "multi": "Multi-Layer Directory",
    }

    # 数据集顺序：与 analyze.py / comparison_results.png 保持一致 —— 按名称排序
    datasets = sorted(df_agg["dataset"].unique())

    x = np.arange(len(datasets))
    width = 0.22  # 三根柱子稍窄一点，避免拥挤

    # 颜色（论文常用：蓝 / 橙 / 绿）
    colors = {
        "offset": "#4472C4",  # 蓝
        "single": "#ED7D31",  # 橙
        "multi": "#70AD47",   # 绿
    }

    fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")
    ax.set_facecolor("white")

    max_io = 0.0

    # 逐实验画图
    for i, exp in enumerate(experiments):
        io_values = []
        io_err = []

        for dataset in datasets:
            row = df_agg[(df_agg["dataset"] == dataset) & (df_agg["experiment"] == exp)]
            if not row.empty:
                v = row["avg_io"].values[0]
                e = row["avg_io_std"].values[0]
            else:
                v = 0.0
                e = 0.0
            io_values.append(v)
            io_err.append(e)
            max_io = max(max_io, v)

        x_pos = x + (i - 1) * width  # 中心对称排开三根柱子
        ax.bar(
            x_pos,
            io_values,
            width,
            label=exp_labels[exp],
            color=colors[exp],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
            yerr=io_err,
            capsize=4,
            error_kw={"elinewidth": 1.0, "alpha": 0.8},
        )

    # 坐标轴与字体优化
    ax.set_xlabel("Dataset", fontsize=14, fontweight="bold")
    # 更明确地标出是平均 IO 时间
    ax.set_ylabel("Average IO Time (μs)", fontsize=14, fontweight="bold")
    # 标题更贴近论文语境，强调是单索引场景下的 IO 时延对比
    ax.set_title(
        "IO Latency Comparison Across Storage Schemes",
        fontsize=15,
        fontweight="bold",
        pad=32,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        datasets,
        rotation=45,
        ha="right",
        fontsize=12,
        fontweight="bold",
    )

    # y 轴范围稍微放大一点，给顶部留白
    ax.set_ylim(0, max_io * 1.15 if max_io > 0 else 1.0)

    # y 轴刻度字体
    ax.tick_params(axis="y", labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    # 网格与边框
    ax.grid(axis="y", alpha=0.3, linestyle="--", color="gray")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # 图例
    # 将图例上移到图内顶部中央，避免遮挡首列
    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 1.08),
        borderaxespad=0.2,
        fontsize=11,
        framealpha=0.95,
        ncol=1,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    # 拉大图表与标题/图例之间的距离，避免互相遮挡
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)

    os.makedirs(results_dir, exist_ok=True)
    png_file = os.path.join(results_dir, "single_index_io_only.png")
    pdf_file = os.path.join(results_dir, "single_index_io_only.pdf")
    plt.savefig(png_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_file, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"单索引 IO 对比图已保存到: {png_file} 和 {pdf_file}")


def main():
    """主函数：从 analyze.py 生成的聚合 CSV 中读取结果，只绘 IO 部分的对比图"""

    # 直接复用 analyze.py 已经聚合好的结果，确保与 comparison_results.png 完全对齐
    src_csv = "/home/zqf/Hulu-Retriever/MakeFsWorkSpace/results/comparison_analysis/comparison_results.csv"
    if not os.path.exists(src_csv):
        print(f"错误: 找不到聚合结果文件: {src_csv}")
        print("请先运行 analyze.py 生成 comparison_results.csv 再调用本脚本。")
        return

    df_agg = pd.read_csv(src_csv)

    # 只保留绘图需要的列
    expected_cols = {"experiment", "dataset", "avg_io", "avg_io_std"}
    missing = expected_cols - set(df_agg.columns)
    if missing:
        print(f"错误: CSV 中缺少列: {missing}")
        return

    output_dir = "/home/zqf/Hulu-Retriever/MakeFsWorkSpace/results/single_index_io_only"
    os.makedirs(output_dir, exist_ok=True)

    # 为了便于核对，这里再导出一份只含 IO 的 CSV
    io_csv = os.path.join(output_dir, "single_index_io_only.csv")
    df_agg[list(expected_cols)].to_csv(io_csv, index=False)
    print(f"IO 结果已保存到: {io_csv}")

    # 绘制仅 IO 的对比图
    plot_io_only(df_agg, output_dir)

    print("\n=== IO Time Summary (μs, from comparison_results.csv) ===")
    print(df_agg[["experiment", "dataset", "avg_io", "avg_io_std"]].to_string(index=False))
    print(f"\n所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
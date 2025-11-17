import os
import sys
import json
import argparse
from math import ceil

# 依赖检查（与现有 analyze.py 一致风格）
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
    from matplotlib.patches import Patch
except ImportError:
    print("Error: matplotlib is not installed. Please install it with: pip install matplotlib")
    sys.exit(1)

# 常量路径（与现有脚本保持一致）
RESULT_DIRS = {
    "offset": "/home/zqf/Hulu-Retriever/MakeFsWorkSpace/end_results/raw_results/Offset_results",
    "single": "/home/zqf/Hulu-Retriever/MakeFsWorkSpace/results/Fs_results",
    "multi": "/home/zqf/Hulu-Retriever/MakeFsWorkSpace/results/MultiLayer_Fs_results",
}

# 颜色与样式（统一配置便于保持风格一致）
COLORS_IO = {
    'offset': '#4472C4',  # 深蓝 Raw Disk Offset
    'single': '#ED7D31',  # 橙色 Single-Layer Dir
    'multi':  '#70AD47',  # 绿色 Multi-Layer Dir
}
COLOR_HNSW = '#D9D9D9'      # 浅灰 HNSW 部分
FIG_BG_COLOR = 'white'
BAR_ALPHA = 0.85
EDGE_COLOR = 'black'
EDGE_WIDTH = 0.8
LABEL_FONT_SIZE = 9
TITLE_FONT_SIZE = 13
AXIS_FONT_SIZE = 12
GRID_STYLE = {'axis': 'y', 'alpha': 0.3, 'linestyle': '--', 'color': 'gray'}


def calculate_avg(json_file, percent=0.95):
    """读取 HNSWIO.json 并计算最后 percent 比例的数据平均值"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        entries = data.get('entries', [])
        if not entries:
            return None, None, None
        n = max(1, int(len(entries) * percent))
        filtered = entries[-n:]
        avg_hnsw = np.mean([e['hnsw'] for e in filtered])
        avg_hnswio = np.mean([e['hnswio'] for e in filtered])
        avg_io = np.mean([e['io'] for e in filtered])
        return avg_hnsw, avg_hnswio, avg_io
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return None, None, None


def parse_config_dir(config_dir):
    """从形如 '{threads}_{ef}_{io_depth}' 的目录名解析 threads, ef, io_depth"""
    try:
        parts = config_dir.split('_')
        threads = int(parts[0]) if len(parts) > 0 else None
        ef = int(parts[1]) if len(parts) > 1 else None
        io_depth = int(parts[2]) if len(parts) > 2 else None
        return threads, ef, io_depth
    except Exception:
        return None, None, None


def collect_results_from_dir(base_dir, experiment_name, percent=0.95):
    """收集某种实验方案（offset/single/multi）的全量结果，保留线程维度
    目录结构: {base_dir}/{dataset}/{threads}_{ef}_{io_depth}/{query_cnt}/{repeat_id}/HNSWIO.json
    """
    results = []
    if not os.path.exists(base_dir):
        print(f"Warning: Directory {base_dir} does not exist")
        return results

    for dataset_name in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        for config_dir in os.listdir(dataset_path):
            config_path = os.path.join(dataset_path, config_dir)
            if not os.path.isdir(config_path):
                continue
            threads, ef, io_depth = parse_config_dir(config_dir)

            for query_cnt_dir in os.listdir(config_path):
                query_cnt_path = os.path.join(config_path, query_cnt_dir)
                if not os.path.isdir(query_cnt_path):
                    continue

                for repeat_id_dir in os.listdir(query_cnt_path):
                    repeat_path = os.path.join(query_cnt_path, repeat_id_dir)
                    if not os.path.isdir(repeat_path):
                        continue

                    json_file = os.path.join(repeat_path, "HNSWIO.json")
                    if os.path.exists(json_file):
                        avg_hnsw, avg_hnswio, avg_io = calculate_avg(json_file, percent=percent)
                        if avg_hnsw is None:
                            continue
                        results.append({
                            'experiment': experiment_name,
                            'dataset': dataset_name,
                            'threads': threads,
                            'ef': ef,
                            'io_depth': io_depth,
                            'query_cnt': query_cnt_dir,
                            'repeat_id': repeat_id_dir,
                            'avg_hnsw': avg_hnsw,
                            'avg_hnswio': avg_hnswio,
                            'avg_io': avg_io,
                        })
    return results


def aggregate_results(all_results):
    """按 experiment, dataset, threads 聚合成均值（跨 repeat_id / query_cnt）"""
    df = pd.DataFrame(all_results)
    if df.empty:
        return pd.DataFrame()

    # 丢弃 threads 缺失的记录
    df = df.dropna(subset=['threads'])

    # 聚合
    df_agg = df.groupby(['experiment', 'dataset', 'threads'], as_index=False).agg({
        'avg_hnsw': 'mean',
        'avg_hnswio': 'mean',
        'avg_io': 'mean',
    })

    # 计算占比
    df_agg['io_ratio'] = (df_agg['avg_io'] / df_agg['avg_hnswio'].replace(0, np.nan)) * 100.0
    df_agg['hnsw_ratio'] = (df_agg['avg_hnsw'] / df_agg['avg_hnswio'].replace(0, np.nan)) * 100.0
    df_agg = df_agg.fillna({'io_ratio': 0.0, 'hnsw_ratio': 0.0})
    return df_agg


# 绘图：1) 单一实验方案在不同线程下、不同数据集的对比

def plot_experiment_by_threads(df_agg, experiment, output_dir):
    sub = df_agg[df_agg['experiment'] == experiment].copy()
    if sub.empty:
        print(f"Warning: No data for experiment '{experiment}'")
        return []

    datasets = sorted(sub['dataset'].unique())
    n = len(datasets)
    cols = min(3, max(1, n))
    rows = ceil(n / cols)

    plt.style.use('default')
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols + 1, 4*rows + 1), squeeze=False, facecolor='white')

    saved_files = []
    for idx, dataset in enumerate(datasets):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ddf = sub[sub['dataset'] == dataset].sort_values('threads')
        x = np.arange(len(ddf))
        width = 0.6

        io_vals = ddf['avg_io'].values
        hnsw_vals = ddf['avg_hnsw'].values
        total_vals = ddf['avg_hnswio'].values
        threads_list = ddf['threads'].astype(int).astype(str).values

        ax.bar(x, io_vals, width, color=COLORS_IO.get(experiment, '#888'), alpha=0.85, edgecolor='black', linewidth=0.8, label=f'{experiment} IO')
        ax.bar(x, hnsw_vals, width, bottom=io_vals, color=COLOR_HNSW, alpha=0.85, edgecolor='black', linewidth=0.8, label='HNSW')

        # 标签与样式
        ax.set_title(dataset, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(threads_list, fontsize=9)
        ax.set_xlabel('Threads', fontsize=10)
        ax.set_ylabel('Time (μs)', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')

        # 顶部总时间标签
        max_total = max(total_vals) if len(total_vals) else 0
        offset = max_total * 0.03 if max_total else 1
        for i, tval in enumerate(total_vals):
            ax.text(x[i], tval + offset, f"{tval:.0f}μs", ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 删除空子图
    for j in range(n, rows*cols):
        r, c = divmod(j, cols)
        fig.delaxes(axes[r][c])

    # 统一图例
    legend_elements = [
        Patch(facecolor=COLORS_IO.get(experiment, '#888'), edgecolor='black', alpha=0.85, label=f'{experiment} IO'),
        Patch(facecolor=COLOR_HNSW, edgecolor='black', alpha=0.85, label='HNSW'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, framealpha=0.95)
    fig.suptitle(f"{experiment.capitalize()} - Latency Breakdown by Threads", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = os.path.join(output_dir, f"{experiment}_by_threads.png")
    out_pdf = os.path.join(output_dir, f"{experiment}_by_threads.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    saved_files.extend([out_png, out_pdf])
    print(f"Saved: {out_png} and {out_pdf}")
    return saved_files


# 绘图：2) 在固定线程数下，不同实验方案的对比（跨数据集）

def plot_schemes_by_thread(df_agg, thread, output_dir):
    """在固定线程数下，跨数据集比较三种方案，每个方案为一个堆叠柱（IO+HNSW）。
    统一与最终整体对比图风格：
    - 顶部显示总延迟
    - IO 部分内部显示 IO 延迟
    - 统一 legend 描述
    """
    sub = df_agg[df_agg['threads'] == thread].copy()
    if sub.empty:
        print(f"Warning: No data for thread={thread}")
        return []

    datasets = sorted(sub['dataset'].unique())
    experiments = [e for e in ['offset', 'single', 'multi'] if e in sub['experiment'].unique()]

    plt.style.use('default')
    fig_width = max(14, 1.8 * len(datasets))
    fig, ax = plt.subplots(figsize=(fig_width, 8), facecolor=FIG_BG_COLOR)

    x = np.arange(len(datasets))
    width = 0.25  # 单个柱宽

    max_total_for_offset = 0
    for i, exp in enumerate(experiments):
        exp_df = sub[sub['experiment'] == exp]
        io_vals, hnsw_vals, total_vals = [], [], []
        for ds in datasets:
            row = exp_df[exp_df['dataset'] == ds]
            if not row.empty:
                io = row['avg_io'].values[0]
                hnsw = row['avg_hnsw'].values[0]
                total = row['avg_hnswio'].values[0]
            else:
                io = hnsw = total = 0
            io_vals.append(io)
            hnsw_vals.append(hnsw)
            total_vals.append(total)
        xpos = x + (i - (len(experiments)-1)/2) * width
        bars_io = ax.bar(xpos, io_vals, width,
                         color=COLORS_IO.get(exp, '#888'), alpha=BAR_ALPHA,
                         edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
        ax.bar(xpos, hnsw_vals, width, bottom=io_vals,
               color=COLOR_HNSW, alpha=BAR_ALPHA,
               edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
        # IO 延迟标签置于 IO 柱内部（居中）
        for b, io_v in zip(bars_io, io_vals):
            if io_v > 0:
                ax.text(b.get_x() + b.get_width()/2, io_v * 0.5, f"{io_v:.0f}μs",
                        ha='center', va='center', fontsize=LABEL_FONT_SIZE-1,
                        fontweight='bold', color='white')
        # 顶部总延迟标签
        offset_pix = max(total_vals) * 0.02 if max(total_vals) else 1
        for j, (xp, tot) in enumerate(zip(xpos, total_vals)):
            if tot > 0:
                ax.text(xp, tot + offset_pix, f"{tot:.0f}μs", ha='center', va='bottom',
                        fontsize=LABEL_FONT_SIZE, fontweight='bold')
        max_total_for_offset = max(max_total_for_offset, max(total_vals))

    ax.set_xlabel('Dataset', fontsize=AXIS_FONT_SIZE, fontweight='bold')
    ax.set_ylabel('Time (μs)', fontsize=AXIS_FONT_SIZE, fontweight='bold')
    ax.set_title(f'Latency Breakdown by Schemes @ {int(thread)} Threads', fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=LABEL_FONT_SIZE+1)
    ax.grid(**GRID_STYLE)

    # 图例统一
    legend_elements = []
    if 'offset' in experiments:
        legend_elements.append(Patch(facecolor=COLORS_IO['offset'], edgecolor=EDGE_COLOR, alpha=BAR_ALPHA, label='Raw Disk Offset (IO)'))
    if 'single' in experiments:
        legend_elements.append(Patch(facecolor=COLORS_IO['single'], edgecolor=EDGE_COLOR, alpha=BAR_ALPHA, label='Single-Layer Directory Structure (IO)'))
    if 'multi' in experiments:
        legend_elements.append(Patch(facecolor=COLORS_IO['multi'], edgecolor=EDGE_COLOR, alpha=BAR_ALPHA, label='Multi-Layer Directory Structure (IO)'))
    legend_elements.append(Patch(facecolor=COLOR_HNSW, edgecolor=EDGE_COLOR, alpha=BAR_ALPHA, label='HNSW'))
    ax.legend(handles=legend_elements, loc='upper left', fontsize=LABEL_FONT_SIZE, framealpha=0.95)

    plt.tight_layout()
    out_png = os.path.join(output_dir, f"schemes_by_thread_{int(thread)}.png")
    out_pdf = os.path.join(output_dir, f"schemes_by_thread_{int(thread)}.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor=FIG_BG_COLOR)
    fig.savefig(out_pdf, bbox_inches='tight', facecolor=FIG_BG_COLOR)
    plt.close(fig)
    print(f"Saved: {out_png} and {out_pdf}")
    return [out_png, out_pdf]


# 绘图：3) 在固定数据集下，不同实验方案随着线程变化的对比（满足用户第二个需求）
def plot_methods_by_dataset(df_agg, dataset, output_dir):
    """固定数据集：比较不同方案在多个线程下的延迟（IO+HNSW 堆叠），单图展示。"""
    sub = df_agg[df_agg['dataset'] == dataset].copy()
    if sub.empty:
        print(f"Warning: No data for dataset='{dataset}'")
        return []

    experiments = [e for e in ['offset', 'single', 'multi'] if e in sub['experiment'].unique()]
    if not experiments:
        print(f"Warning: No experiments found for dataset='{dataset}'")
        return []

    threads = sorted(sub['threads'].dropna().unique())
    if not threads:
        print(f"Warning: No thread data for dataset='{dataset}'")
        return []

    plt.style.use('default')
    fig_width = max(12, 1.4 * len(threads))
    fig, ax = plt.subplots(figsize=(fig_width, 8), facecolor=FIG_BG_COLOR)
    x = np.arange(len(threads))
    width = 0.25

    for i, exp in enumerate(experiments):
        exp_df = sub[sub['experiment'] == exp]
        io_vals, hnsw_vals, total_vals = [], [], []
        for th in threads:
            row = exp_df[exp_df['threads'] == th]
            if not row.empty:
                io = row['avg_io'].values[0]
                hnsw = row['avg_hnsw'].values[0]
                total = row['avg_hnswio'].values[0]
            else:
                io = hnsw = total = 0
            io_vals.append(io)
            hnsw_vals.append(hnsw)
            total_vals.append(total)
        xpos = x + (i - (len(experiments)-1)/2) * width
        bars_io = ax.bar(xpos, io_vals, width,
                         color=COLORS_IO.get(exp, '#888'), alpha=BAR_ALPHA,
                         edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
        ax.bar(xpos, hnsw_vals, width, bottom=io_vals,
               color=COLOR_HNSW, alpha=BAR_ALPHA,
               edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
        # IO 标签
        for b, io_v in zip(bars_io, io_vals):
            if io_v > 0:
                ax.text(b.get_x() + b.get_width()/2, io_v * 0.5, f"{io_v:.0f}μs",
                        ha='center', va='center', fontsize=LABEL_FONT_SIZE-1,
                        fontweight='bold', color='white')
        # 总时间标签
        offset_pix = max(total_vals) * 0.02 if max(total_vals) else 1
        for xp, tot in zip(xpos, total_vals):
            if tot > 0:
                ax.text(xp, tot + offset_pix, f"{tot:.0f}μs", ha='center', va='bottom',
                        fontsize=LABEL_FONT_SIZE, fontweight='bold')

    ax.set_xlabel('Threads', fontsize=AXIS_FONT_SIZE, fontweight='bold')
    ax.set_ylabel('Time (μs)', fontsize=AXIS_FONT_SIZE, fontweight='bold')
    ax.set_title(f'Latency Breakdown by Methods @ Dataset: {dataset}', fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in threads], fontsize=LABEL_FONT_SIZE+1)
    ax.grid(**GRID_STYLE)

    legend_elements = []
    if 'offset' in experiments:
        legend_elements.append(Patch(facecolor=COLORS_IO['offset'], edgecolor=EDGE_COLOR, alpha=BAR_ALPHA, label='Raw Disk Offset (IO)'))
    if 'single' in experiments:
        legend_elements.append(Patch(facecolor=COLORS_IO['single'], edgecolor=EDGE_COLOR, alpha=BAR_ALPHA, label='Single-Layer Directory Structure (IO)'))
    if 'multi' in experiments:
        legend_elements.append(Patch(facecolor=COLORS_IO['multi'], edgecolor=EDGE_COLOR, alpha=BAR_ALPHA, label='Multi-Layer Directory Structure (IO)'))
    legend_elements.append(Patch(facecolor=COLOR_HNSW, edgecolor=EDGE_COLOR, alpha=BAR_ALPHA, label='HNSW'))
    ax.legend(handles=legend_elements, loc='upper left', fontsize=LABEL_FONT_SIZE, framealpha=0.95)

    plt.tight_layout()
    out_png = os.path.join(output_dir, f"methods_by_dataset_{dataset}.png")
    out_pdf = os.path.join(output_dir, f"methods_by_dataset_{dataset}.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor=FIG_BG_COLOR)
    fig.savefig(out_pdf, bbox_inches='tight', facecolor=FIG_BG_COLOR)
    plt.close(fig)
    print(f"Saved: {out_png} and {out_pdf}")
    return [out_png, out_pdf]


def generate_markdown_report(df_agg, experiments, threads, output_dir):
    md_path = os.path.join(output_dir, 'thread_comparison_report.md')
    os.makedirs(output_dir, exist_ok=True)

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# 线程维度对比分析\n\n')
        f.write('本报告包含：\n')
        f.write('1) 指定实验方案在不同线程下、不同数据集的对比结果；\n')
        f.write('2) 指定线程数下，多种方案的对比结果；\n')
        f.write('3) 指定数据集下，多种方案随线程变化的对比结果。\n\n')

        # 汇总表：experiment, dataset, threads, 平均时间 + 占比
        df_rep = df_agg[['experiment', 'dataset', 'threads', 'avg_hnsw', 'avg_io', 'avg_hnswio', 'io_ratio', 'hnsw_ratio']].copy()
        df_rep = df_rep.sort_values(['experiment', 'dataset', 'threads'])
        f.write('## 汇总结果（均值，单位：μs）\n\n')
        f.write(df_rep.to_markdown(index=False))
        f.write('\n\n')

        # 插入图像列表
        f.write('## 图表\n\n')
        for exp in experiments:
            img = f"{exp}_by_threads.png"
            if os.path.exists(os.path.join(output_dir, img)):
                f.write(f"### {exp} 方案在不同线程下的结果\n\n")
                f.write(f"![{exp} by threads]({img})\n\n")

        for th in threads:
            img = f"schemes_by_thread_{int(th)}.png"
            if os.path.exists(os.path.join(output_dir, img)):
                f.write(f"### 不同方案在 {int(th)} 线程下的结果\n\n")
                f.write(f"![schemes by thread {int(th)}]({img})\n\n")

        datasets = sorted(df_agg['dataset'].unique())
        for ds in datasets:
            img = f"methods_by_dataset_{ds}.png"
            if os.path.exists(os.path.join(output_dir, img)):
                f.write(f"### 数据集 {ds} 下不同方案随线程变化结果\n\n")
                f.write(f"![methods by dataset {ds}]({img})\n\n")

    print(f"Markdown报告已保存到: {md_path}")
    return md_path


def main():
    parser = argparse.ArgumentParser(description='Compare results across threads and schemes.')
    parser.add_argument('--experiments', nargs='*', default=['offset', 'single', 'multi'],
                        help='实验方案列表：offset single multi（默认全选）')
    parser.add_argument('--datasets', nargs='*', default=None,
                        help='只分析指定数据集名称列表（默认全部数据集）')
    parser.add_argument('--threads', nargs='*', type=int, default=None,
                        help='需要对比的线程数列表（默认自动从结果中发现）')
    parser.add_argument('--percent', type=float, default=0.95,
                        help='计算平均值使用最后百分比的数据 (默认 0.95)')
    parser.add_argument('--output-dir', type=str, default='/home/zqf/Hulu-Retriever/MakeFsWorkSpace/results/thread_analysis',
                        help='输出目录')
    parser.add_argument('--no-ratio', action='store_true',
                        help='不生成占比（IO/HNSW比例）相关图表（用于快速运行）')
    parser.add_argument('--skip-exp-threads', action='store_true',
                        help='跳过按实验方案分开的多子图 (experiment_by_threads)，直接生成综合对比图。')

    args = parser.parse_args()

    # 收集结果
    all_results = []
    for exp in args.experiments:
        base = RESULT_DIRS.get(exp)
        if not base:
            print(f"Warning: unknown experiment '{exp}', skip")
            continue
        print(f"正在收集 {exp} 实验结果...")
        res = collect_results_from_dir(base, exp, percent=args.percent)
        print(f"  找到 {len(res)} 条记录")
        all_results.extend(res)

    if not all_results:
        print("错误: 未找到任何实验结果")
        return

    # 依据 --datasets 进行数据集过滤（在聚合前减少数据量）
    if args.datasets:
        ds_set = set(args.datasets)
        before_cnt = len(all_results)
        all_results = [r for r in all_results if r.get('dataset') in ds_set]
        after_cnt = len(all_results)
        if after_cnt == 0:
            print('错误: 过滤后的结果为空，请检查 --datasets 参数。')
            return
        print(f"按数据集过滤: {before_cnt} -> {after_cnt}")

    # 聚合
    print("\n正在聚合结果...")
    df_agg = aggregate_results(all_results)
    if df_agg.empty:
        print("错误: 聚合后结果为空")
        return

    # 输出目录
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # 保存 CSV
    csv_path = os.path.join(out_dir, 'thread_comparison_results.csv')
    df_agg.to_csv(csv_path, index=False)
    print(f"CSV结果已保存到: {csv_path}")

    # 依据 --threads 过滤聚合结果
    if args.threads:
        thread_set = set(args.threads)
        before_rows = len(df_agg)
        df_agg = df_agg[df_agg['threads'].isin(thread_set)]
        after_rows = len(df_agg)
        if after_rows == 0:
            print('错误: 过滤后的聚合结果为空，请检查 --threads 参数。')
            return
        print(f"按线程过滤: {before_rows} -> {after_rows}")

    # 决定线程列表
    threads = args.threads if args.threads else sorted(df_agg['threads'].dropna().unique())

    # 1) 每个实验方案：在不同线程下的对比（跨数据集展示）
    saved_imgs = []
    if not args.skip_exp_threads:
        for exp in args.experiments:
            saved_imgs += plot_experiment_by_threads(df_agg, exp, out_dir)
    else:
        print('跳过单方案多线程分离图 (--skip-exp-threads)。')

    # 2) 固定线程数：多种方案对比（跨数据集展示）
    for th in threads:
        saved_imgs += plot_schemes_by_thread(df_agg, th, out_dir)

    # 3) 固定数据集：多种方案随线程变化（跨线程展示）
    datasets_for_plot = sorted(df_agg['dataset'].unique())
    for ds in datasets_for_plot:
        saved_imgs += plot_methods_by_dataset(df_agg, ds, out_dir)

    # 占比图后续任务中实现；这里根据 --no-ratio 预留逻辑（当前占位）
    if args.no_ratio:
        print('跳过占比图生成 (--no-ratio)。')
    else:
        # 占比图函数将在后续“新增比例图函数”任务中加入
        pass

    # 生成 Markdown 报告
    report_path = generate_markdown_report(df_agg, args.experiments if not args.skip_exp_threads else [], threads, out_dir)

    # 控制台汇总
    print("\n=== 线程维度汇总（均值） ===")
    disp = df_agg[['experiment', 'dataset', 'threads', 'avg_hnsw', 'avg_io', 'avg_hnswio', 'io_ratio']].copy()
    disp = disp.sort_values(['experiment', 'dataset', 'threads'])
    print(disp.to_string(index=False))

    print(f"\n所有图表与报告输出到: {out_dir}")


if __name__ == '__main__':
    main()

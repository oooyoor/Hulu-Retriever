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

# 计算 JSON 文件中的平均值
def calculate_avg(json_file, percent=0.95):
    """从JSON文件中计算平均值，默认使用最后95%的数据"""
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)

        # 获取 entries 数据，并取最后 percent% 的数据进行计算
        entries = data['entries']
        if len(entries) == 0:
            return None, None, None
        
        entries_to_consider = max(1, int(len(entries) * percent))
        filtered_entries = entries[-entries_to_consider:]

        # 计算 hnsw, io 和 hnswio 的平均值（单位：微秒，保持原单位）
        avg_hnsw = np.mean([entry['hnsw'] for entry in filtered_entries])
        avg_hnswio = np.mean([entry['hnswio'] for entry in filtered_entries])
        avg_io = np.mean([entry['io'] for entry in filtered_entries])

        return avg_hnsw, avg_hnswio, avg_io
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return None, None, None

# 从目录结构中收集所有实验结果
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
            
            # 遍历query_cnt文件夹 (如 10000)
            for query_cnt_dir in os.listdir(config_path):
                query_cnt_path = os.path.join(config_path, query_cnt_dir)
                if not os.path.isdir(query_cnt_path):
                    continue
                
                # 遍历repeat_id文件夹 (如 1, 2, 3)
                for repeat_id_dir in os.listdir(query_cnt_path):
                    repeat_path = os.path.join(query_cnt_path, repeat_id_dir)
                    if not os.path.isdir(repeat_path):
                        continue
                    
                    # 查找HNSWIO.json文件
                    json_file = os.path.join(repeat_path, "HNSWIO.json")
                    if os.path.exists(json_file):
                        avg_hnsw, avg_hnswio, avg_io = calculate_avg(json_file)
                        if avg_hnsw is not None and avg_hnswio > 0:
                            # 计算IO占比
                            io_ratio = (avg_io / avg_hnswio) * 100
                            hnsw_ratio = (avg_hnsw / avg_hnswio) * 100
                            
                            results.append({
                                'experiment': experiment_name,
                                'dataset': dataset_name,
                                'config': config_dir,
                                'query_cnt': query_cnt_dir,
                                'repeat_id': repeat_id_dir,
                                'avg_hnsw': avg_hnsw,
                                'avg_hnswio': avg_hnswio,
                                'avg_io': avg_io,
                                'io_ratio': io_ratio,
                                'hnsw_ratio': hnsw_ratio
                            })
    
    return results

# 汇总所有实验结果
def aggregate_results(all_results):
    """对每个数据集和实验配置的结果进行聚合（跨多次重复实验）"""
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("Warning: No results collected")
        return pd.DataFrame()
    
    # 按实验方法和数据集分组，计算平均值和标准差
    df_agg = df.groupby(['experiment', 'dataset'], as_index=False).agg({
        'avg_hnsw': 'mean',
        'avg_hnswio': 'mean',
        'avg_io': 'mean',
        'io_ratio': ['mean', 'std'],
        'hnsw_ratio': 'mean'
    })
    
    # 展平列名
    df_agg.columns = ['experiment', 'dataset', 
                      'avg_hnsw', 'avg_hnswio', 'avg_io',
                      'io_ratio_mean', 'io_ratio_std',
                      'hnsw_ratio']
    
    # 重命名列以便后续使用
    df_agg = df_agg.rename(columns={
        'io_ratio_mean': 'io_ratio',
        'io_ratio_std': 'io_ratio_std'
    })
    
    return df_agg

# 绘制IO占比对比图
def plot_io_ratio_comparison(df_agg, results_dir):
    """绘制单层和多层文件系统结构的IO时间占比对比"""
    if df_agg.empty:
        print("Warning: No data to plot")
        return
    
    # 只保留单层和多层的结果
    df_filtered = df_agg[df_agg['experiment'].isin(['single', 'multi'])].copy()
    
    if df_filtered.empty:
        print("Warning: No single or multi layer results found")
        return
    
    # 设置白色背景
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('white')
    
    datasets = sorted(df_filtered['dataset'].unique())
    x = np.arange(len(datasets))
    width = 0.35
    
    # 获取单层和多层的数据
    single_data = df_filtered[df_filtered['experiment'] == 'single']
    multi_data = df_filtered[df_filtered['experiment'] == 'multi']
    
    single_ratios = []
    multi_ratios = []
    single_stds = []
    multi_stds = []
    
    for dataset in datasets:
        single_row = single_data[single_data['dataset'] == dataset]
        multi_row = multi_data[multi_data['dataset'] == dataset]
        
        if not single_row.empty:
            single_ratios.append(single_row['io_ratio'].values[0])
            single_stds.append(single_row['io_ratio_std'].values[0] if 'io_ratio_std' in single_row.columns else 0)
        else:
            single_ratios.append(0)
            single_stds.append(0)
        
        if not multi_row.empty:
            multi_ratios.append(multi_row['io_ratio'].values[0])
            multi_stds.append(multi_row['io_ratio_std'].values[0] if 'io_ratio_std' in multi_row.columns else 0)
        else:
            multi_ratios.append(0)
            multi_stds.append(0)
    
    # 定义颜色
    color_single = '#ED7D31'  # 橙色
    color_multi = '#70AD47'   # 绿色
    
    # 绘制柱状图
    bars1 = ax.bar(x - width/2, single_ratios, width, label='Single-Layer Directory Structure',
                   color=color_single, alpha=0.85, edgecolor='black', linewidth=0.8,
                   yerr=single_stds, capsize=5, error_kw={'elinewidth': 1.5})
    bars2 = ax.bar(x + width/2, multi_ratios, width, label='Multi-Layer Directory Structure',
                   color=color_multi, alpha=0.85, edgecolor='black', linewidth=0.8,
                   yerr=multi_stds, capsize=5, error_kw={'elinewidth': 1.5})
    
    # 添加50%参考线
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='50% Threshold')
    
    # 设置图表属性
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('IO Time Ratio (%)', fontsize=12, fontweight='bold')
    ax.set_title('IO Time Ratio', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=11)
    
    # 计算合适的y轴上限，为图例留出足够空间
    max_ratio = max(max(single_ratios + multi_ratios) if single_ratios + multi_ratios else [0], 50)
    ax.set_ylim([0, max_ratio * 1.3])  # 增加更多空间，确保图例不遮挡
    
    # 将图例放在图片内部右上角
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95, ncol=1)
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')
    
    # 添加数值标签
    max_ratio = max(max(single_ratios + multi_ratios) if single_ratios + multi_ratios else [0], 50)
    label_offset = max_ratio * 0.02
    
    for i, (single_ratio, multi_ratio) in enumerate(zip(single_ratios, multi_ratios)):
        if single_ratio > 0:
            ax.text(i - width/2, single_ratio + label_offset, f'{single_ratio:.1f}%', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        if multi_ratio > 0:
            ax.text(i + width/2, multi_ratio + label_offset, f'{multi_ratio:.1f}%', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    png_file = os.path.join(results_dir, "io_ratio_comparison.png")
    pdf_file = os.path.join(results_dir, "io_ratio_comparison.pdf")
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"IO占比对比图已保存到: {png_file} 和 {pdf_file}")

# 绘制堆叠百分比图
def plot_stacked_ratio(df_agg, results_dir):
    """绘制堆叠百分比图，展示IO和HNSW的占比"""
    if df_agg.empty:
        print("Warning: No data to plot")
        return
    
    # 只保留单层和多层的结果
    df_filtered = df_agg[df_agg['experiment'].isin(['single', 'multi'])].copy()
    
    if df_filtered.empty:
        print("Warning: No single or multi layer results found")
        return
    
    # 设置白色背景
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('white')
    
    datasets = sorted(df_filtered['dataset'].unique())
    x = np.arange(len(datasets))
    width = 0.35
    
    # 获取单层和多层的数据
    single_data = df_filtered[df_filtered['experiment'] == 'single']
    multi_data = df_filtered[df_filtered['experiment'] == 'multi']
    
    single_io_ratios = []
    single_hnsw_ratios = []
    multi_io_ratios = []
    multi_hnsw_ratios = []
    
    for dataset in datasets:
        single_row = single_data[single_data['dataset'] == dataset]
        multi_row = multi_data[multi_data['dataset'] == dataset]
        
        if not single_row.empty:
            single_io_ratios.append(single_row['io_ratio'].values[0])
            single_hnsw_ratios.append(single_row['hnsw_ratio'].values[0])
        else:
            single_io_ratios.append(0)
            single_hnsw_ratios.append(0)
        
        if not multi_row.empty:
            multi_io_ratios.append(multi_row['io_ratio'].values[0])
            multi_hnsw_ratios.append(multi_row['hnsw_ratio'].values[0])
        else:
            multi_io_ratios.append(0)
            multi_hnsw_ratios.append(0)
    
    # 定义颜色
    color_io_single = '#ED7D31'  # 橙色
    color_io_multi = '#70AD47'   # 绿色
    color_hnsw = '#D9D9D9'       # 浅灰色
    
    # 绘制堆叠柱状图
    ax.bar(x - width/2, single_io_ratios, width, label='Single-Layer IO', 
           color=color_io_single, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax.bar(x - width/2, single_hnsw_ratios, width, bottom=single_io_ratios, 
           color=color_hnsw, alpha=0.85, edgecolor='black', linewidth=0.8)
    
    ax.bar(x + width/2, multi_io_ratios, width, label='Multi-Layer IO', 
           color=color_io_multi, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax.bar(x + width/2, multi_hnsw_ratios, width, bottom=multi_io_ratios, 
           color=color_hnsw, alpha=0.85, edgecolor='black', linewidth=0.8)
    
    # 添加50%参考线
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='50% Threshold')
    
    # 设置图表属性
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time Ratio (%)', fontsize=12, fontweight='bold')
    ax.set_title('Time Breakdown Ratio', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=11)
    ax.set_ylim([0, 100])
    # 将图例放在图片内部右上角
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95, ncol=1)
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')
    
    # 添加数值标签
    for i, (s_io, s_hnsw, m_io, m_hnsw) in enumerate(zip(
        single_io_ratios, single_hnsw_ratios, multi_io_ratios, multi_hnsw_ratios)):
        if s_io > 0:
            ax.text(i - width/2, s_io / 2, f'{s_io:.1f}%', 
                   ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        if m_io > 0:
            ax.text(i + width/2, m_io / 2, f'{m_io:.1f}%', 
                   ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    plt.tight_layout()
    
    # 保存图表
    png_file = os.path.join(results_dir, "stacked_ratio_comparison.png")
    pdf_file = os.path.join(results_dir, "stacked_ratio_comparison.pdf")
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"堆叠占比对比图已保存到: {png_file} 和 {pdf_file}")

# 生成Markdown报告
def generate_ratio_report(df_agg, results_dir):
    """生成IO占比分析报告"""
    md_file = os.path.join(results_dir, "io_ratio_analysis.md")
    
    # 只保留单层和多层的结果
    df_filtered = df_agg[df_agg['experiment'].isin(['single', 'multi'])].copy()
    
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# IO Time Ratio Analysis\n\n")
        f.write("本报告分析了单层和多层文件系统结构的IO时间占比。\n\n")
        
        f.write("## 结果汇总\n\n")
        f.write("### IO时间占比对比（单位：%）\n\n")
        
        # 创建对比表格
        df_report = df_filtered[['experiment', 'dataset', 'io_ratio', 'hnsw_ratio', 'avg_io', 'avg_hnswio']].copy()
        df_report['io_ratio'] = df_report['io_ratio'].round(2)
        df_report['hnsw_ratio'] = df_report['hnsw_ratio'].round(2)
        df_report = df_report.rename(columns={
            'experiment': 'Method',
            'dataset': 'Dataset',
            'io_ratio': 'IO Ratio (%)',
            'hnsw_ratio': 'HNSW Ratio (%)',
            'avg_io': 'Avg IO Time (μs)',
            'avg_hnswio': 'Avg Total Time (μs)'
        })
        f.write(df_report.to_markdown(index=False))
        f.write("\n\n")
        
        # 按数据集分组显示
        f.write("### 按数据集分组对比\n\n")
        for dataset in sorted(df_filtered['dataset'].unique()):
            f.write(f"#### {dataset}\n\n")
            dataset_df = df_filtered[df_filtered['dataset'] == dataset][
                ['experiment', 'io_ratio', 'hnsw_ratio', 'avg_io', 'avg_hnswio']
            ]
            dataset_df = dataset_df.rename(columns={
                'experiment': 'Method',
                'io_ratio': 'IO Ratio (%)',
                'hnsw_ratio': 'HNSW Ratio (%)',
                'avg_io': 'Avg IO Time (μs)',
                'avg_hnswio': 'Avg Total Time (μs)'
            })
            dataset_df['IO Ratio (%)'] = dataset_df['IO Ratio (%)'].round(2)
            dataset_df['HNSW Ratio (%)'] = dataset_df['HNSW Ratio (%)'].round(2)
            f.write(dataset_df.to_markdown(index=False))
            f.write("\n\n")
        
        # 统计信息
        f.write("### 统计信息\n\n")
        single_ratios = df_filtered[df_filtered['experiment'] == 'single']['io_ratio']
        multi_ratios = df_filtered[df_filtered['experiment'] == 'multi']['io_ratio']
        
        f.write(f"- **Single-Layer平均IO占比**: {single_ratios.mean():.2f}%\n")
        f.write(f"- **Single-Layer最大IO占比**: {single_ratios.max():.2f}%\n")
        f.write(f"- **Single-Layer最小IO占比**: {single_ratios.min():.2f}%\n")
        f.write(f"- **Single-Layer超过50%的数据集数量**: {(single_ratios > 50).sum()}/{len(single_ratios)}\n\n")
        
        f.write(f"- **Multi-Layer平均IO占比**: {multi_ratios.mean():.2f}%\n")
        f.write(f"- **Multi-Layer最大IO占比**: {multi_ratios.max():.2f}%\n")
        f.write(f"- **Multi-Layer最小IO占比**: {multi_ratios.min():.2f}%\n")
        f.write(f"- **Multi-Layer超过50%的数据集数量**: {(multi_ratios > 50).sum()}/{len(multi_ratios)}\n\n")
        
        f.write("### 图表对比\n\n")
        f.write("![IO占比对比](io_ratio_comparison.png)\n\n")
        f.write("![堆叠占比对比](stacked_ratio_comparison.png)\n")
    
    print(f"IO占比分析报告已保存到: {md_file}")

# 主函数
def main():
    """主函数：分析单层和多层文件系统结构的IO时间占比"""
    
    # 定义两种实验方法的目录路径
    result_dirs = {
        "single": "/home/zqf/Hulu-Retriever/MakeFsWorkSpace/results/Fs_results",
        "multi": "/home/zqf/Hulu-Retriever/MakeFsWorkSpace/results/MultiLayer_Fs_results"
    }
    
    # 收集所有结果
    all_results = []
    for exp_name, base_dir in result_dirs.items():
        print(f"正在收集 {exp_name} 实验结果...")
        results = collect_results_from_dir(base_dir, exp_name)
        print(f"  找到 {len(results)} 个结果文件")
        all_results.extend(results)
    
    if not all_results:
        print("错误: 未找到任何实验结果文件")
        return
    
    # 聚合结果
    print("\n正在聚合结果...")
    df_agg = aggregate_results(all_results)
    
    if df_agg.empty:
        print("错误: 聚合后结果为空")
        return
    
    # 创建输出目录
    output_dir = "/home/zqf/Hulu-Retriever/MakeFsWorkSpace/results/io_ratio_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存CSV
    csv_file = os.path.join(output_dir, "io_ratio_results.csv")
    df_agg.to_csv(csv_file, index=False)
    print(f"\nCSV结果已保存到: {csv_file}")
    
    # 生成图表
    plot_io_ratio_comparison(df_agg, output_dir)
    plot_stacked_ratio(df_agg, output_dir)
    
    # 生成报告
    generate_ratio_report(df_agg, output_dir)
    
    # 打印汇总信息
    print("\n=== IO占比汇总 ===")
    df_display = df_agg[['experiment', 'dataset', 'io_ratio', 'hnsw_ratio']].copy()
    df_display['io_ratio'] = df_display['io_ratio'].round(2)
    df_display['hnsw_ratio'] = df_display['hnsw_ratio'].round(2)
    print(df_display.to_string(index=False))
    
    print(f"\n所有结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()


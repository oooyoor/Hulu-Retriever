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
                        if avg_hnsw is not None:
                            results.append({
                                'experiment': experiment_name,
                                'dataset': dataset_name,
                                'config': config_dir,
                                'query_cnt': query_cnt_dir,
                                'repeat_id': repeat_id_dir,
                                'avg_hnsw': avg_hnsw,
                                'avg_hnswio': avg_hnswio,
                                'avg_io': avg_io
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
        'avg_hnsw': ['mean', 'std'],
        'avg_hnswio': ['mean', 'std'],
        'avg_io': ['mean', 'std']
    })
    
    # 展平列名
    df_agg.columns = ['experiment', 'dataset', 
                      'avg_hnsw_mean', 'avg_hnsw_std',
                      'avg_hnswio_mean', 'avg_hnswio_std',
                      'avg_io_mean', 'avg_io_std']
    
    # 重命名列以便后续使用
    df_agg = df_agg.rename(columns={
        'avg_hnsw_mean': 'avg_hnsw',
        'avg_hnswio_mean': 'avg_hnswio',
        'avg_io_mean': 'avg_io'
    })
    
    return df_agg

# 生成 Markdown 报告
def generate_markdown(df_agg, results_dir):
    """生成Markdown格式的报告"""
    md_file = os.path.join(results_dir, "comparison_results.md")
    
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# 实验结果对比分析\n\n")
        f.write("本报告对比了三种不同实验配置（Offset、Fs、MultiLayer_Fs）在不同数据集上的性能表现。\n\n")
        
        f.write("## 结果汇总\n\n")
        f.write("### 平均耗时对比（单位：微秒）\n\n")
        
        # 创建简化的表格（只显示平均值）
        df_simple = df_agg[['experiment', 'dataset', 'avg_hnsw', 'avg_hnswio', 'avg_io']].copy()
        f.write(df_simple.to_markdown(index=False))
        f.write("\n\n")
        
        # 按数据集分组显示
        f.write("### 按数据集分组对比\n\n")
        for dataset in sorted(df_agg['dataset'].unique()):
            f.write(f"#### {dataset}\n\n")
            dataset_df = df_agg[df_agg['dataset'] == dataset][
                ['experiment', 'avg_hnsw', 'avg_hnswio', 'avg_io']
            ]
            f.write(dataset_df.to_markdown(index=False))
            f.write("\n\n")
        
        f.write("### 图表对比\n\n")
        f.write("![对比结果](comparison_results.png)\n\n")
        
        f.write("![对比结果PDF](comparison_results.pdf)\n")
    
    print(f"Markdown报告已保存到: {md_file}")

# 绘制对比图表
def plot_comparison(df_agg, results_dir):
    """绘制三种方法的堆叠柱状图对比，突出IO时间的变化"""
    if df_agg.empty:
        print("Warning: No data to plot")
        return
    
    # 设置字体和样式 - 白色背景
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('default')  # 使用默认样式，白色背景
    
    datasets = sorted(df_agg['dataset'].unique())
    experiments = ['offset', 'single', 'multi']
    
    # 创建单个图表，使用堆叠柱状图，白色背景
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # 准备数据：为每个数据集和每个实验方法准备HNSW和IO时间
    x = np.arange(len(datasets))
    width = 0.25  # 每个方法之间的间距
    
    # 定义颜色：使用论文风格的配色方案
    # IO部分在底部，用不同颜色突出三种方法的差异
    colors_io = {
        'offset': '#4472C4',  # 深蓝色 - Raw Disk (论文常用蓝色)
        'single': '#ED7D31',  # 橙色 - Single Layer FS
        'multi': '#70AD47'    # 绿色 - Multi Layer FS
    }
    # HNSW部分统一用浅灰色（因为HNSW时间应该相似）
    color_hnsw = '#D9D9D9'  # 浅灰色，所有方法统一
    
    # 为每个实验方法绘制堆叠柱状图
    max_total_time = 0  # 用于后续添加标签时计算偏移
    
    for i, exp in enumerate(experiments):
        exp_df = df_agg[df_agg['experiment'] == exp]
        
        # 收集HNSW和IO数据
        hnsw_values = []
        io_values = []
        total_values = []
        
        for dataset in datasets:
            dataset_data = exp_df[exp_df['dataset'] == dataset]
            if not dataset_data.empty:
                hnsw_val = dataset_data['avg_hnsw'].values[0]
                io_val = dataset_data['avg_io'].values[0]
                total_val = dataset_data['avg_hnswio'].values[0]
                hnsw_values.append(hnsw_val)
                io_values.append(io_val)
                total_values.append(total_val)
                max_total_time = max(max_total_time, total_val)
            else:
                hnsw_values.append(0)
                io_values.append(0)
                total_values.append(0)
        
        # 绘制堆叠柱状图：先绘制IO（底部），再绘制HNSW（顶部）
        x_pos = x + i * width
        # IO部分在底部，使用不同颜色突出差异（不添加label，后面统一处理图例）
        ax.bar(x_pos, io_values, width, 
               color=colors_io[exp], alpha=0.85, 
               edgecolor='black', linewidth=0.8)
        
        # HNSW部分在IO之上（不添加label，后面统一处理图例）
        ax.bar(x_pos, hnsw_values, width, bottom=io_values, 
               color=color_hnsw, alpha=0.85, 
               edgecolor='black', linewidth=0.8)
    
    # 设置图表属性
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (μs)', fontsize=12, fontweight='bold')
    ax.set_title('Query Latency Breakdown', fontsize=13, fontweight='bold', pad=15)
    # 横轴只显示数据集名字，位置在三个柱状图的中心
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=11)
    
    # 创建自定义图例：显示三种实验方法和HNSW（英文描述）
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors_io['offset'], edgecolor='black', alpha=0.85, label='Raw Disk Offset'),
        Patch(facecolor=colors_io['single'], edgecolor='black', alpha=0.85, label='Single-Layer Directory Structure'),
        Patch(facecolor=colors_io['multi'], edgecolor='black', alpha=0.85, label='Multi-Layer Directory Structure'),
        Patch(facecolor=color_hnsw, edgecolor='black', alpha=0.85, label='HNSW')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.95, ncol=1)
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')
    
    # 添加数值标签：在柱状图顶部显示总时间
    label_offset = max_total_time * 0.02
    for i, exp in enumerate(experiments):
        exp_df = df_agg[df_agg['experiment'] == exp]
        for j, dataset in enumerate(datasets):
            dataset_data = exp_df[exp_df['dataset'] == dataset]
            if not dataset_data.empty:
                total_time = dataset_data['avg_hnswio'].values[0]
                io_time = dataset_data['avg_io'].values[0]
                hnsw_time = dataset_data['avg_hnsw'].values[0]
                x_pos = x[j] + i * width
                # 在顶部显示总时间
                ax.text(x_pos, total_time + label_offset, 
                       f'{total_time:.0f}μs', ha='center', va='bottom', 
                       fontsize=8, fontweight='bold')
                # 在IO部分（底部）显示IO时间（只有当IO部分足够高时才显示）
                if io_time > max_total_time * 0.05:  # 只有当IO时间超过总时间的5%时才显示
                    ax.text(x_pos, io_time / 2, 
                           f'{io_time:.0f}μs', ha='center', va='center', 
                           fontsize=7, color='white', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
    
    plt.tight_layout()
    
    # 保存图表
    png_file = os.path.join(results_dir, "comparison_results.png")
    pdf_file = os.path.join(results_dir, "comparison_results.pdf")
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_file, bbox_inches='tight')
    plt.close()
    
    print(f"堆叠柱状图已保存到: {png_file} 和 {pdf_file}")

# 主函数：汇总分析所有实验结果
def main():
    """主函数：收集三种方法的结果并进行对比分析"""
    
    # 定义三种实验方法的目录路径
    result_dirs = {
        "offset": "/home/zqf/Hulu-Retriever/MakeFsWorkSpace/end_results/raw_results/Offset_results",
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
    output_dir = "/home/zqf/Hulu-Retriever/MakeFsWorkSpace/results/comparison_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存CSV
    csv_file = os.path.join(output_dir, "comparison_results.csv")
    df_agg.to_csv(csv_file, index=False)
    print(f"\nCSV结果已保存到: {csv_file}")
    
    # 生成Markdown报告
    generate_markdown(df_agg, output_dir)
    
    # 绘制对比图
    plot_comparison(df_agg, output_dir)
    
    # 打印汇总信息
    print("\n=== 结果汇总 ===")
    print(df_agg[['experiment', 'dataset', 'avg_hnsw', 'avg_hnswio', 'avg_io']].to_string(index=False))
    
    print(f"\n所有结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()

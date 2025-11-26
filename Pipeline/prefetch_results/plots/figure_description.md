# 图表说明文档 / Figure Description

## 图表标题 / Figure Title

**中文**: 预取执行时间线分析  
**English**: Prefetch Execution Timeline Analysis

---

## 图表组成部分说明 / Figure Components Description

### 1. 预取开始前 / Before Prefetch Start

**中文说明**:
- **含义**: 在预取操作开始之前，搜索过程已经执行的迭代次数
- **范围**: 从第 0 次迭代到 `avg_start_prefetch` 位置
- **特点**: 在这个阶段，搜索过程独立执行，没有预取操作参与

**English Description**:
- **Meaning**: The number of iterations executed before the prefetch operation starts
- **Range**: From iteration 0 to `avg_start_prefetch` position
- **Characteristics**: During this phase, the search process executes independently without prefetch operations

---

### 2. 重叠执行空间 / Overlap Execution Space

**中文说明**:
- **含义**: 预取操作与搜索过程同时执行的重叠时间段
- **范围**: 从 `avg_start_prefetch` 到 `avg_last_prefetch` 位置
- **特点**: 
  - 这是预取和搜索**并行执行**的关键区域
  - 重叠空间的大小反映了预取策略的有效性
  - 重叠空间越长，说明预取操作覆盖的搜索迭代范围越大
  - 百分比标注显示重叠空间占总迭代次数的比例

**English Description**:
- **Meaning**: The overlapping time period where prefetch operations and search process execute simultaneously
- **Range**: From `avg_start_prefetch` to `avg_last_prefetch` position
- **Characteristics**:
  - This is the key region where prefetch and search execute **in parallel**
  - The size of overlap space reflects the effectiveness of prefetch strategy
  - Longer overlap space indicates that prefetch covers a larger range of search iterations
  - The percentage label shows the ratio of overlap space to total iterations

---

### 3. 预取结束后 / After Prefetch End

**中文说明**:
- **含义**: 在理论上应该结束预取操作之后，搜索过程继续执行的剩余迭代次数
- **范围**: 从 `avg_last_prefetch` 到 `avg_iter_count` 位置
- **特点**:
  - 这个阶段预取操作理论上已经结束，但搜索仍在继续
  - 表示搜索过程在预取结束后还需要执行的迭代次数
  - 这部分迭代不再受益于预取优化

**English Description**:
- **Meaning**: The remaining iterations executed after the prefetch operation should theoretically end
- **Range**: From `avg_last_prefetch` to `avg_iter_count` position
- **Characteristics**:
  - During this phase, prefetch should have ended theoretically, but search continues
  - Represents the iterations that search still needs to execute after prefetch ends
  - These iterations no longer benefit from prefetch optimization

---

## 关键指标解释 / Key Metrics Explanation

### 重叠执行空间占比 / Overlap Execution Space Ratio

**计算公式 / Formula**:
```
重叠空间占比 = (avg_last_prefetch - avg_start_prefetch) / avg_iter_count × 100%
Overlap Ratio = (avg_last_prefetch - avg_start_prefetch) / avg_iter_count × 100%
```

**中文说明**:
- 这个百分比显示重叠执行空间占总迭代次数的比例
- 百分比越高，说明预取操作覆盖的搜索迭代范围越大
- 理想情况下，我们希望这个比例尽可能大，以最大化预取的收益

**English Description**:
- This percentage shows the ratio of overlap execution space to total iterations
- Higher percentage indicates that prefetch covers a larger range of search iterations
- Ideally, we want this ratio to be as large as possible to maximize prefetch benefits

---

## 图表解读建议 / Interpretation Guidelines

### 中文建议

1. **重叠空间占比分析**:
   - 如果重叠空间占比高（>30%），说明预取策略有效，覆盖了大部分搜索迭代
   - 如果重叠空间占比低（<10%），可能需要调整预取开始时机，让预取更早介入

2. **三个部分的平衡**:
   - 理想情况下，"预取开始前"部分应该较小，让预取尽早开始
   - "重叠执行空间"部分应该尽可能大，最大化预取的并行收益
   - "预取结束后"部分表示搜索的剩余工作，这部分无法通过预取优化

3. **不同数据集对比**:
   - 通过对比不同数据集的堆叠柱状图，可以识别哪些数据集从预取中受益最多
   - 重叠空间占比高的数据集，说明预取策略更适合该数据集的搜索特性

### English Guidelines

1. **Overlap Space Ratio Analysis**:
   - If overlap ratio is high (>30%), it indicates effective prefetch strategy covering most search iterations
   - If overlap ratio is low (<10%), consider adjusting prefetch start timing to engage earlier

2. **Balance of Three Components**:
   - Ideally, "Before Prefetch Start" should be small to start prefetch early
   - "Overlap Execution Space" should be as large as possible to maximize parallel benefits
   - "After Prefetch End" represents remaining search work that cannot benefit from prefetch

3. **Cross-Dataset Comparison**:
   - By comparing stacked bars across datasets, identify which datasets benefit most from prefetch
   - Datasets with high overlap ratio indicate that prefetch strategy suits their search characteristics

---

## 技术细节 / Technical Details

### 数据来源 / Data Source
- 数据文件: `prefetch.json`
- 关键字段:
  - `avg_start_prefetch`: 平均预取开始迭代位置
  - `avg_last_prefetch`: 平均预取结束迭代位置（理论上）
  - `avg_iter_count`: 平均总迭代次数

### 可视化参数 / Visualization Parameters
- 图表类型: 堆叠柱状图 (Stacked Bar Chart)
- 分辨率: 300 DPI (PNG), 矢量格式 (PDF)
- 配色方案: 专业学术配色 (Professional Academic Color Scheme)

---

## 使用场景 / Use Cases

### 中文
- **论文插图**: 用于展示预取策略在不同数据集上的执行特征
- **性能分析**: 评估预取策略的有效性和优化潜力
- **策略优化**: 识别需要调整预取时机的数据集

### English
- **Paper Illustration**: Demonstrate prefetch execution characteristics across different datasets
- **Performance Analysis**: Evaluate prefetch strategy effectiveness and optimization potential
- **Strategy Optimization**: Identify datasets that need prefetch timing adjustments


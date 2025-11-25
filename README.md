Retriever_Improvement

1. `set_env.sh` : Prepare third-party libraries

# 1. 查询难度划分
本文将基于`iter_base[i]`对查询难度进行分类：
- 把 query 按这个值排序，然后分成 5 桶，比如：
Q_easy = 前 25%：iter_base 最小
Q_mid1 = 25% - 50%
Q_mid2 = 50% - 75%
Q_mid3 = 50% - 75%
Q_hard = 最后 25%：iter_base 最大
Q_hard = recall < 1.0 

### 1. 结果
- baseline结果:
`/home/zqf/Hulu-Retriever/SearchDifficultyResults/BaselineResults/`
- 
`/home/zqf/Hulu-Retriever/SearchDifficultyResults`:

# 2. 
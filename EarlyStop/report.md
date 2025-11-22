# 实验结果对比报告

| method   | dataset       |   threads |   ef |   iodepth |   query_cnt |   repeat |     recall |     iter |     dist |
|:---------|:--------------|----------:|-----:|----------:|------------:|---------:|-----------:|---------:|---------:|
| end      | fashion-mnist |         1 |  200 |       256 |       10000 |        1 |   0.900456 | nan      |   nan    |
| end      | glove1M       |         1 |  500 |       256 |       10000 |        1 |   0.932409 | nan      |   nan    |
| end      | deep          |         1 |  400 |       256 |       10000 |        1 |   0.986298 | nan      |   nan    |
| end      | mnist         |         1 |  200 |       256 |       10000 |        1 |   0.907611 | nan      |   nan    |
| end      | sift          |         1 |  300 |       256 |       10000 |        1 |   0.972507 | nan      |   nan    |
| end      | glove         |         1 |  500 |       256 |       10000 |        1 |   0.813555 | nan      |   nan    |
| end      | fashion-mnist |         1 |  200 |       256 |       10000 |        1 | nan        |  77.5303 |  1262.94 |
| end      | glove1M       |         1 |  500 |       256 |       10000 |        1 | nan        | 465.521  | 22475.2  |
| end      | deep          |         1 |  400 |       256 |       10000 |        1 | nan        | 361.033  |  8180.07 |
| end      | mnist         |         1 |  200 |       256 |       10000 |        1 | nan        |  72.4405 |  1310.08 |
| end      | sift          |         1 |  300 |       256 |       10000 |        1 | nan        | 231.961  |  5256.62 |
| end      | glove         |         1 |  500 |       256 |       10000 |        1 | nan        | 481.119  | 12620.8  |
| recall   | fashion-mnist |         1 |  100 |       256 |       10000 |        1 |   0.986846 | nan      |   nan    |
| recall   | fashion-mnist |         1 |  200 |       256 |       10000 |        1 |   0.996173 | nan      |   nan    |
| recall   | glove1M       |         1 |  450 |       256 |       10000 |        1 |   0.928585 | nan      |   nan    |
| recall   | glove1M       |         1 |  500 |       256 |       10000 |        1 |   0.935751 | nan      |   nan    |
| recall   | deep          |         1 |  350 |       256 |       10000 |        1 |   0.987066 | nan      |   nan    |
| recall   | deep          |         1 |  400 |       256 |       10000 |        1 |   0.989786 | nan      |   nan    |
| recall   | mnist         |         1 |  100 |       256 |       10000 |        1 |   0.993002 | nan      |   nan    |
| recall   | mnist         |         1 |  200 |       256 |       10000 |        1 |   0.999045 | nan      |   nan    |
| recall   | sift          |         1 |  300 |       256 |       10000 |        1 |   0.990321 | nan      |   nan    |
| recall   | sift          |         1 |  185 |       256 |       10000 |        1 |   0.973706 | nan      |   nan    |
| recall   | glove         |         1 |  450 |       256 |       10000 |        1 |   0.80712  | nan      |   nan    |
| recall   | glove         |         1 |  500 |       256 |       10000 |        1 |   0.816267 | nan      |   nan    |
| recall   | fashion-mnist |         1 |  100 |       256 |       10000 |        1 | nan        | 100.264  |  1637.18 |
| recall   | fashion-mnist |         1 |  200 |       256 |       10000 |        1 | nan        | 200.121  |  3222.16 |
| recall   | glove1M       |         1 |  450 |       256 |       10000 |        1 | nan        | 451.497  | 21782.6  |
| recall   | glove1M       |         1 |  500 |       256 |       10000 |        1 | nan        | 501.379  | 24186.4  |
| recall   | deep          |         1 |  350 |       256 |       10000 |        1 | nan        | 350.454  |  7960.47 |
| recall   | deep          |         1 |  400 |       256 |       10000 |        1 | nan        | 400.404  |  9085.27 |
| recall   | mnist         |         1 |  100 |       256 |       10000 |        1 | nan        | 100.298  |  1846.13 |
| recall   | mnist         |         1 |  200 |       256 |       10000 |        1 | nan        | 200.162  |  3668.87 |
| recall   | sift          |         1 |  300 |       256 |       10000 |        1 | nan        | 300.387  |  6814.94 |
| recall   | sift          |         1 |  185 |       256 |       10000 |        1 | nan        | 185.593  |  4217.9  |
| recall   | glove         |         1 |  450 |       256 |       10000 |        1 | nan        | 456.042  | 11971.5  |
| recall   | glove         |         1 |  500 |       256 |       10000 |        1 | nan        | 505.782  | 13270.2  |

## 图像
![](plots/recall.png)

![](plots/iter.png)

![](plots/dist.png)


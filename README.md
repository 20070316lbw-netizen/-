# 基于 K-Means 的汽车行驶运动学片段分类

模式识别与机器学习课程设计 · 项目三

## 项目简介

对真实采集的汽车行驶数据（`roadfile.xlsx`，86927 行、1Hz 采样、14 个通道）进行
**运动学片段（kinematic segment）切分**，提取每个片段的运动学特征，再用
**K-Means 聚类**将片段归为若干典型行驶工况（市区/城郊/高速等），
并通过可视化分析各工况的特征差异。

## 目录结构

```
0616/
├── data/
│   └── roadfile.xlsx          # 原始数据（不入仓库）
├── src/
│   ├── load_data.py           # 阶段1：读取+清洗+算加速度
│   ├── segment.py             # 阶段2：切运动学片段
│   ├── features.py            # 阶段3：片段特征工程
│   └── cluster.py             # 阶段4：标准化+PCA+选K+KMeans+评估
├── run.py                     # 主流程：一键跑完并出图
├── outputs/                   # 图表与结果 CSV（运行后生成）
└── README.md
```

## 核心定义

- **运动学片段**：从一次怠速起点，到下一次怠速起点，构成
  「怠速→加速→巡航→减速→停车」一个完整循环（汽车行驶工况标准定义）。
- **怠速判据**：`GPS车速 < 1 km/h`（取 <1 而非 ==0，以容忍 GPS 零点抖动）。

## 运行方法

```bash
# 1. 安装依赖
uv add pandas numpy scikit-learn matplotlib openpyxl

# 2. 一键运行完整流程
uv run python run.py
```

运行后 `outputs/` 下会生成：

| 文件 | 内容 |
|------|------|
| `01_speed_profile.png` | 车速曲线 + 怠速切分示意 |
| `02_segment_length_hist.png` | 片段时长分布 |
| `03_k_selection.png` | 肘部法 + 轮廓系数选 K |
| `04_cluster_scatter.png` | PCA 投影聚类散点 |
| `05_cluster_radar.png` | 各簇特征雷达对比 |
| `feature_matrix.csv` | 片段特征矩阵 |
| `k_selection.csv` | 各 K 评估指标 |
| `cluster_profile.csv` | 各簇画像 |
| `segment_clusters.csv` | 带簇标签与工况解释的片段汇总 |

## 单模块调试

每个 `src/` 模块都可单独运行查看中间结果（需在项目根目录下）：

```bash
uv run python src/load_data.py    # 看清洗后数据
uv run python src/segment.py      # 看切了多少片段
uv run python src/features.py     # 看特征矩阵
uv run python src/cluster.py      # 看选K与聚类
```

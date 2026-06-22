# 基于 K-Means 的汽车行驶运动学片段分类

模式识别与机器学习课程设计 · 项目三

## 项目简介

对真实采集的汽车行驶数据（`roadfile.xlsx`，86927 行、1Hz 采样、14 个通道）进行
**运动学片段（kinematic segment）切分**，提取每个片段的运动学特征，再用
**K-Means 聚类**将片段归为若干典型行驶工况（市区/城郊/高速等），
并通过可视化分析各工况的特征差异。

项目提供两种使用方式：

- **一键批处理**（`run.py`）：跑完整流程并把所有图表与结果 CSV 输出到 `outputs/`。
- **交互式界面**（`app.py`，基于 Streamlit）：在网页上实时调节怠速阈值、最小片段
  长度、K 值等参数，即时查看切分与聚类结果。

## 目录结构

```
.
├── data/
│   └── roadfile.xlsx          # 原始行驶数据
├── src/
│   ├── load_data.py           # 阶段1：读取 + 清洗 + 算加速度
│   ├── segment.py             # 阶段2：切运动学片段
│   ├── features.py            # 阶段3：片段特征工程（14 个运动学特征）
│   └── cluster.py             # 阶段4：标准化 + PCA + 选 K + KMeans + 评估
├── run.py                     # 主流程：一键跑完并出图（批处理）
├── app.py                     # Streamlit 交互界面
├── main.py                    # 占位入口
├── outputs/                   # 图表与结果 CSV（运行 run.py 后生成）
├── pyproject.toml             # 项目元数据与依赖（uv）
├── requirements.txt           # 依赖清单（pip）
└── README.md
```

## 核心定义

- **运动学片段**：从一次怠速起点，到下一次怠速起点，构成
  「怠速→加速→巡航→减速→停车」一个完整循环（汽车行驶工况标准定义）。
- **怠速判据**：`GPS车速 < 1 km/h`（取 <1 而非 ==0，以容忍 GPS 零点抖动）。
- **片段过滤**：丢弃时长 < 20 秒、或非怠速点 < 5 个（车几乎没动）的片段。

## 片段特征（共 14 个）

| 类别 | 特征 | 含义 |
|------|------|------|
| 时间结构 | `duration` | 片段总时长 (s) |
| | `idle_ratio` | 怠速时间占比（speed < 阈值）|
| | `accel_ratio` | 加速时间占比（acc > 0.1 m/s²）|
| | `decel_ratio` | 减速时间占比（acc < -0.1 m/s²）|
| | `cruise_ratio` | 匀速时间占比（其余运动中点）|
| 速度 | `v_mean` | 平均车速 (km/h，含怠速) |
| | `v_mean_run` | 运行平均车速 (km/h，仅非怠速点) |
| | `v_max` | 最高车速 (km/h) |
| | `v_std` | 车速标准差 |
| 加速度 | `a_mean_pos` | 平均加速度（仅加速段, m/s²）|
| | `a_mean_neg` | 平均减速度（仅减速段, m/s²）|
| | `a_max` | 最大加速度 |
| | `a_std` | 加速度标准差 |
| | `v_times_a_max` | 最大比功率近似（v·a 最大值），刻画驾驶激烈程度 |

聚类时排除绝对量纲后保留 13 个特征，经标准化与 PCA 降维再做 K-Means。

## 安装依赖

使用 uv（推荐）：

```bash
uv add pandas numpy scikit-learn matplotlib openpyxl streamlit
```

或使用 pip：

```bash
pip install -r requirements.txt
```

需要 Python ≥ 3.12。

## 运行方法

### 方式一：一键批处理

```bash
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

### 方式二：交互式界面

```bash
uv run streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`。界面包含：

- **侧边栏**：实时调节怠速阈值、最小片段长度、K 值（支持按轮廓系数自动选 K）。
- **数据概览**：原始数据基本信息 + 车速曲线。
- **片段切分**：切出的片段数、时长分布、单片段车速曲线。
- **选 K 分析**：肘部法 + 轮廓系数。
- **聚类结果**：PCA 散点图、各簇画像与工况解释，并可下载结果 CSV。

## 单模块调试

每个 `src/` 模块都可单独运行查看中间结果（需在项目根目录下）：

```bash
uv run python src/load_data.py    # 看清洗后数据
uv run python src/segment.py      # 看切了多少片段
uv run python src/features.py     # 看特征矩阵
uv run python src/cluster.py      # 看选K与聚类
```

"""汽车行驶运动学片段 K-Means 聚类分析 —— 单文件整合版。

本文件把原 src/ 下的四个模块（load_data / segment / features / cluster）
与 app.py 的 Streamlit 交互界面整合到一起，功能与拆分版完全一致。

运行： uv run streamlit run main.py
然后浏览器会自动打开 http://localhost:8501

界面结构：
    侧边栏：参数调节（怠速阈值、最小片段长度、K 值、是否自动选 K）
    主区域：四个标签页
        1. 数据概览   —— 原始数据基本信息 + 车速曲线
        2. 片段切分   —— 切出的片段数、时长分布
        3. 选 K 分析  —— 肘部法 + 轮廓系数
        4. 聚类结果   —— 散点图、各簇画像、工况解释

----------------------------------------------------------------------
整体流程（四个阶段）：
    阶段1  读取与清洗原始行车数据
    阶段2  切分运动学片段（kinematic segment）
    阶段3  片段特征工程
    阶段4  标准化 -> PCA 降维 -> 选 K -> K-Means 聚类 -> 评估
----------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import font_manager
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler


# ======================================================================
# 阶段1：读取与清洗原始行车数据
# ----------------------------------------------------------------------
# 数据为 1Hz 等间隔采样，已确认无缺失值。本部分负责：
# - 读取 Excel 原始数据
# - 规范列名（中文 -> 英文，便于后续编程）
# - 基础清洗：去除车速异常值、确保数值类型
# - 计算逐秒加速度（由车速差分得到，单位 m/s^2）
# ======================================================================

# 创建字典用,防止中文导致出现问题
# 原始中文列名 -> 英文列名映射
# 配合 `df.rename` 使用, 批量改名字
COLUMN_MAP = {
    "时间": "time",
    "GPS车速": "speed",          # km/h
    "X轴加速度": "acc_x",
    "Y轴加速度": "acc_y",
    "Z轴加速度": "acc_z",
    "经度": "lon",
    "纬度": "lat",
    "发动机转速": "engine_rpm",
    "扭矩百分比": "torque_pct",
    "瞬时油耗": "fuel_rate",
    "油门踏板开度": "throttle",
    "空燃比": "afr",
    "发动机负荷百分比": "engine_load",
    "进气流量": "air_flow",
}


def load_raw(path: str) -> pd.DataFrame:
    """读取原始 Excel 并规范列名。"""
    df = pd.read_excel(path)

    # 使用创建的字典对文件内容进行名字修改
    df = df.rename(columns=COLUMN_MAP)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """基础清洗 + 计算加速度。

    步骤：
    1. 车速转为数值，负值/异常大值（>200km/h）视为无效，置 0。
    2. 计算逐秒加速度：a = dv/dt。采样间隔 dt = 1s，
       车速 km/h 需转 m/s（÷3.6）再差分，单位为 m/s^2。
    3. 首行加速度无前值，填 0。
    """
    df = df.copy()  # 神秘的 copy(),不过这个是真有用,否则会直接改原数据, 同时也跳出了 `SettingWithCopyWarning` 这个坑, `df = df.copy()` 则直接表明是独立副本,不出发警报


    # 1. 车速清洗

    # 强行转数据, 将无法转换的转成 NAN ,紧接着变成 0.0
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0.0)
    df.loc[df["speed"] < 0, "speed"] = 0.0
    df.loc[df["speed"] > 200, "speed"] = 0.0  # 物理上不可能，视为采集异常

    # 2. 加速度（m/s^2）：车速 km/h -> m/s 后逐秒差分
    speed_ms = df["speed"] / 3.6
    df["acc"] = speed_ms.diff().fillna(0.0)

    return df


def load_and_clean(path: str) -> pd.DataFrame:
    """便捷入口：读取 + 清洗一步到位。"""
    return clean(load_raw(path))


# ======================================================================
# 阶段2：切分运动学片段（kinematic segment）
# ----------------------------------------------------------------------
# 定义（方案A，汽车工程造行驶工况的标准定义）：
#     一个运动学片段 = 从一次怠速的起点，到下一次怠速的起点。
#     即「怠速 -> 加速 -> 巡航 -> 减速 -> 停车怠速」构成一个完整循环，
#     对应车辆一次"起步到再次停车"的过程。
#
# 怠速判据：speed < IDLE_SPEED_THRESHOLD（km/h）。
#     取 < 1 而非 == 0，是为了容忍 GPS 车速的零点抖动。
#
# 切分逻辑：
#     1. 给每个采样点打标记：是否怠速（speed < 阈值）。
#     2. 找出所有"怠速段"（连续的怠速点构成一段）。
#     3. 每个运动学片段从一个怠速段的【起点】开始，
#        到下一个怠速段的【起点】之前结束。
#     4. 过滤掉过短的片段（噪声）和全程怠速的片段（车没动）。
# ======================================================================

IDLE_SPEED_THRESHOLD = 1.0   # km/h，低于此视为怠速/停车
MIN_SEGMENT_LENGTH = 20      # 秒，短于此的片段丢弃（视为噪声）
MIN_RUNNING_POINTS = 5       # 片段内至少要有这么多个非怠速点，否则视为"车没真正动"


def _mark_idle(df: pd.DataFrame, threshold: float) -> pd.Series:
    """标记每个点是否为怠速点。"""
    return df["speed"] < threshold


def _find_idle_starts(is_idle: pd.Series) -> list[int]:
    """找出每一段连续怠速的【起始下标】。

    原理：is_idle 由 False->True 翻转处，即一个怠速段的开始。
    用 diff 检测上升沿。第0个点若本身是怠速，也算一个起点。
    """
    idle = is_idle.to_numpy()
    starts = []
    if idle[0]:
        starts.append(0)
    # 上升沿：前一个 False，当前 True
    rises = np.where((~idle[:-1]) & (idle[1:]))[0] + 1
    starts.extend(rises.tolist())
    return sorted(set(starts))


def split_segments(
    df: pd.DataFrame,
    threshold: float = IDLE_SPEED_THRESHOLD,
    min_length: int = MIN_SEGMENT_LENGTH,
    min_running: int = MIN_RUNNING_POINTS,
) -> list[pd.DataFrame]:
    """把整段行程切成若干运动学片段。

    返回：片段列表，每个元素是原始数据的一个连续切片（已 reset_index）。
    """
    df = df.reset_index(drop=True)
    is_idle = _mark_idle(df, threshold)
    idle_starts = _find_idle_starts(is_idle)

    segments: list[pd.DataFrame] = []
    # 相邻两个怠速段起点之间，构成一个候选片段
    for i in range(len(idle_starts) - 1):
        start = idle_starts[i]
        end = idle_starts[i + 1]  # 不含
        seg = df.iloc[start:end]

        # 过滤：太短的丢弃
        if len(seg) < min_length:
            continue
        # 过滤：全程几乎没动（非怠速点太少）的丢弃
        running_points = (seg["speed"] >= threshold).sum()
        if running_points < min_running:
            continue

        segments.append(seg.reset_index(drop=True))

    return segments


# ======================================================================
# 阶段3：片段特征工程
# ----------------------------------------------------------------------
# 对每个运动学片段，计算一组刻画其"行驶模式"的特征。
# 这些是汽车行驶工况研究中的标准运动学特征参数。
#
# 特征清单（共 14 个）：
#     时间结构类：
#         duration        片段总时长(s)
#         idle_ratio      怠速时间占比（speed<阈值）
#         accel_ratio     加速时间占比（acc > 0.1 m/s^2）
#         decel_ratio     减速时间占比（acc < -0.1 m/s^2）
#         cruise_ratio    匀速时间占比（其余，运动中近似匀速）
#     速度类：
#         v_mean          平均车速(km/h)，含怠速
#         v_mean_run      运行平均车速(km/h)，仅非怠速点
#         v_max           最高车速(km/h)
#         v_std           车速标准差
#     加速度类：
#         a_mean_pos      平均加速度（仅加速段, m/s^2）
#         a_mean_neg      平均减速度（仅减速段, m/s^2）
#         a_max           最大加速度
#         a_std           加速度标准差
#         v_times_a_max   最大比功率近似 (v*a 的最大值)，刻画激烈程度
# ======================================================================

IDLE_THRESHOLD = 1.0       # km/h
ACC_DEADZONE = 0.1         # m/s^2，|acc|<此值视为匀速


def segment_features(seg: pd.DataFrame) -> dict[str, float]:
    """计算单个片段的特征字典。"""
    speed = seg["speed"].to_numpy()          # km/h
    acc = seg["acc"].to_numpy()              # m/s^2
    n = len(seg)

    is_idle = speed < IDLE_THRESHOLD
    is_accel = acc > ACC_DEADZONE
    is_decel = acc < -ACC_DEADZONE
    is_cruise = (~is_idle) & (~is_accel) & (~is_decel)

    run_mask = ~is_idle                       # 运行（非怠速）点
    speed_ms = speed / 3.6                     # 转 m/s 用于比功率

    feat = {
        # 时间结构
        "duration": float(n),
        "idle_ratio": float(is_idle.mean()),
        "accel_ratio": float(is_accel.mean()),
        "decel_ratio": float(is_decel.mean()),
        "cruise_ratio": float(is_cruise.mean()),
        # 速度
        "v_mean": float(speed.mean()),
        "v_mean_run": float(speed[run_mask].mean()) if run_mask.any() else 0.0,
        "v_max": float(speed.max()),
        "v_std": float(speed.std()),
        # 加速度
        "a_mean_pos": float(acc[is_accel].mean()) if is_accel.any() else 0.0,
        "a_mean_neg": float(acc[is_decel].mean()) if is_decel.any() else 0.0,
        "a_max": float(acc.max()),
        "a_std": float(acc.std()),
        # 比功率近似（激烈程度）：v(m/s) * a(m/s^2) 的最大正值
        "v_times_a_max": float(np.max(speed_ms * acc)),
    }
    return feat


def build_feature_matrix(segments: list[pd.DataFrame]) -> pd.DataFrame:
    """对所有片段批量提特征，返回特征矩阵 DataFrame。

    每行一个片段，每列一个特征。
    """
    rows = [segment_features(seg) for seg in segments]
    fm = pd.DataFrame(rows)
    fm.index.name = "segment_id"
    return fm


# 用于聚类的特征列（排除 duration 这类绝对量纲，保留模式刻画类）
# duration 保留——不同工况片段长度本身有区分意义，标准化后无量纲问题
FEATURE_COLUMNS = [
    "idle_ratio", "accel_ratio", "decel_ratio", "cruise_ratio",
    "v_mean", "v_mean_run", "v_max", "v_std",
    "a_mean_pos", "a_mean_neg", "a_max", "a_std",
    "v_times_a_max",
]


# ======================================================================
# 阶段4：标准化 -> PCA 降维 -> 选 K -> K-Means 聚类 -> 评估
# ----------------------------------------------------------------------
# 流程：
#     1. 标准化：StandardScaler，消除量纲差异（聚类对量纲敏感）。
#     2. PCA 降维：把十几个相关特征压到少数主成分，
#        既去冗余，又便于二维可视化。保留累计方差>=85% 的主成分。
#     3. 选 K：肘部法（inertia）+ 轮廓系数（silhouette），综合判断。
#     4. K-Means 聚类：用选定的 K 拟合，random_state 固定保证可复现。
#     5. 评估：轮廓系数、CH 指数、各簇规模。
# ======================================================================

RANDOM_STATE = 42
PCA_VARIANCE_TARGET = 0.85


def standardize(fm: pd.DataFrame, columns: list[str]):
    """对选定特征列做标准化。返回 (标准化数组, scaler)。"""
    X = fm[columns].to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def reduce_pca(Xs: np.ndarray, variance_target: float = PCA_VARIANCE_TARGET):
    """PCA 降维，保留累计方差达标的主成分。返回 (降维数组, pca)。"""
    pca_full = PCA(random_state=RANDOM_STATE).fit(Xs)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cum, variance_target) + 1)
    n_comp = max(n_comp, 2)  # 至少2维，便于可视化
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    Xp = pca.fit_transform(Xs)
    return Xp, pca


def evaluate_k_range(X: np.ndarray, k_min: int = 2, k_max: int = 8) -> pd.DataFrame:
    """对一系列 K 值评估肘部 inertia 与轮廓系数，辅助选 K。"""
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
        })
    return pd.DataFrame(rows)


def run_kmeans(X: np.ndarray, k: int) -> tuple[np.ndarray, KMeans]:
    """用指定 K 拟合 K-Means，返回 (标签, 模型)。"""
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    return labels, km


def cluster_profile(fm: pd.DataFrame, labels: np.ndarray,
                    columns: list[str]) -> pd.DataFrame:
    """计算每个簇在各原始特征上的均值，用于解释每类代表什么工况。"""
    tmp = fm[columns].copy()
    tmp["cluster"] = labels
    profile = tmp.groupby("cluster").mean()
    profile["count"] = pd.Series(labels).value_counts().sort_index()
    return profile


# ======================================================================
# Streamlit 交互界面
# ======================================================================

# ---- 中文字体 ----
_PREFERRED_FONTS = [
    "Arial Unicode MS", "PingFang SC", "PingFang HK", "Heiti SC",
    "Heiti TC", "STHeiti", "Songti SC", "Hiragino Sans GB",
    "SimHei", "Microsoft YaHei",
]
_available = {f.name for f in font_manager.fontManager.ttflist}
_chosen = next((f for f in _PREFERRED_FONTS if f in _available), None)
if _chosen:
    matplotlib.rcParams["font.sans-serif"] = [_chosen]
matplotlib.rcParams["axes.unicode_minus"] = False

DATA_PATH = "data/roadfile.xlsx"


# ============ 数据加载（带缓存，避免每次交互都重读 Excel）============
@st.cache_data
def get_clean_data():
    """读取+清洗，结果缓存。8.6万行的 Excel 只在第一次读。"""
    return load_and_clean(DATA_PATH)


@st.cache_data
def get_segments_and_features(threshold: float, min_length: int):
    """切片段 + 提特征，按参数缓存。参数不变时直接复用结果。"""
    df = get_clean_data()
    segs = split_segments(df, threshold=threshold, min_length=min_length)
    fm = build_feature_matrix(segs)
    return segs, fm


# ============ 页面配置 ============
st.set_page_config(
    page_title="汽车行驶工况聚类分析",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 汽车行驶运动学片段 K-Means 聚类分析")
st.caption("模式识别与机器学习课程设计 · 项目三 · 交互式分析界面")

# ============ 侧边栏：参数 ============
st.sidebar.header("⚙️ 参数设置")

idle_threshold = st.sidebar.slider(
    "怠速阈值 (km/h)",
    min_value=0.0, max_value=5.0, value=1.0, step=0.5,
    help="车速低于此值视为怠速/停车。默认 1.0，容忍 GPS 抖动。",
)

min_seg_len = st.sidebar.slider(
    "最小片段长度 (秒)",
    min_value=5, max_value=60, value=20, step=5,
    help="短于此时长的片段视为噪声丢弃。",
)

st.sidebar.divider()

auto_k = st.sidebar.checkbox(
    "自动选 K（按轮廓系数最优）", value=True,
)

manual_k = st.sidebar.slider(
    "手动指定 K", min_value=2, max_value=8, value=3, step=1,
    disabled=auto_k,
    help="取消上方勾选后可手动调节聚类数。",
)

st.sidebar.divider()
st.sidebar.markdown(
    "**当前定义**\n\n"
    "- 运动学片段：怠速起点 → 下一怠速起点\n"
    "- 怠速判据：车速 < 阈值\n"
    f"- 字体：{_chosen or '未找到中文字体'}"
)

# ============ 主区域：四个标签页 ============
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 数据概览", "✂️ 片段切分", "🔍 选 K 分析", "🎯 聚类结果"]
)

# ---- 加载数据（缓存）----
with st.spinner("正在读取数据..."):
    df = get_clean_data()
    segs, fm = get_segments_and_features(idle_threshold, min_seg_len)


# ========== Tab 1: 数据概览 ==========
with tab1:
    st.subheader("原始数据概览")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总采样点", f"{len(df):,}")
    c2.metric("采样时长", f"{len(df)/3600:.1f} 小时")
    c3.metric("怠速点占比", f"{(df['speed'] < idle_threshold).mean():.1%}")
    c4.metric("最高车速", f"{df['speed'].max():.1f} km/h")

    st.markdown("#### 车速曲线（前 3000 秒）")
    fig, ax = plt.subplots(figsize=(11, 3.5))
    window = df.iloc[:3000]
    ax.plot(window.index, window["speed"].to_numpy(),
            linewidth=0.8, color="steelblue")
    ax.axhline(idle_threshold, color="red", linestyle="--",
               linewidth=0.8, label=f"怠速阈值 {idle_threshold}")
    ax.fill_between(window.index, 0, window["speed"].to_numpy(),
                    where=(window["speed"] < idle_threshold).to_numpy(),
                    color="orange", alpha=0.3, label="怠速")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("车速 (km/h)")
    ax.legend(loc="upper right")
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("查看原始数据前 20 行"):
        st.dataframe(df.head(20))


# ========== Tab 2: 片段切分 ==========
with tab2:
    st.subheader("运动学片段切分结果")

    lengths = [len(s) for s in segs]
    c1, c2, c3 = st.columns(3)
    c1.metric("片段总数", len(segs))
    c2.metric("平均时长", f"{np.mean(lengths):.0f} s" if lengths else "—")
    c3.metric("行程覆盖率",
              f"{sum(lengths)/len(df):.1%}" if lengths else "—")

    if lengths:
        st.markdown("#### 片段时长分布")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(lengths, bins=30, color="mediumseagreen",
                edgecolor="white")
        ax.axvline(np.mean(lengths), color="red", linestyle="--",
                   label=f"平均 {np.mean(lengths):.0f}s")
        ax.set_xlabel("片段时长 (s)")
        ax.set_ylabel("片段数量")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        seg_idx = st.number_input(
            "查看第几个片段的车速曲线", min_value=0,
            max_value=len(segs) - 1, value=0, step=1,
        )
        seg = segs[int(seg_idx)]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(seg["speed"].to_numpy(), color="steelblue")
        ax.set_title(f"片段 #{int(seg_idx)}（{len(seg)} 秒）")
        ax.set_xlabel("片段内时间 (s)")
        ax.set_ylabel("车速 (km/h)")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("当前参数下没有切出任何片段，请调小最小片段长度。")


# ========== Tab 3: 选 K 分析 ==========
with tab3:
    st.subheader("聚类数 K 的选择")

    if len(fm) < 8:
        st.warning("片段太少，无法可靠地评估 K。请调整切分参数。")
    else:
        Xs, _ = standardize(fm, FEATURE_COLUMNS)
        Xp, pca = reduce_pca(Xs)
        st.info(f"PCA 降维：{Xs.shape[1]} 维 → {Xp.shape[1]} 维"
                f"（累计方差 {pca.explained_variance_ratio_.sum():.1%}）")

        scan = evaluate_k_range(Xp)
        best_k = int(scan.loc[scan["silhouette"].idxmax(), "k"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(scan["k"], scan["inertia"], "o-", color="steelblue")
        ax1.set_title("肘部法 (Elbow)")
        ax1.set_xlabel("K")
        ax1.set_ylabel("簇内平方和")
        ax2.plot(scan["k"], scan["silhouette"], "o-", color="darkorange")
        ax2.axvline(best_k, color="red", linestyle="--",
                    label=f"最优 K={best_k}")
        ax2.set_title("轮廓系数 (Silhouette)")
        ax2.set_xlabel("K")
        ax2.set_ylabel("轮廓系数")
        ax2.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### 各 K 值评估指标")
        st.dataframe(scan.round(4), use_container_width=True)


# ========== Tab 4: 聚类结果 ==========
with tab4:
    st.subheader("K-Means 聚类结果")

    if len(fm) < 8:
        st.warning("片段太少，无法聚类。请调整切分参数。")
    else:
        Xs, _ = standardize(fm, FEATURE_COLUMNS)
        Xp, pca = reduce_pca(Xs)
        scan = evaluate_k_range(Xp)
        best_k = int(scan.loc[scan["silhouette"].idxmax(), "k"])

        k = best_k if auto_k else manual_k
        st.success(f"使用 K = {k}"
                   f"（{'自动选择' if auto_k else '手动指定'}）")

        labels, km = run_kmeans(Xp, k)
        profile = cluster_profile(fm, labels, FEATURE_COLUMNS)

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("#### PCA 投影散点图")
            fig, ax = plt.subplots(figsize=(7, 5.5))
            sc = ax.scatter(Xp[:, 0], Xp[:, 1], c=labels, cmap="viridis",
                            s=40, alpha=0.7, edgecolors="white",
                            linewidths=0.5)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            legend = ax.legend(*sc.legend_elements(), title="簇")
            ax.add_artist(legend)
            st.pyplot(fig)
            plt.close(fig)

        with col_right:
            st.markdown("#### 各簇规模")
            counts = pd.Series(labels).value_counts().sort_index()
            for c in counts.index:
                v = profile.loc[c, "v_mean"]
                idle = profile.loc[c, "idle_ratio"]
                if v < profile["v_mean"].quantile(0.34) or \
                        idle > profile["idle_ratio"].quantile(0.66):
                    tag = "市区拥堵"
                elif v > profile["v_mean"].quantile(0.66) and \
                        idle < profile["idle_ratio"].quantile(0.34):
                    tag = "高速畅通"
                else:
                    tag = "城郊过渡"
                st.metric(
                    f"簇 {c} · {tag}",
                    f"{int(counts[c])} 个片段",
                    f"均速 {v:.1f} km/h | 怠速 {idle:.0%}",
                )

        st.markdown("#### 各簇特征画像")
        st.dataframe(profile.round(2), use_container_width=True)

        # 下载结果
        summary = fm.copy()
        summary["cluster"] = labels
        csv = summary.to_csv(encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "⬇️ 下载片段聚类结果 CSV",
            data=csv,
            file_name="segment_clusters.csv",
            mime="text/csv",
        )

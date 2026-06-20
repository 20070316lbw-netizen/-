"""Streamlit 交互界面：汽车行驶运动学片段 K-Means 聚类。

运行： uv run streamlit run app.py
然后浏览器会自动打开 http://localhost:8501

界面结构：
    侧边栏：参数调节（怠速阈值、最小片段长度、K 值、是否自动选 K）
    主区域：四个标签页
        1. 数据概览   —— 原始数据基本信息 + 车速曲线
        2. 片段切分   —— 切出的片段数、时长分布
        3. 选 K 分析  —— 肘部法 + 轮廓系数
        4. 聚类结果   —— 散点图、各簇画像、工况解释
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# 让 src 可被导入
sys.path.insert(0, str(Path(__file__).parent / "src"))

from load_data import load_and_clean          # noqa: E402
from segment import split_segments            # noqa: E402
from features import (                          # noqa: E402
    FEATURE_COLUMNS,
    build_feature_matrix,
)
from cluster import (                           # noqa: E402
    cluster_profile,
    evaluate_k_range,
    reduce_pca,
    run_kmeans,
    standardize,
)

# ---- 中文字体 ----
from matplotlib import font_manager  # noqa: E402

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

"""主流程：一键跑完整个分析并输出所有图表与结果。

运行： uv run python run.py
输出： outputs/ 下的图表 PNG + 结果 CSV，终端打印关键统计。

流程对应实验步骤：
    1. 读取+清洗      (load_data)
    2. 切运动学片段    (segment)
    3. 片段特征工程    (features)
    4. 标准化+PCA+选K+KMeans+评估  (cluster)
    5. 可视化+结果解释 (本文件)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# ---- 中文字体（matplotlib 默认不显示中文，做兼容处理）----
# 关键：必须从 matplotlib 实际注册的字体里挑，否则设了一个系统有、
# 但 matplotlib 没注册的名字（如某些机器上的 "PingFang SC"），
# 不会报错但会回退成方块。这里先取交集再选。
from matplotlib import font_manager  # noqa: E402

_PREFERRED_FONTS = [
    "Arial Unicode MS",   # macOS 自带，覆盖中日韩，最稳
    "PingFang SC", "PingFang HK", "Heiti SC", "Heiti TC", "STHeiti",
    "Songti SC", "Hiragino Sans GB",
    "SimHei", "Microsoft YaHei",  # Windows 常见
]
_available = {f.name for f in font_manager.fontManager.ttflist}
_chosen = next((f for f in _PREFERRED_FONTS if f in _available), None)
if _chosen:
    matplotlib.rcParams["font.sans-serif"] = [_chosen]
    print(f"[字体] 使用中文字体: {_chosen}")
else:
    print("[字体] 警告：未找到 matplotlib 可用的中文字体，"
          "图中中文可能为方块。可手动安装字体或改用英文标签。")
matplotlib.rcParams["axes.unicode_minus"] = False

DATA_PATH = "data/roadfile.xlsx"
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def plot_speed_profile(df: pd.DataFrame, segs: list[pd.DataFrame]) -> None:
    """图1：原始车速曲线 + 片段切分示意（取前 2000 秒局部放大）。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

    ax1.plot(df["speed"].to_numpy(), linewidth=0.4, color="steelblue")
    ax1.axhline(1.0, color="red", linestyle="--", linewidth=0.8,
                label="怠速阈值 (1 km/h)")
    ax1.set_title("原始 GPS 车速曲线（全程）")
    ax1.set_xlabel("时间 (s)")
    ax1.set_ylabel("车速 (km/h)")
    ax1.legend(loc="upper right")

    window = df.iloc[:2000]
    ax2.plot(window.index, window["speed"].to_numpy(),
             linewidth=0.8, color="steelblue")
    ax2.axhline(1.0, color="red", linestyle="--", linewidth=0.8)
    ax2.fill_between(window.index, 0, window["speed"].to_numpy(),
                     where=(window["speed"] < 1.0).to_numpy(),
                     color="orange", alpha=0.3, label="怠速区间")
    ax2.set_title("车速曲线局部放大（前 2000 秒，橙色为怠速）")
    ax2.set_xlabel("时间 (s)")
    ax2.set_ylabel("车速 (km/h)")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT / "01_speed_profile.png", dpi=150)
    plt.close(fig)


def plot_segment_hist(segs: list[pd.DataFrame]) -> None:
    """图2：片段时长分布直方图。"""
    lengths = [len(s) for s in segs]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=30, color="mediumseagreen", edgecolor="white")
    ax.set_title(f"运动学片段时长分布（共 {len(segs)} 个片段）")
    ax.set_xlabel("片段时长 (s)")
    ax.set_ylabel("片段数量")
    ax.axvline(np.mean(lengths), color="red", linestyle="--",
               label=f"平均 {np.mean(lengths):.0f}s")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "02_segment_length_hist.png", dpi=150)
    plt.close(fig)


def plot_k_selection(scan: pd.DataFrame) -> None:
    """图3：选 K —— 肘部法 + 轮廓系数双子图。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(scan["k"], scan["inertia"], "o-", color="steelblue")
    ax1.set_title("肘部法 (Elbow Method)")
    ax1.set_xlabel("聚类数 K")
    ax1.set_ylabel("簇内平方和 (Inertia)")

    ax2.plot(scan["k"], scan["silhouette"], "o-", color="darkorange")
    best_k = int(scan.loc[scan["silhouette"].idxmax(), "k"])
    ax2.axvline(best_k, color="red", linestyle="--",
                label=f"最优 K = {best_k}")
    ax2.set_title("轮廓系数 (Silhouette Score)")
    ax2.set_xlabel("聚类数 K")
    ax2.set_ylabel("轮廓系数")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(OUT / "03_k_selection.png", dpi=150)
    plt.close(fig)


def plot_cluster_scatter(Xp: np.ndarray, labels: np.ndarray, k: int) -> None:
    """图4：PCA 前两主成分上的聚类散点图。"""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(Xp[:, 0], Xp[:, 1], c=labels,
                         cmap="viridis", s=40, alpha=0.7,
                         edgecolors="white", linewidths=0.5)
    ax.set_title(f"运动学片段聚类结果（K={k}，PCA 投影）")
    ax.set_xlabel("第一主成分 PC1")
    ax.set_ylabel("第二主成分 PC2")
    legend = ax.legend(*scatter.legend_elements(),
                       title="簇", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(OUT / "04_cluster_scatter.png", dpi=150)
    plt.close(fig)


def plot_cluster_radar(profile: pd.DataFrame, k: int) -> None:
    """图5：各簇特征雷达图（标准化后对比，看每类的"性格"）。"""
    radar_feats = ["idle_ratio", "accel_ratio", "decel_ratio",
                   "cruise_ratio", "v_mean", "v_max", "a_std"]
    sub = profile[radar_feats].copy()
    # 各特征 min-max 归一到 [0,1] 便于雷达对比
    norm = (sub - sub.min()) / (sub.max() - sub.min() + 1e-9)

    angles = np.linspace(0, 2 * np.pi, len(radar_feats), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    cmap = plt.get_cmap("viridis", k)
    for c in range(k):
        vals = norm.loc[c].to_numpy()
        vals = np.concatenate([vals, vals[:1]])
        ax.plot(angles, vals, "o-", linewidth=1.5,
                color=cmap(c), label=f"簇 {c}")
        ax.fill(angles, vals, color=cmap(c), alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_feats)
    ax.set_title(f"各簇运动学特征对比雷达图（K={k}）", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    fig.savefig(OUT / "05_cluster_radar.png", dpi=150)
    plt.close(fig)


def interpret_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """根据各簇画像，启发式地给出工况标签（市区/郊区/高速等）。"""
    labels_map = {}
    v_means = profile["v_mean"]
    idle = profile["idle_ratio"]
    for c in profile.index:
        v = v_means[c]
        i = idle[c]
        if v < v_means.quantile(0.34) or i > idle.quantile(0.66):
            tag = "市区拥堵工况（低速、高怠速比）"
        elif v > v_means.quantile(0.66) and i < idle.quantile(0.34):
            tag = "高速/畅通工况（高速、低怠速比）"
        else:
            tag = "城郊过渡工况（中速）"
        labels_map[int(c)] = tag
    return labels_map


def main() -> None:
    print("=" * 60)
    print("汽车行驶运动学片段 K-Means 聚类分析")
    print("=" * 60)

    # 阶段1
    print("\n[1/5] 读取与清洗数据 ...")
    df = load_and_clean(DATA_PATH)
    print(f"      数据规模: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"      怠速点占比: {(df['speed'] < 1).mean():.1%}")

    # 阶段2
    print("\n[2/5] 切分运动学片段 ...")
    segs = split_segments(df)
    lengths = [len(s) for s in segs]
    print(f"      共切出 {len(segs)} 个片段，"
          f"平均时长 {np.mean(lengths):.0f}s，"
          f"覆盖 {sum(lengths)/len(df):.1%} 的行程")

    # 阶段3
    print("\n[3/5] 提取片段特征 ...")
    fm = build_feature_matrix(segs)
    print(f"      特征矩阵: {fm.shape[0]} 片段 x {fm.shape[1]} 特征")
    fm.to_csv(OUT / "feature_matrix.csv", encoding="utf-8-sig")

    # 阶段4
    print("\n[4/5] 标准化 -> PCA -> 选 K -> 聚类 ...")
    Xs, _ = standardize(fm, FEATURE_COLUMNS)
    Xp, pca = reduce_pca(Xs)
    print(f"      PCA: {Xs.shape[1]} 维 -> {Xp.shape[1]} 维 "
          f"(累计方差 {pca.explained_variance_ratio_.sum():.1%})")
    scan = evaluate_k_range(Xp)
    scan.to_csv(OUT / "k_selection.csv", index=False)
    best_k = int(scan.loc[scan["silhouette"].idxmax(), "k"])
    print(f"      轮廓系数最优 K = {best_k}")
    labels, km = run_kmeans(Xp, best_k)
    profile = cluster_profile(fm, labels, FEATURE_COLUMNS)
    profile.to_csv(OUT / "cluster_profile.csv", encoding="utf-8-sig")

    # 阶段5：可视化
    print("\n[5/5] 生成图表 ...")
    plot_speed_profile(df, segs)
    plot_segment_hist(segs)
    plot_k_selection(scan)
    plot_cluster_scatter(Xp, labels, best_k)
    plot_cluster_radar(profile, best_k)

    # 工况解释
    interp = interpret_clusters(profile)
    print("\n" + "=" * 60)
    print("聚类结果解释（各簇代表的行驶工况）:")
    print("=" * 60)
    for c in sorted(interp):
        cnt = int((labels == c).sum())
        print(f"  簇 {c} （{cnt} 个片段）: {interp[c]}")
        print(f"        平均车速 {profile.loc[c, 'v_mean']:.1f} km/h, "
              f"怠速比 {profile.loc[c, 'idle_ratio']:.1%}, "
              f"最高车速 {profile.loc[c, 'v_max']:.1f} km/h")

    # 保存带簇标签的片段汇总
    summary = fm.copy()
    summary["cluster"] = labels
    summary["工况"] = [interp[int(c)] for c in labels]
    summary.to_csv(OUT / "segment_clusters.csv", encoding="utf-8-sig")

    print(f"\n完成！所有图表与结果已保存到 {OUT}/ 目录。")


if __name__ == "__main__":
    main()

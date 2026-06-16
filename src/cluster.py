"""阶段4：标准化 -> PCA 降维 -> 选 K -> K-Means 聚类 -> 评估。

流程：
    1. 标准化：StandardScaler，消除量纲差异（聚类对量纲敏感）。
    2. PCA 降维：把十几个相关特征压到少数主成分，
       既去冗余，又便于二维可视化。保留累计方差>=85% 的主成分。
    3. 选 K：肘部法（inertia）+ 轮廓系数（silhouette），综合判断。
    4. K-Means 聚类：用选定的 K 拟合，random_state 固定保证可复现。
    5. 评估：轮廓系数、CH 指数、各簇规模。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

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


if __name__ == "__main__":
    from load_data import load_and_clean
    from segment import split_segments
    from features import FEATURE_COLUMNS, build_feature_matrix

    df = load_and_clean("data/roadfile.xlsx")
    segs = split_segments(df)
    fm = build_feature_matrix(segs)

    Xs, _ = standardize(fm, FEATURE_COLUMNS)
    Xp, pca = reduce_pca(Xs)
    print(f"PCA: {Xs.shape[1]} 维 -> {Xp.shape[1]} 维, "
          f"累计方差 {pca.explained_variance_ratio_.sum():.1%}")

    scan = evaluate_k_range(Xp)
    print("\n选 K 评估:")
    print(scan.round(3).to_string(index=False))

    best_k = int(scan.loc[scan["silhouette"].idxmax(), "k"])
    print(f"\n轮廓系数最优 K = {best_k}")
    labels, km = run_kmeans(Xp, best_k)
    print("\n各簇画像（原始特征均值）:")
    print(cluster_profile(fm, labels, FEATURE_COLUMNS).round(2).to_string())

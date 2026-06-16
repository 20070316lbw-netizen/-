"""阶段3：片段特征工程。

对每个运动学片段，计算一组刻画其"行驶模式"的特征。
这些是汽车行驶工况研究中的标准运动学特征参数。

特征清单（共 14 个）：
    时间结构类：
        duration        片段总时长(s)
        idle_ratio      怠速时间占比（speed<阈值）
        accel_ratio     加速时间占比（acc > 0.1 m/s^2）
        decel_ratio     减速时间占比（acc < -0.1 m/s^2）
        cruise_ratio    匀速时间占比（其余，运动中近似匀速）
    速度类：
        v_mean          平均车速(km/h)，含怠速
        v_mean_run      运行平均车速(km/h)，仅非怠速点
        v_max           最高车速(km/h)
        v_std           车速标准差
    加速度类：
        a_mean_pos      平均加速度（仅加速段, m/s^2）
        a_mean_neg      平均减速度（仅减速段, m/s^2）
        a_max           最大加速度
        a_std           加速度标准差
        v_times_a_max   最大比功率近似 (v*a 的最大值)，刻画激烈程度
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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


if __name__ == "__main__":
    from load_data import load_and_clean
    from segment import split_segments

    df = load_and_clean("data/roadfile.xlsx")
    segs = split_segments(df)
    fm = build_feature_matrix(segs)
    print(f"特征矩阵形状: {fm.shape}  (片段数 x 特征数)")
    print("\n特征列:", list(fm.columns))
    print("\n各特征统计:")
    print(fm.describe().round(3).to_string())

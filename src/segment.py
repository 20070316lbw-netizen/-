"""阶段2：切分运动学片段（kinematic segment）。

定义（方案A，汽车工程造行驶工况的标准定义）：
    一个运动学片段 = 从一次怠速的起点，到下一次怠速的起点。
    即「怠速 -> 加速 -> 巡航 -> 减速 -> 停车怠速」构成一个完整循环，
    对应车辆一次"起步到再次停车"的过程。

怠速判据：speed < IDLE_SPEED_THRESHOLD（km/h）。
    取 < 1 而非 == 0，是为了容忍 GPS 车速的零点抖动。

切分逻辑：
    1. 给每个采样点打标记：是否怠速（speed < 阈值）。
    2. 找出所有"怠速段"（连续的怠速点构成一段）。
    3. 每个运动学片段从一个怠速段的【起点】开始，
       到下一个怠速段的【起点】之前结束。
    4. 过滤掉过短的片段（噪声）和全程怠速的片段（车没动）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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


if __name__ == "__main__":
    from load_data import load_and_clean

    df = load_and_clean("data/roadfile.xlsx")
    segs = split_segments(df)
    print(f"共切出 {len(segs)} 个运动学片段")
    lengths = [len(s) for s in segs]
    if lengths:
        print(f"片段时长(秒): 最短={min(lengths)}, 最长={max(lengths)}, "
              f"平均={np.mean(lengths):.1f}, 中位={np.median(lengths):.0f}")
        print(f"总覆盖时长: {sum(lengths)} 秒 / 原始 {len(df)} 秒 "
              f"({sum(lengths)/len(df):.1%})")

"""阶段1：读取与清洗原始行车数据。

数据为 1Hz 等间隔采样，已确认无缺失值。本模块负责：
- 读取 Excel 原始数据
- 规范列名（中文 -> 英文，便于后续编程）
- 基础清洗：去除车速异常值、确保数值类型
- 计算逐秒加速度（由车速差分得到，单位 m/s^2）
"""

from __future__ import annotations

import pandas as pd

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


if __name__ == "__main__":
    df = load_and_clean("data/roadfile.xlsx")
    print("形状:", df.shape)
    print("\n规范后列名:", list(df.columns))
    print("\n车速统计:")
    print(df["speed"].describe())
    print("\n加速度统计 (m/s^2):")
    print(df["acc"].describe())
    print("\n怠速点(speed<1)占比: {:.1%}".format((df["speed"] < 1).mean()))

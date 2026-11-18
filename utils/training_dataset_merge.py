"""读取训练文件，并按完全相同的字段名纵向拼接。"""

import os

import pandas as pd

from utils.data_column_scan import read_csv_with_fallback, scan_dataframe_columns
from utils.prediction_result_io import load_mat_dataframe


def read_training_file(path):
    """使用训练界面已支持的格式读取一个文件。"""
    if not os.path.isfile(path):
        raise ValueError("文件不存在")

    # Excel 先保留原表头行，避免 pandas 自动改写重复字段名。
    extension = os.path.splitext(path)[1].lower()
    if extension in (".xls", ".xlsx"):
        raw = pd.read_excel(path, header=None)
        if raw.empty:
            raise ValueError("文件没有表头")
        header = raw.iloc[0].tolist()
        data = raw.iloc[1:].copy().reset_index(drop=True)
        data.columns = header
    elif extension == ".mat":
        data = load_mat_dataframe(path)
    elif extension == ".csv":
        data = read_csv_with_fallback(path)
    else:
        raise ValueError("仅支持 CSV、xls、xlsx 和 MAT 文件")

    # 字段名保持原样；重复或空白表头不能可靠地按名称对齐。
    names = list(data.columns)
    if any(pd.isna(name) or not str(name).strip() or str(name).startswith("Unnamed:") for name in names):
        raise ValueError("存在空白字段名")
    names = [str(name) for name in names]
    repeated = list(dict.fromkeys(name for name in names if names.count(name) > 1))
    if repeated:
        raise ValueError("存在重复字段名：" + "、".join(repeated))
    if data.empty or not names:
        raise ValueError("文件没有可用的数据行或字段")
    data.columns = names
    return data


def merge_training_frames(frames):
    """按文件列表顺序拼接行，字段按首次出现顺序排列。"""
    if not frames:
        raise ValueError("请至少添加一个训练文件")

    # 第一份文件确定初始列序；后续只追加首次出现的新字段。
    columns = list(frames[0].columns)
    for frame in frames[1:]:
        columns.extend(name for name in frame.columns if name not in columns)

    # pandas 按列名对齐；某文件没有的字段自然保留为 NaN。
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.reindex(columns=columns)

    # 比较各文件的有效内容类型，保留原值并提示跨文件类型混合。
    types_by_name = {}
    for frame in frames:
        for entry in scan_dataframe_columns(frame):
            if entry["type"] != "无法识别":
                types_by_name.setdefault(entry["name"], set()).add(entry["type"])
    mixed = [name for name in columns if len(types_by_name.get(name, ())) > 1]
    return merged, mixed

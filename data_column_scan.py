"""CSV 读取与后续列扫描共用的辅助函数。"""

import csv

import pandas as pd


def read_utf8_csv(file_path):
    """读取 UTF-8 CSV，并在重复列名被自动改名之前报错。"""
    try:
        # 先读取原始表头；utf-8-sig 兼容普通 UTF-8 和带 BOM 的 UTF-8。
        with open(file_path, "r", encoding="utf-8-sig", newline="") as source:
            header = next((row for row in csv.reader(source) if row), None)

        if header is None:
            raise ValueError("CSV 文件没有表头")

        # 在读取数据前检查原始表头，避免重复列名被自动重命名。
        seen = set()
        repeated = []
        for name in header:
            if name in seen and name not in repeated:
                repeated.append(name)
            seen.add(name)

        if repeated:
            raise ValueError(f"CSV 表头存在重复列名：{', '.join(repeated)}")

        data = pd.read_csv(file_path, encoding="utf-8-sig")

    except UnicodeError as exc:
        raise ValueError(
            f"CSV 解码失败：{file_path}；请使用 UTF-8 编码。原因：{exc}"
        ) from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"CSV 格式错误：{file_path}；原因：{exc}") from exc

    if len(data) == 0:
        raise ValueError("CSV 文件只有表头，没有数据行")

    return data


def scan_dataframe_columns(data):
    """按原列序返回列名、内容类型和缺失数量，不修改原数据。"""
    result = []

    for name in data.columns:
        column = data[name]

        # 将 NaN、空字符串和只有空格的字符串计为缺失。
        missing = column.isna() | column.map(
            lambda value: isinstance(value, str) and not value.strip()
        )
        present = column[~missing]

        if present.empty:
            column_type = "无法识别"
        else:
            # 仅用临时转换结果判断类型，不改动原始列。
            numeric = pd.to_numeric(
                present.astype(str).str.strip(), errors="coerce"
            )
            column_type = "数值" if numeric.notna().all() else "字符串"

        result.append({
            "name": name,
            "type": column_type,
            "missing_count": int(missing.sum()),
        })

    return result

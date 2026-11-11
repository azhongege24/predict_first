"""CSV 读取与后续列扫描共用的辅助函数。"""

import csv

import pandas as pd


def read_csv_with_fallback(file_path):
    """优先读取 UTF-8 CSV，解码失败时尝试 GB18030。"""
    last_decode_error = None

    # utf-8-sig 同时读取普通 UTF-8 和带 BOM 的 UTF-8；GB18030 兼容 GBK。
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            # 表头和数据使用同一种编码，避免列名与数据内容解释不一致。
            with open(file_path, "r", encoding=encoding, newline="") as source:
                header = next((row for row in csv.reader(source) if row), None)

            if header is None:
                raise ValueError("CSV 文件没有表头")

            # 先检查原始表头，避免 pandas 将重复列名自动改写。
            seen = set()
            repeated = []
            for name in header:
                if name in seen and name not in repeated:
                    repeated.append(name)
                seen.add(name)

            if repeated:
                raise ValueError(f"CSV 表头存在重复列名：{', '.join(repeated)}")

            data = pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError as exc:
            # 只有解码失败才尝试下一种编码；文件结构错误应直接提示用户。
            last_decode_error = exc
            continue
        except pd.errors.ParserError as exc:
            raise ValueError(f"CSV 格式错误：{file_path}；原因：{exc}") from exc

        if len(data) == 0:
            raise ValueError("CSV 文件只有表头，没有数据行")

        # 两种编码的读取结果结构相同，供训练和预测继续使用。
        return data

    raise ValueError(
        f"CSV 解码失败：{file_path}；请使用 UTF-8、GBK 或 GB18030 编码。"
        f"原因：{last_decode_error}"
    ) from last_decode_error


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

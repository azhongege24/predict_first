"""预测结果表合并、保存及 MAT 文件读取。"""

import json
import os

import numpy as np
import pandas as pd
import scipy.io as sio


# MAT变量使用安全的英文名称，并通过附带的映射恢复原始表头。
MAT_EXPORT_VERSION_KEY = "vibration_prediction_format_version"
MAT_EXPORT_COLUMNS_KEY = "vibration_prediction_column_names_json"
MAT_EXPORT_COLUMN_PREFIX = "vibration_prediction_column_"
MAT_EXPORT_MISSING_PREFIX = "vibration_prediction_missing_"


def merge_prediction_results(original_data, predictions, output_columns, start, end):
    """保留完整输入表，并在右侧追加指定行范围内的预测结果。"""
    # 单输出模型可能返回一维数组；统一成样本数×输出数的二维形状。
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    # 确认模型返回的行列数与所选范围及输出列数量一致。
    expected_rows = end - start
    if predictions.ndim != 2:
        raise ValueError("预测结果必须是二维数据")
    if predictions.shape != (expected_rows, len(output_columns)):
        raise ValueError("预测结果的行数或列数与预测范围不一致")

    # 在副本中增加预测列，避免改写界面保留的原始数据。
    merged = original_data.copy()
    for output_index, output_name in enumerate(output_columns):
        # 若原数据已有预测列同名字段，则为新列添加递增后缀。
        base_name = f"预测_{output_name}"
        prediction_name = base_name
        suffix = 1
        while prediction_name in merged.columns:
            prediction_name = f"{base_name}_{suffix}"
            suffix += 1

        # 全文件先填空值，再按位置写入本次实际预测的行。
        prediction_values = np.full(len(merged), np.nan, dtype=float)
        prediction_values[start:end] = predictions[:, output_index]
        merged[prediction_name] = prediction_values

    return merged


def dataframe_to_mat_export(dataframe):
    """把DataFrame转换为带列名映射和缺失标记的MAT变量字典。"""
    # 列名以转义JSON保存，避免中文列名直接成为MAT变量名。
    mat_data = {
        MAT_EXPORT_VERSION_KEY: np.array([[1]], dtype=np.uint8),
        MAT_EXPORT_COLUMNS_KEY: json.dumps(
            [str(column) for column in dataframe.columns], ensure_ascii=True
        ),
    }

    for column_index, column_name in enumerate(dataframe.columns):
        series = dataframe[column_name]
        variable_suffix = f"{column_index:04d}"

        # 文本列保存为MAT单元格，缺失文本先用空值占位。
        if series.dtype == object:
            values = np.array(
                ["" if pd.isna(value) else value for value in series], dtype=object
            )
        else:
            values = series.to_numpy()
        mat_data[f"{MAT_EXPORT_COLUMN_PREFIX}{variable_suffix}"] = values

        # 单独记录缺失状态，读取时可区分NaN和真实空文本。
        mat_data[f"{MAT_EXPORT_MISSING_PREFIX}{variable_suffix}"] = (
            series.isna().to_numpy(dtype=np.uint8)
        )

    return mat_data


def dataframe_from_mat_export(mat_data):
    """读取本程序导出的MAT变量；遇到普通MAT格式时返回None。"""
    # 普通MAT文件交由旧格式读取分支处理。
    required_keys = (MAT_EXPORT_VERSION_KEY, MAT_EXPORT_COLUMNS_KEY)
    if not all(key in mat_data for key in required_keys):
        return None

    # 按保存顺序解析原始列名和数据变量。
    raw_names = np.asarray(mat_data[MAT_EXPORT_COLUMNS_KEY]).reshape(-1)
    names_json = "".join(str(value) for value in raw_names)
    column_names = json.loads(names_json)
    columns = {}
    for column_index, column_name in enumerate(column_names):
        variable_suffix = f"{column_index:04d}"
        values = np.asarray(
            mat_data[f"{MAT_EXPORT_COLUMN_PREFIX}{variable_suffix}"]
        ).reshape(-1)
        missing = np.asarray(
            mat_data[f"{MAT_EXPORT_MISSING_PREFIX}{variable_suffix}"]
        ).reshape(-1).astype(bool)

        # MAT单元格中的字符数组还原为文本，数值单元格保持数值。
        decoded_values = []
        for value in values:
            if isinstance(value, np.ndarray) and value.dtype.kind in ("U", "S"):
                value = "".join(str(part) for part in value.reshape(-1))
            elif isinstance(value, np.ndarray) and value.size == 1:
                value = value.reshape(-1)[0].item()
            decoded_values.append(value)

        # 依据缺失标记恢复NaN，保留空文本原值。
        columns[column_name] = [
            np.nan if is_missing else value
            for value, is_missing in zip(decoded_values, missing)
        ]

    return pd.DataFrame(columns)


def load_mat_dataframe(file_path):
    """读取本程序导出格式或项目原有布局的MAT文件。"""
    mat_data = sio.loadmat(file_path)
    loaded_data = dataframe_from_mat_export(mat_data)
    if loaded_data is not None:
        return loaded_data

    # 旧MAT文件的有效特征位于前置元数据之后，并沿用原列顺序调整方式。
    all_keys = list(mat_data.keys())
    feature_keys = all_keys[3:]
    data = pd.DataFrame({
        key: mat_data[key].squeeze()
        for key in feature_keys
    })
    columns = data.columns.tolist()
    return data[columns[1:] + [columns[0]]]


def save_prediction_dataframe(dataframe, file_path):
    """按文件扩展名保存预测结果表。"""
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".mat":
        # MAT使用安全变量名和列名映射保存完整数据表。
        sio.savemat(file_path, dataframe_to_mat_export(dataframe))
    elif extension in (".xls", ".xlsx"):
        # 数据行号不属于原始字段，保存表格时不额外写入索引列。
        dataframe.to_excel(file_path, index=False)
    else:
        # 带BOM的UTF-8便于常见表格软件正确显示中文列名。
        dataframe.to_csv(file_path, index=False, encoding="utf-8-sig")

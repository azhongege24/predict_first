"""生成输入特征与输出特征关系图的分页数据。"""

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 中文列名和类别使用系统常见中文字体，缺少首选字体时由 Matplotlib 回退。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def make_relationship_pages(input_data, y_true, y_pred, input_columns,
                            output_columns, column_types):
    """按输入与输出列的组合建立可按需绘制的页面描述。"""
    actual = np.asarray(y_true, dtype=object)
    predicted = np.asarray(y_pred, dtype=object)
    if actual.ndim == 1:
        actual = actual.reshape(-1, 1)
    if predicted.ndim == 1:
        predicted = predicted.reshape(-1, 1)

    # 校验行数和列数，保证每个点都能对应同一条输入记录。
    if len(input_data) != len(actual) or actual.shape != predicted.shape:
        raise ValueError("关系图输入、真实值和预测值的行列数不一致")
    if actual.shape[1] != len(output_columns):
        raise ValueError("关系图输出列名数量与数据列数不一致")

    pages = []
    for input_name in input_columns:
        if input_name not in input_data.columns:
            continue
        for output_index, output_name in enumerate(output_columns):
            pages.append({
                "kind": "relationship",
                "input_values": input_data[input_name].to_numpy(copy=True),
                "input_name": input_name,
                "input_type": column_types.get(input_name, "数值"),
                "actual": actual[:, output_index].copy(),
                "predicted": predicted[:, output_index].copy(),
                "output_name": output_name,
            })
    return pages


def render_relationship_page(page):
    """将一页关系图绘制成内存中的 PNG 数据。"""
    x_values = np.asarray(page["input_values"], dtype=object)
    actual = np.asarray(page["actual"], dtype=object)
    predicted = np.asarray(page["predicted"], dtype=object)

    # 只绘制有限数值输出，避免 NaN 或无穷值进入坐标轴。
    actual_numeric = np.full(len(actual), np.nan, dtype=float)
    predicted_numeric = np.full(len(predicted), np.nan, dtype=float)
    for index, value in enumerate(actual):
        try:
            actual_numeric[index] = float(value)
        except (TypeError, ValueError):
            pass
    for index, value in enumerate(predicted):
        try:
            predicted_numeric[index] = float(value)
        except (TypeError, ValueError):
            pass

    # 使用与样本序号图相同的画布尺寸和分辨率，保持翻页时视觉比例一致。
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    try:
        # 类别横轴保留原始文字；缺失输入作为独立类别显示。
        if page["input_type"] == "字符串":
            labels = ["缺失" if value is None or _is_missing(value)
                      else str(value) for value in x_values]
            categories = list(dict.fromkeys(labels))
            category_positions = {name: index for index, name in enumerate(categories)}
            x_plot = np.asarray([category_positions[name] for name in labels], dtype=float)
            # 轻微且固定的错位帮助分辨同一类别内重叠的数据点。
            offsets = (np.arange(len(x_plot)) % 7 - 3) * 0.025
            x_plot += offsets
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=30, ha="right")
            valid_x = np.ones(len(x_plot), dtype=bool)
            omitted = 0
        else:
            x_numeric = np.full(len(x_values), np.nan, dtype=float)
            for index, value in enumerate(x_values):
                try:
                    x_numeric[index] = float(value)
                except (TypeError, ValueError):
                    pass
            valid_x = np.isfinite(x_numeric)
            x_plot = x_numeric
            omitted = int((~valid_x).sum())

        valid_prediction = valid_x & np.isfinite(predicted_numeric)
        valid_actual = valid_x & np.isfinite(actual_numeric)
        # 同一输入记录的真实值和预测值之间用细线标出误差。
        valid_pair = valid_prediction & valid_actual
        for row in np.flatnonzero(valid_pair):
            ax.plot([x_plot[row], x_plot[row]],
                    [actual_numeric[row], predicted_numeric[row]],
                    color="#2F5597", linestyle="-", linewidth=2.5,
                    alpha=0.5, zorder=0)
        if valid_prediction.any():
            ax.scatter(x_plot[valid_prediction], predicted_numeric[valid_prediction],
                       c="r", marker="X", alpha=0.6, s=20, label="预测值")
        if valid_actual.any():
            ax.scatter(x_plot[valid_actual], actual_numeric[valid_actual],
                       c="b", marker="o", alpha=0.8, s=10, label="真实值")

        # 在图中说明数值输入缺失而未能定位到横轴的行数。
        suffix = f"；输入缺失或非数值，省略 {omitted} 行" if omitted else ""
        ax.set_title(f"{page['input_name']} 与 {page['output_name']} 的关系{suffix}", fontsize=10)
        ax.set_xlabel(page["input_name"], fontsize=10)
        ax.set_ylabel(page["output_name"], fontsize=10)
        ax.set_facecolor("#f8f9fa")
        ax.grid(linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if valid_prediction.any() or valid_actual.any():
            ax.legend(fontsize=10, loc="upper left")
        else:
            ax.text(0.5, 0.5, "没有可绘制的有效数据", ha="center", va="center",
                    transform=ax.transAxes)
        fig.tight_layout()

        # 图片只交给界面显示；内存流关闭后保留 PNG 字节供翻页复用。
        with BytesIO() as image_buffer:
            fig.savefig(image_buffer, format="png", bbox_inches="tight")
            return image_buffer.getvalue()
    finally:
        plt.close(fig)


def _is_missing(value):
    """识别 pandas、NumPy 和 Python 标量的缺失值。"""
    try:
        result = pd.isna(value)
        return bool(result) if np.isscalar(result) else bool(np.asarray(result).all())
    except (TypeError, ValueError):
        return False

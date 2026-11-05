"""准备普通回归训练使用的数值列，不修改上传后的原始数据。"""

import pandas as pd


def prepare_numeric_regression_data(
    data_train, data_test, input_columns, output_columns
):
    """用训练部分的有效数据填补输入，并返回实际删行数。"""
    train = data_train.copy(deep=True)
    test = data_test.copy(deep=True)

    # 只转换本次选中的输入和输出；附带的时间列保持原样。
    for name in dict.fromkeys(input_columns + output_columns):
        for frame in (train, test):
            # 空字符串和纯空格与文件中的 NaN 一样，作为缺失值处理。
            values = frame[name].map(
                lambda value: float("nan")
                if pd.isna(value)
                or (isinstance(value, str) and not value.strip())
                else value
            )
            try:
                # 数字文本转为数值；其他文字明确报错，避免悄悄变成缺失。
                frame[name] = pd.to_numeric(values, errors="raise").astype(float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"列“{name}”包含无法转换为数值的内容"
                ) from exc

    # 训练行没有所选输出的真实值时，不能用于拟合模型。
    missing_output = train[output_columns].isna().any(axis=1)
    removed_output_rows = int(missing_output.sum())
    train = train.loc[~missing_output].copy()
    if train.empty:
        raise ValueError(
            "训练范围内没有同时具备所选输出值的行，无法训练："
            + "、".join(output_columns)
        )

    # 测试输出的缺失值保留，预测所有行后只评价有真实值的行。

    input_medians = {}
    for name in input_columns:
        # 只从删除无效输出行后的训练部分学习中位数。
        if not train[name].notna().any():
            raise ValueError(
                f"输入列“{name}”在有效训练行中全部缺失，无法填补"
            )
        median = train[name].median()
        input_medians[name] = float(median)
        # 测试集复用训练中位数，不能参与规则计算。
        train[name] = train[name].fillna(median)
        test[name] = test[name].fillna(median)

    return train, test, input_medians, removed_output_rows

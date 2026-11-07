"""准备普通回归训练使用的数值列，不修改上传后的原始数据。"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


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


def _category_token(value):
    """把字符串输入转换为内部类别，同时区分空值与真实文字。"""
    # 空值和只含空格的文字使用专用标记；真实文字都加另一种前缀。
    # 因此用户实际填写的“缺失”不会与空值被当成同一类别。
    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return "M:"
    return "V:" + str(value)


def prepare_mixed_regression_data(
    data_train, data_test, input_columns, output_columns, types_by_name
):
    """按训练集学习数值填补和字符串编码，返回模型数据及规则。"""
    input_columns = list(input_columns)
    output_columns = list(output_columns)

    # 类型来自上传时的列扫描结果；只处理本次勾选的输入列。
    numeric_columns = [
        name for name in input_columns if types_by_name[name] == "数值"
    ]
    category_columns = [
        name for name in input_columns if types_by_name[name] == "字符串"
    ]
    if len(numeric_columns) + len(category_columns) != len(input_columns):
        raise ValueError("所选输入列存在无法识别的类型")
    if not category_columns:
        raise ValueError("混合输入处理需要至少一列字符串输入")

    # 复用现有数值流程：先删训练输出缺失行，再从剩余训练行
    # 计算数值输入的中位数，测试集只复用这些中位数。
    train, test, medians, removed_rows = prepare_numeric_regression_data(
        data_train, data_test, numeric_columns, output_columns
    )

    # 类别也只从有效训练行学习。保留原行索引，以便输出和时间列对齐。
    train_categories = pd.DataFrame(
        {name: train[name].map(_category_token) for name in category_columns},
        index=train.index,
    )
    test_categories = pd.DataFrame(
        {name: test[name].map(_category_token) for name in category_columns},
        index=test.index,
    )

    # 每列都预留空值类别，即使这次训练恰好没有缺失输入。
    # 显式指定类别顺序，保证训练集和测试集生成相同位置的特征。
    categories = [
        sorted(set(train_categories[name]) | {"M:"})
        for name in category_columns
    ]
    encoder = OneHotEncoder(
        categories=categories, sparse=False, handle_unknown="error"
    )
    encoder.fit(train_categories)

    # 空值的专用标记已预留；其他训练时未见过的类别不能随意编码。
    # 提示使用文件中的原始列名和值，供界面在下一步直接展示。
    for name, known in zip(category_columns, encoder.categories_):
        unknown = sorted(set(test_categories[name]) - set(known))
        if unknown:
            values = "、".join(token[2:] for token in unknown)
            raise ValueError(
                f"输入列“{name}”出现训练时未见过的类别：{values}"
            )

    encoded_train = encoder.transform(train_categories)
    encoded_test = encoder.transform(test_categories)

    # 内部特征名按原始输入列的位置编号，不依赖中文类别的拼写。
    # 若上传文件恰好使用了相同前缀，就增加下划线避开原始列名。
    reserved_names = set(train.columns) | set(test.columns)
    prefix = "__model_feature_"
    while any(name.startswith(prefix) for name in reserved_names):
        prefix = "_" + prefix

    # 删除模型不能直接读取的原始输入列；输出和附带时间列保留。
    # 原始上传表不在这里改写，train/test 也是数值函数创建的副本。
    train_result = train.drop(columns=input_columns).copy()
    test_result = test.drop(columns=input_columns).copy()
    feature_columns = []
    category_offsets = {}
    offset = 0
    for name, known in zip(category_columns, encoder.categories_):
        # 独热编码器把各字符串列依次拼接，此处记录每列的起始位置。
        category_offsets[name] = (offset, len(known))
        offset += len(known)

    # 为修改后的列重命名
    for position, name in enumerate(input_columns):
        if name in numeric_columns:
            feature_name = f"{prefix}{position}_numeric"
            train_result[feature_name] = train[name].to_numpy()
            test_result[feature_name] = test[name].to_numpy()
            feature_columns.append(feature_name)
        else:
            start, count = category_offsets[name]
            for category_position in range(count):
                # 一个中文类别对应一列 0/1；缺失类别也占固定的一列。
                feature_name = (
                    f"{prefix}{position}_category_{category_position}"
                )
                train_result[feature_name] = (
                    encoded_train[:, start + category_position]
                )
                test_result[feature_name] = (
                    encoded_test[:, start + category_position]
                )
                feature_columns.append(feature_name)

    # 无论训练和测试各出现哪些类别，模型输入列都按原选列顺序排列。
    remaining_columns = [
        name for name in train_result.columns if name not in feature_columns
    ]
    train_result = train_result[feature_columns + remaining_columns]
    test_result = test_result[feature_columns + remaining_columns]

    # 后续保存模型时必须保存这些规则，预测不能重新学习中位数或类别。
    rules = {
        "input_columns": input_columns,
        "output_columns": output_columns,
        "numeric_columns": numeric_columns,
        "category_columns": category_columns,
        "feature_columns": feature_columns,
        "input_medians": medians,
        "category_encoder": encoder,
    }
    return train_result, test_result, rules, removed_rows

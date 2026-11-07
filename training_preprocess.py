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


def _category_token(value, missing_token="M:", value_prefix="V:"):
    """把字符串输入转换为内部类别，同时区分空值与真实文字。"""
    # 空值和只含空格的文字使用专用标记；真实文字都加另一种前缀。
    # 因此用户实际填写的“缺失”不会与空值被当成同一类别。
    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return missing_token
    return value_prefix + str(value)


def prepare_mixed_regression_data(
    data_train, data_test, input_columns, output_columns, types_by_name
):
    """按训练集学习填补和编码，返回模型数据、规则及未知类别明细。"""
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
    # 训练时学到的类别表已经固定，测试数据不会参与拟合。
    # 测试集若出现新类别，编码器会给该字段的已知类别位全部填 0。
    # 这里不能使用 error，否则在弹窗询问用户之前就会中断。
    # 使用 sparse 以兼容项目当前的 scikit-learn 1.0.2。
    encoder = OneHotEncoder(
        categories=categories, sparse=False, handle_unknown="ignore"
    )
    encoder.fit(train_categories)

    # 单独收集新类别，不能因为编码器允许继续就略过对用户的提醒。
    # by_column 使用原始列名，便于界面直接指出是哪一列有问题。
    unknown_categories = {}
    # 同一测试行可能在多个输入列都含新类别，受影响总行数只算一次。
    unknown_rows = pd.Series(False, index=test_categories.index)
    for name, known in zip(category_columns, encoder.categories_):
        # 已知类别只来自有效训练行，外加预留的空值类别。
        known_tokens = set(known)
        # 统计测试段每个类别出现的次数，以供弹窗显示具体影响。
        counts = test_categories[name].value_counts()
        unseen_tokens = [
            token for token in counts.index if token not in known_tokens
        ]
        if not unseen_tokens:
            continue

        # 去掉内部前缀，只把文件中的真实类别文字交给界面。
        # 真正的空值使用已预留的 M:，不会进入未知类别列表。
        unknown_categories[name] = {
            token[2:]: int(counts[token]) for token in unseen_tokens
        }
        # 把本列受影响的行合并到总行掩码，不重复计算同一行。
        unknown_rows |= test_categories[name].isin(unseen_tokens)

    # 预处理只报告发现了什么；是否继续由训练入口的弹窗决定。
    unknown_details = {
        "by_column": unknown_categories,
        "affected_rows": int(unknown_rows.sum()),
    }

    # 训练和测试都使用同一个编码器，内部特征数量与顺序相同。
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
        # 预测时使用相同前缀，才能区分真实文字与缺失值。
        "missing_token": "M:",
        "value_prefix": "V:",
    }
    # 没有新类别时也返回空明细，调用方始终按相同的结构取结果。
    return train_result, test_result, rules, removed_rows, unknown_details


def prepare_prediction_features(data, metadata):
    """复用模型保存的填补、类别编码和缩放规则准备预测输入。"""
    # 只有保存了完整规则的新模型才能使用这条预测输入路径。
    # 版本不同意味着字段含义可能变化，不能猜测其处理方式。
    if metadata.get("format_version") != 2:
        raise ValueError("模型的预处理规则版本不受支持")
    # 规则说明如何处理原始列；特征列表说明模型最终读取的列序。
    rules = metadata.get("preprocessing_rules")
    feature_columns = metadata.get("model_input_columns")
    if not isinstance(rules, dict) or not feature_columns:
        raise ValueError("模型缺少输入预处理规则或特征顺序")
    # 保存文件中的两份特征顺序必须一致，避免把数值放错位置。
    if feature_columns != rules.get("feature_columns"):
        raise ValueError("模型保存的特征顺序不一致")

    # 按训练时的原始列顺序读取输入，不依赖预测文件的列顺序。
    # 输入被分成数值列和类别列；数值列还需要对应的填充值。
    input_columns = rules.get("input_columns")
    numeric_columns = rules.get("numeric_columns")
    category_columns = rules.get("category_columns")
    medians = rules.get("input_medians")
    # 缺少任何一组基础规则时，无法可靠地恢复模型输入。
    if not input_columns or numeric_columns is None or category_columns is None or not isinstance(medians, dict):
        raise ValueError("模型保存的输入列或填补规则不完整")
    # 每个原始输入只能出现一次，并且只能属于一种类型。
    if len(set(input_columns)) != len(input_columns) or set(input_columns) != set(numeric_columns) | set(category_columns) or set(numeric_columns) & set(category_columns):
        raise ValueError("模型保存的输入列类型不一致")
    # 在读取数据之前列出缺失的原始列名，便于用户修正文件。
    missing_columns = [name for name in input_columns if name not in data.columns]
    if missing_columns:
        raise ValueError("预测文件缺少模型所需输入列：" + "、".join(missing_columns))

    # 数字文本按数值读取；空白和 NaN 使用训练时保存的逐列中位数。
    numeric_values = {}
    for name in numeric_columns:
        # 每个数值输入都必须有训练时计算的中位数。
        if name not in medians:
            raise ValueError(f"模型缺少输入列“{name}”的填补值")
        # 把空格文本和真正的 NaN 统一视为缺失，其他内容保留原值。
        values = data[name].map(
            lambda value: float("nan")
            if pd.isna(value) or (isinstance(value, str) and not value.strip())
            else value
        )
        try:
            # 数字文本转为浮点数；只填当前输入列，不删除整行。
            numeric_values[name] = pd.to_numeric(values, errors="raise").astype(float).fillna(medians[name])
        except (TypeError, ValueError, OverflowError) as exc:
            # 非数字文字不能悄悄转成缺失，否则会掩盖文件内容错误。
            raise ValueError(f"输入列“{name}”包含无法转换为数值的内容") from exc

    # 字符串缺失和真实文字分别使用模型保存的标记，不重新学习类别。
    # 纯数值模型没有类别编码器，后面的类别分支会直接跳过。
    encoder = rules.get("category_encoder")
    encoded = None
    if category_columns:
        # 编码器保存的类别列数必须和原始字符串输入列数相同。
        if encoder is None or len(encoder.categories_) != len(category_columns):
            raise ValueError("模型保存的类别编码规则不完整")
        # 读取训练时使用的两种标记，区分空值与实际文字。
        missing_token = rules.get("missing_token")
        value_prefix = rules.get("value_prefix")
        if not missing_token or not value_prefix:
            raise ValueError("模型缺少字符串类别标记")
        # 保持原行索引；每个字符串值变成编码器认识的内部类别。
        category_data = pd.DataFrame(
            {
                name: data[name].map(
                    lambda value: _category_token(value, missing_token, value_prefix)
                )
                for name in category_columns
            },
            index=data.index,
        )

        # 编码器即使允许忽略新类别，也不能在预测时悄悄使用全零编码。
        for name, known in zip(category_columns, encoder.categories_):
            # 逐列比较本次值与训练时记录的类别集合。
            unknown = sorted(set(category_data[name]) - set(known))
            if unknown:
                # 提示使用原始列名和真实文字，不暴露内部类别前缀。
                values = "、".join(
                    f"“{value[len(value_prefix):]}”" for value in unknown
                )
                raise ValueError(f"输入列“{name}”出现训练时未见过的类别：{values}")
        # 类别全部已知后才使用保存的编码器转换，不调用 fit。
        encoded = encoder.transform(category_data)
    elif encoder is not None:
        # 没有类别输入却保存了编码器，说明模型规则前后矛盾。
        raise ValueError("模型保存的类别列和编码器不一致")

    # 一个原始输入列依次对应一个数值特征或该列的全部独热编码位。
    # 用原始行索引创建结果表，便于数值输入和类别编码逐行对齐。
    prepared = pd.DataFrame(index=data.index)
    # 两个位置分别跟踪模型特征列表和编码器输出数组的读取进度。
    feature_index = 0
    category_offset = 0
    category_positions = {name: position for position, name in enumerate(category_columns)}
    for name in input_columns:
        if name in numeric_values:
            # 数值输入占一个模型特征位；先防止损坏规则造成越界。
            if feature_index >= len(feature_columns):
                raise ValueError("模型保存的特征数量与输入规则不一致")
            # 使用模型保存的内部特征名放入已填补的数值。
            prepared[feature_columns[feature_index]] = numeric_values[name]
            feature_index += 1
        else:
            # 字符串输入占多个特征位，数量等于训练时该列的类别数。
            count = len(encoder.categories_[category_positions[name]])
            for position in range(count):
                # 每写入一个编码位，都检查模型特征列表是否足够长。
                if feature_index >= len(feature_columns):
                    raise ValueError("模型保存的特征数量与编码规则不一致")
                # 从编码器结果中取这一列的 0/1 值，顺序与训练时一致。
                prepared[feature_columns[feature_index]] = encoded[:, category_offset + position]
                feature_index += 1
            # 下一列字符串输入从编码器结果的下一个区段开始读取。
            category_offset += count
    # 所有输入写完后还剩下特征名，说明模型文件中的规则不匹配。
    if feature_index != len(feature_columns):
        raise ValueError("模型保存的特征数量与编码规则不一致")

    # 缩放器只能应用训练时拟合的参数，预测数据绝不参与重新拟合。
    # 按模型特征顺序取出二维数组，不把原始输出或无关列送给模型。
    features = prepared[feature_columns].to_numpy(dtype=float)
    scaler = metadata.get("scaler")
    if scaler is not None:
        # 使用保存的均值和方差；这里不能重新调用 fit_transform。
        features = scaler.transform(features)
    # 返回每行对应一个样本、每列对应一个训练特征的数值数组。
    return features

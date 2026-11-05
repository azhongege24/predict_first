"""按每个输出有真实值的测试行计算回归指标。"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def display_metric(value, digits=5):
    """没有可计算的指标时，在界面上显示清楚的原因。"""
    return f"{value:.{digits}f}" if np.isfinite(value) else "无有效真实值"


def evaluate_regression(y_test, y_pred, db_factor=10):
    """保留全部预测行；每个输出仅用其真实值非空的行评价。"""
    actual = np.asarray(y_test, dtype=float).reshape(len(y_test), -1)
    predicted = np.asarray(y_pred, dtype=float).reshape(len(y_test), -1)
    if actual.shape != predicted.shape:
        raise ValueError("测试真实值与预测值的行列数不一致")

    mse_list, rmse_list, mae_list, r2_list = [], [], [], []
    ratio_list, deviation_list, valid_counts = [], [], []
    for column in range(actual.shape[1]):
        valid = np.isfinite(actual[:, column])
        count = int(valid.sum())
        valid_counts.append(count)
        if count == 0:
            mse_list.append(float("nan"))
            rmse_list.append(float("nan"))
            mae_list.append(float("nan"))
            r2_list.append(float("nan"))
            ratio_list.append(float("nan"))
            deviation_list.append(float("nan"))
            continue

        truth = actual[valid, column]
        estimate = predicted[valid, column]
        mse = float(mean_squared_error(truth, estimate))
        mse_list.append(mse)
        rmse_list.append(float(np.sqrt(mse)))
        mae_list.append(float(mean_absolute_error(truth, estimate)))
        # R² 至少需要两条真实值；其余误差指标一条即可计算。
        r2_list.append(float(r2_score(truth, estimate)) if count >= 2 else float("nan"))
        epsilon = np.maximum(np.abs(truth) * 1e-12, 1e-8)
        with np.errstate(divide="ignore", invalid="ignore"):
            db_diff = db_factor * np.log10(
                (np.abs(estimate) + epsilon) / (np.abs(truth) + epsilon)
            )
        ratio_list.append(float(np.mean(np.abs(db_diff) <= 3)))
        deviation_list.append(float(np.sum(np.abs(np.clip(db_diff, -30, 30)))))

    def average(values):
        finite = [value for value in values if np.isfinite(value)]
        return float(np.mean(finite)) if finite else float("nan")

    return {
        "MSE": average(mse_list), "MSE_list": mse_list,
        "RMSE": average(rmse_list), "RMSE_list": rmse_list,
        "MAE": average(mae_list), "MAE_list": mae_list,
        "R2": average(r2_list), "R2_list": r2_list,
        "db_within_3_ratio": average(ratio_list),
        "db_within_3_ratio_list": ratio_list,
        "total_db_deviation": float(np.nansum(deviation_list)) if any(valid_counts) else float("nan"),
        "total_db_deviation_per_feature": deviation_list,
        "valid_counts": valid_counts,
    }

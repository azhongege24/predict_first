import os
import time
import numpy as np
import pandas as pd
from mutar import GroupLasso
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

def group_lasso_predictor(
    data_train,
    data_test,
    input_columns,
    output_columns,
    alpha=0.1,
    random_state=42,
    show_plots=False,
    show_prints=False
):
    """
    Group Lasso 的接口函数，适配通用输入数据格式并返回统一结果。

    参数:
    --------
    data_train : DataFrame
        训练数据集（包含输入和输出列）。
    data_test : DataFrame
        测试数据集（包含输入和输出列）。
    input_columns : list
        输入特征列名列表。
    output_columns : list
        输出特征列名列表。
    alpha : float, default=0.1
        Group Lasso 的正则化强度。
    random_state : int, default=42
        随机种子。
    show_plots : bool, default=False
        是否显示可视化结果。
    show_prints : bool, default=False
        是否打印中间结果。

    返回:
    --------
    tuple
        (model, X_test, y_test, y_pred, metrics)
        - model: 训练好的 Group Lasso 模型。
        - X_test: 测试集特征。
        - y_test: 测试集真实值。
        - y_pred: 测试集预测值。
        - metrics: 包含 MSE 和 R² 的字典。
    """
    # 1. 数据准备
    X_train = data_train[input_columns].values.astype(float)
    y_train = data_train[output_columns].values.astype(float)
    X_test = data_test[input_columns].values.astype(float)
    y_test = data_test[output_columns].values.astype(float)

    # 转换为 3D 格式 (n_tasks, n_samples, n_features)
    n_tasks = len(output_columns)
    X_train_3d = np.repeat(X_train[None, :, :], n_tasks, axis=0)
    y_train_3d = y_train.T  # 转置为 (n_tasks, n_samples)
    X_test_3d = np.repeat(X_test[None, :, :], n_tasks, axis=0)

    if show_prints:
        print("=" * 50)
        print("数据维度信息:")
        print(f"训练集特征: {X_train.shape} -> 3D格式: {X_train_3d.shape}")
        print(f"训练集标签: {y_train.shape} -> 3D格式: {y_train_3d.shape}")
        print(f"测试集特征: {X_test.shape} -> 3D格式: {X_test_3d.shape}")
        print(f"任务数量: {n_tasks}")
        print("=" * 50)

    # 2. 模型训练
    grouplasso = GroupLasso(
        alpha=alpha,
        fit_intercept=True,
        normalize=False,
        max_iter=2000,
        tol=1e-4
    )

    if show_prints:
        print("开始训练 Group Lasso 模型...")

    grouplasso.fit(X_train_3d, y_train_3d)

    if show_prints:
        print("\n训练完成!")
        print("共享系数矩阵维度:", grouplasso.coef_shared_.shape)
        print("特定系数矩阵维度:", grouplasso.coef_specific_.shape)
        print("=" * 50)

    # 3. 预测与评估
    y_pred = grouplasso.predict(X_test_3d).T  # 转置回 (n_samples, n_tasks)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'MSE': mse,
        'R2': r2
    }

    if show_prints:
        print("\n测试集评估结果:")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
        print("\n前5个样本的预测结果:")
        print(pd.DataFrame(y_pred[:5], columns=output_columns))
        print("=" * 50)

    # 4. 可视化结果（可选）
    if show_plots:
        plt.figure(figsize=(12, 6))
        sns.heatmap(grouplasso.coef_shared_ + grouplasso.coef_specific_,
                    annot=True, fmt=".2f",
                    cmap="coolwarm",
                    xticklabels=output_columns,
                    yticklabels=input_columns)
        plt.title("总系数矩阵 (共享+特定)")
        plt.xlabel("输出任务")
        plt.ylabel("输入特征")
        plt.show()

    # 返回结果
    return grouplasso, X_test, y_test, y_pred, metrics

    #Group_Lasso的绘图函数，多任务输出
def group_lasso_plot_and_evaluate(self, coef_shared, coef_specific, method, input_columns, output_columns, MSE, R2):

    """
    绘制系数热力图（Coefficient Heatmap），并将每张图像保存到 GL_view 文件夹中。
    """
    self.end_time = time.time()
    print(f"运行时间: {self.end_time - self.start_time:.2f} 秒")

    # 创建保存图像的目录
    output_dir = "GL_view"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建第一个热力图页面并保存
    figure1 = plt.figure(figsize=(7, 4), dpi=120)
    ax1 = figure1.add_subplot(111)
    sns.heatmap(coef_shared + coef_specific, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=output_columns, yticklabels=input_columns, ax=ax1)
    ax1.set_title("总系数矩阵 (共享 + 特定)")
    ax1.set_xlabel("输出任务")
    ax1.set_ylabel("输入特征")
    figure1_path = os.path.join(output_dir, "heatmap_total.png")
    figure1.savefig(figure1_path)
    plt.close(figure1)

    # 创建第二个热力图页面并保存
    figure2 = plt.figure(figsize=(7, 4), dpi=120)
    ax2 = figure2.add_subplot(111)
    sns.heatmap(coef_shared, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=output_columns, yticklabels=input_columns, ax=ax2)
    ax2.set_title("共享系数矩阵")
    ax2.set_xlabel("输出任务")
    ax2.set_ylabel("输入特征")
    figure2_path = os.path.join(output_dir, "heatmap_shared.png")
    figure2.savefig(figure2_path)
    plt.close(figure2)

    # 将图像路径存储到列表中
    self.figures = [figure1_path, figure2_path]

    # 显示第一页
    self.current_page = 0
    self.update_graphics_view()

    # 更新界面控件
    self.lineEdit_state.setText('Finish!')
    self.lineEdit_MSE.setText(str(round(MSE, 5)))
    self.lineEdit_R2.setText(str(round(R2, 5)))
    self.lineEdit_Algorithm_name.setText(f"当前算法: {method}")
    #多任务Wasserstein的绘图函数

if __name__ == "__main__":
    # 模拟数据
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'target1': 2.5 * np.random.randn(100) + 3,
        'target2': 1.8 * np.random.randn(100) + 5
    })

    # 定义输入和输出列
    input_columns = ['feature1', 'feature2', 'feature3']
    output_columns = ['target1', 'target2']

    # 划分训练集和测试集
    train_size = 70
    data_train = data.iloc[:train_size]
    data_test = data.iloc[train_size:]

    # 调用 group_lasso_predictor 函数
    grouplasso, X_test, y_test, y_pred, metrics = group_lasso_predictor(
        data_train=data_train,
        data_test=data_test,
        input_columns=input_columns,
        output_columns=output_columns,
        alpha=0.1,
        random_state=42,
        show_plots=True,
        show_prints=True
    )

    # 打印结果
    print("\n测试完成!")
    print("MSE:", metrics['MSE'])
    print("R²:", metrics['R2'])
    print("预测值前5个样本:\n", pd.DataFrame(y_pred[:5], columns=output_columns))
import os
import time
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from mutar import ReMTW
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def REMTW_Lasso(data_train,
              data_test,
              input_columns,
              output_columns,
              model_type='REMTW',alpha=0.2,beta=0.1,gpu=True):
    #转换输入数据的格式

    X_train = data_train[input_columns].values
    y_train = data_train[output_columns].values
    X_test = data_test[input_columns].values
    y_test = data_test[output_columns].values

    n_tasks = y_train.shape[1]  # 任务数
    X_train_3d = np.repeat(X_train[np.newaxis, :, :], n_tasks, axis=0)  # 复制为三维
    y_train_3d = y_train.T  # 转置为 
    X_test_3d = np.repeat(X_test[np.newaxis, :, :], n_tasks, axis=0)  # 复制为三维

    model = ReMTW(alpha=alpha, beta=beta,gpu=gpu)  # 加强正则化
    model.fit(X_train_3d, y_train_3d)  # 训练模型
    y_pred = model.predict(X_test_3d).T  # 转置回 (n_samples, n_tasks)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {
        'MSE': mse,
        'R2': r2
    }
    return model, X_test, y_test, y_pred, metrics

def remtw_plot_and_evaluate(self, remtw_model, method, input_columns, output_columns, MSE, R2):
    """
    绘制 ReMTW 模型的可视化结果，包括系数热力图和 Wasserstein Barycenter。
    """
    self.end_time = time.time()
    print(f"运行时间: {self.end_time - self.start_time:.2f} 秒")

    # 创建保存图像的目录
    output_dir = "ReMTW_view"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建第一个热力图页面并保存
    # 1. 系数矩阵热力图
    fig1 = plt.figure(figsize=(7, 4), dpi=120)
    ax1 = fig1.add_subplot(111)
    sns.heatmap(remtw_model.coef_, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=output_columns, yticklabels=input_columns, ax=ax1)
    ax1.set_title("ReMTW Coefficient Matrix (W+ - W-)")
    fig1_path = os.path.join(output_dir, "heatmap_coef.png")
    fig1.savefig(fig1_path)
    plt.close(fig1)

    # 创建第二个热力图页面并保存
    # 2. Wasserstein Barycenter
    fig2 = plt.figure(figsize=(7, 4), dpi=120)
    ax2 = fig2.add_subplot(111)
    ax2.bar(range(len(remtw_model.barycenter_)), remtw_model.barycenter_)
    ax2.set_title("Wasserstein Barycenter")
    ax2.set_xlabel("Feature Index")
    ax2.set_ylabel("Barycenter Value")
    fig2_path = os.path.join(output_dir, "barycenter.png")
    fig2.savefig(fig2_path)
    plt.close(fig2)

    # 将图像路径存储到列表中
    self.figures = [fig1_path, fig2_path]

    # 显示第一页
    self.current_page = 0
    self.update_graphics_view()

    # 更新界面控件
    self.lineEdit_state.setText('Finish!')
    self.lineEdit_MSE.setText(str(round(MSE, 5)))
    self.lineEdit_R2.setText(str(round(R2, 5)))
    self.lineEdit_Algorithm_name.setText(f"当前算法: {method}")

def test_remtw_plot_and_evaluate(remtw_model, method, input_columns, output_columns, MSE, R2):
    """
    绘制 ReMTW 模型的可视化结果，包括系数热力图和 Wasserstein Barycenter。
    """
    import time
    start_time = time.time()

    # 创建保存图像的目录
    output_dir = "ReMTW_view"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 系数矩阵热力图
    fig1 = plt.figure(figsize=(7, 4), dpi=120)
    ax1 = fig1.add_subplot(111)
    sns.heatmap(remtw_model.coef_, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=output_columns, yticklabels=input_columns, ax=ax1)
    ax1.set_title("ReMTW Coefficient Matrix (W+ - W-)")
    fig1_path = os.path.join(output_dir, "heatmap_coef.png")
    fig1.savefig(fig1_path)
    plt.close(fig1)

    # 2. Wasserstein Barycenter
    fig2 = plt.figure(figsize=(7, 4), dpi=120)
    ax2 = fig2.add_subplot(111)
    ax2.bar(range(len(remtw_model.barycenter_)), remtw_model.barycenter_)
    ax2.set_title("Wasserstein Barycenter")
    ax2.set_xlabel("Feature Index")
    ax2.set_ylabel("Barycenter Value")
    fig2_path = os.path.join(output_dir, "barycenter.png")
    fig2.savefig(fig2_path)
    plt.close(fig2)

    print(f"运行时间: {time.time() - start_time:.2f} 秒")
    print("保存的图片：", fig1_path, fig2_path)
    print(f"MSE: {MSE:.5f}, R2: {R2:.5f}, 算法: {method}")


if __name__ == "__main__":
    # 示例数据
    data_train = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'task1': np.random.rand(100),
        'task2': np.random.rand(100)
    })
    data_test = pd.DataFrame({
        'feature1': np.random.rand(50),
        'feature2': np.random.rand(50),
        'task1': np.random.rand(50),
        'task2': np.random.rand(50)
    })

    input_columns = ['feature1', 'feature2']
    output_columns = ['task1', 'task2']

    # 调用函数
    model, X_test, y_test, y_pred, metrics = REMTW_Lasso(data_train, data_test, input_columns, output_columns)
    test_remtw_plot_and_evaluate(model, "ReMTW", input_columns, output_columns, metrics['MSE'], metrics['R2'])   
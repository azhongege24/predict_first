import os
import time
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from mutar import GroupLasso, MTW,ReMTW
from sklearn.metrics import r2_score, mean_squared_error
# 可视化各任务的特征权重热图
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import MultiTaskLasso
# 读取数据（假设CSV文件路径为data.csv）

def MTW_Lasso(data_train,
              data_test,
              input_columns,
              output_columns,
              model_type='MTW',alpha=1,beta=0.8,max_iter=2000,tol=1e-4,gpu=True):
    #转换输入数据的格式

    X_train = data_train[input_columns].values
    y_train = data_train[output_columns].values
    X_test = data_test[input_columns].values
    y_test = data_test[output_columns].values

    n_tasks = y_train.shape[1]  # 任务数
    X_train_3d = np.repeat(X_train[np.newaxis, :, :], n_tasks, axis=0)  # 复制为三维
    y_train_3d = y_train.T  # 转置为 
    X_test_3d = np.repeat(X_test[np.newaxis, :, :], n_tasks, axis=0)  # 复制为三维

    model = MTW(alpha=alpha, beta=beta,gpu=gpu,max_iter=max_iter, tol=tol)  # 加强正则化
    model.fit(X_train_3d, y_train_3d)  # 训练模型
    y_pred = model.predict(X_test_3d).T  # 转置回 (n_samples, n_tasks)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {
        'MSE': mse,
        'R2': r2,
        'RMSE': np.sqrt(mse),
        'MAE': np.mean(np.abs(y_test - y_pred))
    }
    return model, X_test, y_test, y_pred, metrics

def mtw_plot_and_evaluate(self,mtw_model, method, input_columns, output_columns, MSE,RMSE,MAE, R2):
    """
    绘制系数热力图（Coefficient Heatmap），并将每张图像保存到 GL_view 文件夹中。
    """
    self.end_time = time.time()
    print(f"运行时间: {self.end_time - self.start_time:.2f} 秒")

    # 创建保存图像的目录
    output_dir = "MTW_view"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建第一个热力图页面并保存
    # 1. 系数矩阵热力图
    fig1 = plt.figure(figsize=(7, 4), dpi=120)
    ax1 = fig1.add_subplot(111)
    sns.heatmap(mtw_model.coef_, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=output_columns, yticklabels=input_columns, ax=ax1)
    ax1.set_title("MTW Coefficient Matrix (W+ - W-)")
    fig1_path = os.path.join(output_dir, "heatmap_coef.png")
    fig1.savefig(fig1_path)
    plt.close(fig1)

    # 创建第二个热力图页面并保存
    fig2 = plt.figure(figsize=(7, 4), dpi=120)
    ax2 = fig2.add_subplot(111)
    ax2.bar(range(len(mtw_model.barycenter_)), mtw_model.barycenter_)
    ax2.set_title("Wasserstein Barycenter")
    ax2.set_xlabel("Feature Index")
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
    self.lineEdit_RMSE.setText(str(round(RMSE, 5)))
    self.lineEdit_MAE.setText(str(round(MAE, 5)))
    self.lineEdit_R2.setText(str(round(R2, 5)))
    self.lineEdit_Algorithm_name.setText(f"当前算法: {method}")
    




def Group_Lasso(file_path='path to your data',input_num=22,output_num=16):
    path = file_path
    df = pd.read_csv(path)  # 读取数据
    X_2d = df.iloc[:, :22].values  # 输入特征 (样本数×22)
    y_2d = df.iloc[:, 22:].values  # 输出特征 (样本数×16)

    # 转换为MuTaR所需的三维输入格式 (16任务×样本数×22特征)
    n_samples, n_features = X_2d.shape
    n_tasks = y_2d.shape[1]
    X_3d = np.repeat(X_2d[np.newaxis, :, :], n_tasks, axis=0)  # 复制为三维
    y_3d = y_2d.T  # 转置为 (16×样本数)
    X_3d_scaled = np.zeros_like(X_3d)
    for i in range(n_tasks):
        scaler = StandardScaler()
        X_3d_scaled[i] = scaler.fit_transform(X_3d[i])

    # 模型训练修正
    model = MTW(alpha=1, beta=0.8,gpu=True)  # 加强正则化
    model.fit(X_3d_scaled, y_3d)

    # 可视化修正
    coef = model.coef_
    coef_df = pd.DataFrame(coef,
                        index=df.columns[:22],
                        columns=[f"Task_{i+1}" for i in range(16)])

    plt.figure(figsize=(16, 6))
    sns.heatmap(coef_df.T,
            cmap='coolwarm',
            center=0,
            annot=False,
            linewidths=0.5)
    plt.xlabel("Features")
    plt.ylabel("Tasks")
    plt.title("Feature Importance Matrix (16 Tasks × 22 Features)")
    plt.tight_layout()
    plt.show()
    param_grid = {'alpha': np.logspace(-3, 1, 20)}
    cv_model = GridSearchCV(GroupLasso(), param_grid, cv=5, scoring='neg_mean_squared_error')
    cv_model.fit(X_3d, y_3d)
    print(f"最优alpha: {cv_model.best_params_['alpha']}")

if __name__ == "__main__":
    # 示例数据
    data_train = pd.DataFrame(np.random.rand(100, 22), columns=[f"feature_{i}" for i in range(22)])
    data_train['target_0'] = np.random.rand(100)
    data_train['target_1'] = np.random.rand(100)
    data_train['target_2'] = np.random.rand(100)
    data_train['target_3'] = np.random.rand(100)
    data_train['target_4'] = np.random.rand(100)
    data_train['target_5'] = np.random.rand(100)
    data_train['target_6'] = np.random.rand(100)
    data_train['target_7'] = np.random.rand(100)
    data_train['target_8'] = np.random.rand(100)
    data_train['target_9'] = np.random.rand(100)
    data_train['target_10'] = np.random.rand(100)
    data_train['target_11'] = np.random.rand(100)
    data_train['target_12'] = np.random.rand(100)
    data_train['target_13'] = np.random.rand(100)
    data_train['target_14'] = np.random.rand(100)
    data_train['target_15'] = np.random.rand(100)

    input_columns = [f"feature_{i}" for i in range(22)]
    output_columns = [f"target_{i}" for i in range(16)]

    # 调用函数
    model, X_test, y_test, y_pred, metrics = MTW_Lasso(data_train, data_test=data_train,
                                                       input_columns=input_columns,
                                                       output_columns=output_columns,
                                                       model_type='MTW', alpha=1, beta=0.8, gpu=True)

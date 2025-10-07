import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from sklearn.metrics import mean_squared_error, r2_score
from mutar import MTW
from sklearn.preprocessing import StandardScaler

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


def MTW_Lasso(data_train,
              data_test,
              input_columns,
              output_columns,
              model_type='MTW',
              alpha=1,
              beta=0.8,
              max_iter=2000,
              tol=1e-4,
              gpu=True):
    # 转换输入数据的格式
    X_train = data_train[input_columns].values
    y_train = data_train[output_columns].values
    X_test = data_test[input_columns].values
    y_test = data_test[output_columns].values

    n_tasks = y_train.shape[1]  # 任务数
    X_train_3d = np.repeat(X_train[np.newaxis, :, :], n_tasks, axis=0)  # 复制为三维
    y_train_3d = y_train.T  # 转置为 (n_tasks, n_samples)
    X_test_3d = np.repeat(X_test[np.newaxis, :, :], n_tasks, axis=0)  # 复制为三维

    model = MTW(alpha=alpha, beta=beta, gpu=gpu, max_iter=max_iter, tol=tol)  # 初始化模型
    model.fit(X_train_3d, y_train_3d)  # 训练模型
    y_pred = model.predict(X_test_3d).T  # 转置回 (n_samples, n_tasks)

    # 计算每个输出的单独指标（用于可视化）
    mse_list = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(n_tasks)]
    r2_list = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(n_tasks)]
    rmse_list = [np.sqrt(mse) for mse in mse_list]
    mae_list = [np.mean(np.abs(y_test[:, i] - y_pred[:, i])) for i in range(n_tasks)]

    # 整体指标（平均值）
    metrics = {
        'MSE': np.mean(mse_list),
        'MSE_list': mse_list,
        'R2': np.mean(r2_list),
        'R2_list': r2_list,
        'RMSE': np.mean(rmse_list),
        'RMSE_list': rmse_list,
        'MAE': np.mean(mae_list),
        'MAE_list': mae_list
    }
    return model, X_test, y_test, y_pred, metrics, data_test.index  # 新增返回测试集索引


def mtw_plot_and_evaluate(self, mtw_model, method, input_columns, output_columns,
                         MSE, MSE_list, RMSE, RMSE_list, MAE, MAE_list, R2, R2_list,
                         y_test, y_pred, data_test_index):
    """
    绘制MTW模型的完整可视化结果：
    1. 系数矩阵热力图
    2. Wasserstein Barycenter柱状图
    3. 各输出变量的真实值-预测值对比图
    """
    self.end_time = time.time()
    print(f"运行时间: {self.end_time - self.start_time:.2f} 秒")

    # 创建保存图像的目录
    output_dir = "MTW_view"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # 清空旧图
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"无法删除文件 {file_path}。原因: {e}")

    # 存储所有图像路径
    self.figures = []

    # 1. 系数矩阵热力图
    fig1 = plt.figure(figsize=(7, 4), dpi=120)
    ax1 = fig1.add_subplot(111)
    sns.heatmap(mtw_model.coef_, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=output_columns, yticklabels=input_columns, ax=ax1)
    ax1.set_title("MTW 系数矩阵 (W+ - W-)")
    ax1.set_xlabel("输出任务")
    ax1.set_ylabel("输入特征")
    fig1_path = os.path.join(output_dir, "heatmap_coef.png")
    fig1.savefig(fig1_path, bbox_inches='tight')
    plt.close(fig1)
    self.figures.append(fig1_path)

    # 2. Wasserstein Barycenter柱状图
    fig2 = plt.figure(figsize=(7, 4), dpi=120)
    ax2 = fig2.add_subplot(111)
    ax2.bar(range(len(mtw_model.barycenter_)), mtw_model.barycenter_, color='#2196F3')
    ax2.set_title("Wasserstein 重心")
    ax2.set_xlabel("特征索引")
    ax2.set_ylabel("重心值")
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    fig2_path = os.path.join(output_dir, "barycenter.png")
    fig2.savefig(fig2_path, bbox_inches='tight')
    plt.close(fig2)
    self.figures.append(fig2_path)

    # 3. 各输出变量的真实值-预测值对比图
    n_outputs = len(output_columns)
    # 处理测试集索引（兼容RangeIndex）
    test_index_range = f"[{data_test_index[0]}:{data_test_index[-1]}]" if not data_test_index.empty else "[0:0]"

    for i in range(n_outputs):
        # 当前输出的指标
        current_mse = MSE_list[i]
        current_r2 = R2_list[i]

        # 创建对比图
        fig = plt.figure(figsize=(7, 4), dpi=120)
        ax = fig.add_subplot(111)

        # 绘制真实值与预测值
        ax.scatter(np.arange(len(y_test[:, i])), y_test[:, i],
                  c='b', marker='o', s=10, label='真实值', alpha=0.8)
        ax.scatter(np.arange(len(y_test[:, i])), y_pred[:, i],
                  c='r', marker='X', s=20, label=f'预测值_{method}', alpha=0.8)

        # 绘制连接线段（显示偏差）
        x = np.arange(len(y_test[:, i]))
        ax.plot([x, x], [y_test[:, i], y_pred[:, i]],
                color='#2F5597', linestyle='-', linewidth=1.5, alpha=0.6, zorder=0)

        # 图表标题和标签
        ax.set_title(
            f'输出变量: {output_columns[i]}\nMSE: {current_mse:.4f}, R: {current_r2:.4f}\n索引范围: {test_index_range}',
            fontsize=10
        )
        ax.set_xlabel('样本索引', fontsize=10)
        ax.set_ylabel('数值', fontsize=10)
        ax.grid(linestyle='--', alpha=0.5)
        ax.legend()
        ax.set_xlim(-5, len(y_test[:, i]) + 5)
        ax.set_facecolor('#f8f9fa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 保存图像
        fig_path = os.path.join(output_dir, f"pred_vs_true_{output_columns[i]}.png")
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)
        self.figures.append(fig_path)

    # 初始化分页显示（所有图像：热力图+柱状图+对比图）
    self.current_page = 0
    self.update_graphics_view()

    # 更新界面控件（显示整体指标）
    self.lineEdit_state.setText('Finish!')
    self.lineEdit_MSE.setText(f"{MSE:.5f}")
    self.lineEdit_RMSE.setText(f"{RMSE:.5f}")
    self.lineEdit_MAE.setText(f"{MAE:.5f}")
    self.lineEdit_R2.setText(f"{R2:.5f}")
    self.lineEdit_Algorithm_name.setText(f"当前算法: {method}")


def test_mtw_plot_and_evaluate(model, method, input_columns, output_columns, metrics, y_test, y_pred, data_test_index):
    """测试用可视化函数（模拟类环境）"""
    class MockSelf:
        def __init__(self):
            self.start_time = time.time()
            self.figures = []
            self.current_page = 0
        def update_graphics_view(self):
            print(f"分页显示初始化，共 {len(self.figures)} 张图像")

    mock_self = MockSelf()
    mtw_plot_and_evaluate(
        mock_self,
        mtw_model=model,
        method=method,
        input_columns=input_columns,
        output_columns=output_columns,
        MSE=metrics['MSE'],
        MSE_list=metrics['MSE_list'],
        RMSE=metrics['RMSE'],
        RMSE_list=metrics['RMSE_list'],
        MAE=metrics['MAE'],
        MAE_list=metrics['MAE_list'],
        R2=metrics['R2'],
        R2_list=metrics['R2_list'],
        y_test=y_test,
        y_pred=y_pred,
        data_test_index=data_test_index
    )
    print("可视化完成，图像保存路径：")
    for path in mock_self.figures:
        print(path)


# 原Group_Lasso函数（保留但注释说明其用途）
def Group_Lasso_deprecated(file_path='path to your data', input_num=22, output_num=16):
    """
    注意：此函数为历史遗留代码，用于演示Group Lasso的用法，与当前MTW模型无关
    如需使用请单独调用或重构
    """
    path = file_path
    df = pd.read_csv(path)  # 读取数据
    X_2d = df.iloc[:, :input_num].values  # 输入特征
    y_2d = df.iloc[:, input_num:input_num+output_num].values  # 输出特征

    # 转换为三维格式
    n_samples, n_features = X_2d.shape
    n_tasks = y_2d.shape[1]
    X_3d = np.repeat(X_2d[np.newaxis, :, :], n_tasks, axis=0)
    y_3d = y_2d.T
    X_3d_scaled = np.zeros_like(X_3d)
    for i in range(n_tasks):
        scaler = StandardScaler()
        X_3d_scaled[i] = scaler.fit_transform(X_3d[i])

    # 模型训练与可视化
    model = MTW(alpha=1, beta=0.8, gpu=True)
    model.fit(X_3d_scaled, y_3d)
    
    coef_df = pd.DataFrame(
        model.coef_,
        index=df.columns[:input_num],
        columns=[f"Task_{i+1}" for i in range(output_num)]
    )
    plt.figure(figsize=(16, 6))
    sns.heatmap(coef_df.T, cmap='coolwarm', center=0, annot=False, linewidths=0.5)
    plt.xlabel("特征")
    plt.ylabel("任务")
    plt.title(f"特征重要性矩阵 ({output_num}任务 × {input_num}特征)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 生成示例数据（训练集和测试集分离）
    np.random.seed(42)  # 固定随机种子
    data_train = pd.DataFrame(
        np.random.rand(100, 22), 
        columns=[f"feature_{i}" for i in range(22)]
    )
    data_test = pd.DataFrame(
        np.random.rand(50, 22),  # 独立测试集
        columns=[f"feature_{i}" for i in range(22)]
    )
    
    # 生成目标变量（模拟与特征的弱相关性）
    for i in range(16):
        data_train[f"target_{i}"] = 0.3 * data_train[f"feature_{i%22}"] + 0.1 * np.random.rand(100)
        data_test[f"target_{i}"] = 0.3 * data_test[f"feature_{i%22}"] + 0.1 * np.random.rand(50)

    input_columns = [f"feature_{i}" for i in range(22)]
    output_columns = [f"target_{i}" for i in range(16)]

    # 训练模型并获取结果
    model, X_test, y_test, y_pred, metrics, test_index = MTW_Lasso(
        data_train, data_test, input_columns, output_columns,
        alpha=1, beta=0.8, gpu=True
    )

    # 测试可视化函数
    test_mtw_plot_and_evaluate(
        model, "MTW", input_columns, output_columns,
        metrics, y_test, y_pred, test_index
    )
    
import os
import shutil
import time
import numpy as np
import pandas as pd
from mutar import GroupLasso
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import rcParams
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from ALL_Algorithms.Algorithms import Multi_output_plot_and_evaluate

# 设置中文显示 matplotlib
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

def group_lasso_predictor(
    data_train,
    data_test,
    input_columns,
    output_columns,
    alpha=0.1,
    max_iter=2000,
    tol=1e-4,
    random_state=42,
    scale=True,
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
    # 检查并清理NaN和无穷大的值
    def clean_data(arr):
        """清理数据中的NaN和无穷大值"""
        arr = np.asarray(arr)
        # 替换NaN和无穷大值为0
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
        return arr
    X_train = clean_data(X_train)
    y_train = clean_data(y_train)
    X_test = clean_data(X_test)
    y_test = clean_data(y_test)

    if scale:
        
        # 对特征进行标准化
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        # 对目标变量进行标准化
        
        
        # 使用标准化后的数据
        X_train = X_train_scaled
        
        X_test = X_test_scaled
        


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
        normalize=True,
        max_iter=max_iter,
        tol=tol
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
    
    # 清理预测结果中的NaN和无穷大值
    y_pred = clean_data(y_pred)
    
    # 计算每个输出的单独指标（用于可视化）
    mse_list = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(n_tasks)]
    r2_list = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(n_tasks)]
    rmse_list = [np.sqrt(mse) for mse in mse_list]
    mae_list = [np.mean(np.abs(y_test[:, i] - y_pred[:, i])) for i in range(n_tasks)]
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'MSE': mse,
        'MSE_list': mse_list,
        'R2': r2,
        'R2_list': r2_list,
        'RMSE': np.sqrt(mse),
        'RMSE_list': rmse_list,
        'MAE': np.mean(np.abs(y_test - y_pred)),
        'MAE_list': mae_list
    }
    # 新增：计算分贝偏差指标
    epsilon = 1e-8  # 避免零值导致的计算错误
    y_test_pos = y_test + epsilon  # 确保非负
    y_pred_pos = y_pred + epsilon
    # 计算每个样本的分贝偏差 (20*log10(预测值/真实值))
    db_diff = 20 * np.log10(y_pred_pos / y_test_pos)
    # 指标1：每个输出特征中偏差在±3dB内的样本比例（再求平均）
    db_within_3_ratio_list = []
    for j in range(n_tasks):
        # 计算当前特征中偏差在±3dB内的样本占比
        within_range = np.abs(db_diff[:, j]) <= 3
        ratio = np.mean(within_range)  # 比例
        db_within_3_ratio_list.append(ratio)
    db_within_3_ratio = np.mean(db_within_3_ratio_list)  # 整体平均比例
    # 指标2：所有特征的偏差分贝数绝对值之和
    # total_db_deviation = np.sum(np.abs(db_diff))
    total_db_deviation_per_feature = [np.sum(np.abs(db_diff[:, j])) for j in range(n_tasks)]
    # 更新metrics字典
    metrics.update({
        'db_within_3_ratio': db_within_3_ratio,  # 整体±3dB内比例
        'db_within_3_ratio_list': db_within_3_ratio_list,  # 各特征±3dB内比例
        # 'total_db_deviation': total_db_deviation,  # 总分贝偏差和
        'total_db_deviation_per_feature': total_db_deviation_per_feature  # 各特征分贝偏差和
    })
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
    return grouplasso, X_test, y_test, y_pred, metrics,data_test.index

    #Group_Lasso的绘图函数，多任务输出

def group_lasso_plot_and_evaluate(self, coef_shared, coef_specific, method, input_columns, output_columns, 
                                 MSE,MSE_list, RMSE,RMSE_list, MAE,MAE_list, 
                                 R2,R2_list,
                                 db_within_3_ratio_list,total_db_deviation_per_feature,
                                 y_test, y_pred, data_test):
    """
    绘制Group Lasso模型的系数热力图和各输出变量的真实值-预测值对比图
    
    参数:
        coef_shared: 共享系数矩阵
        coef_specific: 特定系数矩阵
        method: 算法名称
        input_columns: 输入特征列名
        output_columns: 输出特征列名
        MSE: 均方误差（整体或各输出列表）
        RMSE: 均方根误差（整体或各输出列表）
        MAE: 平均绝对误差（整体或各输出列表）
        R2: 决定系数（整体或各输出列表）
        y_test: 测试集真实值 (形状 [N_test, n_outputs])
        y_pred: 测试集预测值 (形状 [N_test, n_outputs])
        data_test_index: 测试集索引列表
    """
    self.end_time = time.time()
    print(f"运行时间: {self.end_time - self.start_time:.2f} 秒")

    # 创建保存图像的目录
    output_dir = "GL_view"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # 清空目录避免旧图干扰
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"无法删除文件 {file_path}。原因: {e}")

    # 存储所有图像路径的列表
    self.figures = []

    # 1. 绘制系数热力图（总系数矩阵）
    fig1 = plt.figure(figsize=(7, 4), dpi=120)
    ax1 = fig1.add_subplot(111)
    sns.heatmap(coef_shared + coef_specific, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=output_columns, yticklabels=input_columns, ax=ax1)
    ax1.set_title("总系数矩阵 (共享 + 特定)")
    ax1.set_xlabel("输出任务")
    ax1.set_ylabel("输入特征")
    fig1_path = os.path.join(output_dir, "heatmap_total.png")
    fig1.savefig(fig1_path, bbox_inches='tight')
    plt.close(fig1)
    self.figures.append(fig1_path)

    # 2. 绘制共享系数矩阵热力图
    fig2 = plt.figure(figsize=(7, 4), dpi=120)
    ax2 = fig2.add_subplot(111)
    sns.heatmap(coef_shared, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=output_columns, yticklabels=input_columns, ax=ax2)
    ax2.set_title("共享系数矩阵")
    ax2.set_xlabel("输出任务")
    ax2.set_ylabel("输入特征")
    fig2_path = os.path.join(output_dir, "heatmap_shared.png")
    fig2.savefig(fig2_path, bbox_inches='tight')
    plt.close(fig2)
    self.figures.append(fig2_path)

    # # 3. 绘制各输出变量的真实值-预测值对比图
    # n_outputs = len(output_columns)
    # test_index_range = f"[{data_test_index[0]}:{data_test_index[-1]}]" if not data_test_index.empty else "[0:0]"
    
    for i,col in enumerate(output_columns):
        # 获取当前输出的评估指标（支持单值或列表）
        current_mse = MSE_list[i]
        current_r2 = R2_list[i]
        current_rmse = RMSE_list[i] if RMSE_list else np.sqrt(current_mse)
        current_mae = MAE_list[i] if MAE_list else np.mean(np.abs(y_test[:, i] - y_pred[:, i]))
        current_db_within_3_ratio = db_within_3_ratio_list[i] if db_within_3_ratio_list else None
        current_total_db_deviation_per_feature = total_db_deviation_per_feature[i] if total_db_deviation_per_feature else None
        # 创建对比图
        fig = plt.figure(figsize=(7, 4), dpi=120)
        ax = fig.add_subplot(111)
        
        # 绘制真实值与预测值散点
        flag = (len(np.unique(y_test)) > 1)
        if flag:
        # 绘制真实值与预测值散点
            ax.scatter(np.arange(len(y_test[:, i])), y_test[:, i], 
                    c='b', marker='o', s=10, label='真实值', alpha=0.8)
            ax.scatter(np.arange(len(y_test[:, i])), y_pred[:, i], 
                    c='r', marker='X', s=20, label=f'预测值_{method}', alpha=0.8)
            
            # 绘制连接线段
            x = np.arange(len(y_test[:, i]))
            ax.plot([x, x], [y_test[:, i], y_pred[:, i]],
                    color='#2F5597', linestyle='-', linewidth=1.5, alpha=0.6, zorder=0)
        else:
            ax.scatter(np.arange(len(y_test[:, i])), y_pred[:, i], 
                    c='r', marker='X', s=20, label=f'预测值_{method}', alpha=0.8)
        
        # 设置图表属性
        # 设置子图标题和标签（显示当前输出的独立指标）
        ax.set_title(
            f'输出变量: {col}\n'
            f'MSE: {current_mse:.4f}, RMSE: {current_rmse:.4f}\n'
            f'MAE: {current_mae:.4f}, R^2: {current_r2:.4f}\n'
            f'±3dB比例: {current_db_within_3_ratio:.2%}\n',
            # f'总分贝偏差: {current_total_db_deviation_per_feature:.2f} dB',
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

    # 初始化分页显示（包含所有图像：热力图+对比图）
    self.current_page = 0
    self.update_graphics_view()

    # 更新界面控件（显示整体指标）
    if flag:
        # 更新界面控件（显示整体指标）
        overall_mse = np.mean(MSE) if isinstance(MSE, (list, np.ndarray)) else MSE
        overall_rmse = np.mean(RMSE) if isinstance(RMSE, (list, np.ndarray)) else RMSE
        overall_mae = np.mean(MAE) if isinstance(MAE, (list, np.ndarray)) else MAE
        overall_r2 = np.mean(R2) if isinstance(R2, (list, np.ndarray)) else R2

        self.lineEdit_state.setText('Finish!')
        self.lineEdit_MSE.setText(f"{overall_mse:.5f}")
        self.lineEdit_RMSE.setText(f"{overall_rmse:.5f}")
        self.lineEdit_MAE.setText(f"{overall_mae:.5f}")
        self.lineEdit_R2.setText(f"{overall_r2:.5f}")
        self.lineEdit_db_within_3_ratio.setText(f"{np.mean(db_within_3_ratio_list)*100:.2f}%")
    
    else:
        self.lineEdit_state.setText('Finish!')
        self.lineEdit_MSE.setText("None")
        self.lineEdit_RMSE.setText("None")
        self.lineEdit_MAE.setText("None")
        self.lineEdit_R2.setText("None")
        self.lineEdit_db_within_3_ratio.setText("None")
        
    self.lineEdit_Algorithm_name.setText(f"当前算法: {method}")
    # 新增：保存预测结果到data_save属性
    # 检查测试集数据是否包含时间列
    # 修改：使用data_test而不是data_test_index来检查时间列
    time_columns = ['start_time', 'end_time']
    time_columns_present = [col for col in time_columns if col in data_test.columns] if hasattr(data_test, 'columns') else []
    
    if len(time_columns_present) == 2 and hasattr(data_test, 'columns'):
        # 计算时间中心点
        time_center = (data_test['start_time'] + data_test['end_time']) / 2
        # 创建包含真实值、预测值和时间中心点的DataFrame
        pred_df = pd.DataFrame(
            y_pred, 
            index=data_test.index,
            columns=[f"pred_{col}" for col in output_columns]
        )
        true_df = pd.DataFrame(
            y_test,
            index=data_test.index,
            columns=[f"true_{col}" for col in output_columns]
        )
        time_center_df = pd.DataFrame({
            'original_index': data_test.index,
            'time_center': time_center.values
        }).reset_index(drop=True)
        
        pred_df_reset = pred_df.reset_index(drop=True)
        true_df_reset = true_df.reset_index(drop=True)
        self.data_save = pd.concat([time_center_df, true_df_reset, pred_df_reset], axis=1)
        print("GroupLasso预测结果已保存到data_save，包含时间中心点列")
    else:
        # 如果没有时间列，使用基本保存方式
        pred_df = pd.DataFrame(
            y_pred, 
            index=data_test.index if hasattr(data_test, 'index') else range(len(y_pred)),
            columns=[f"pred_{col}" for col in output_columns]
        )
        true_df = pd.DataFrame(
            y_test,
            index=data_test.index if hasattr(data_test, 'index') else range(len(y_test)),
            columns=[f"true_{col}" for col in output_columns]
        )
        self.data_save = pd.concat([true_df, pred_df], axis=1)
        print("GroupLasso预测结果已保存到data_save")


if __name__ == "__main__":
    # 示例数据
    data_train = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'output1': np.random.rand(100),
        'output2': np.random.rand(100)
    })

    data_test = pd.DataFrame({
        'feature1': np.random.rand(50),
        'feature2': np.random.rand(50),
        'output1': np.random.rand(50),
        'output2': np.random.rand(50)
    })

    input_columns = ['feature1', 'feature2']
    output_columns = ['output1', 'output2']

    model, X_test, y_test, y_pred, metrics = group_lasso_predictor(
        data_train,
        data_test,
        input_columns,
        output_columns,
        alpha=0.1,
        random_state=42,
        show_plots=True,
        show_prints=True
    )
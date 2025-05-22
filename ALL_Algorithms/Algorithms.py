import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView,QMessageBox, QFileDialog
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from tqdm import tqdm
import shutil
import joblib



def multi_task_regression_predictor(
    data_train: pd.DataFrame,          # ▶ 接收主程序处理后的训练集
    data_test: pd.DataFrame,           # ▶ 接收主程序处理后的测试集
    input_columns: list,               # ▶ 输入特征列名列表
    output_columns: list,              # ▶ 输出目标列名列表
    model_type: str = 'RF',
    scale_features: bool = True,
    random_state: int = 42,
    max_depth: int = 4,
    n_estimators=100,                   #用于RF/ET等模型
    kernel='rbf',
    C=1.0,
    epsilon=0.1,
    n_jobs=-1,
    max_iter=500,                   # MLP最大迭代次数
    mlp_hidden_layers: tuple = (100, 50), # MLP隐藏层结构

):
    
    
    # 返回：
    # model : 训练好的回归模型
    # y_test : 测试集真实值（形状 [N_test, output_cols]）
    # y_pred : 预测结果（形状 [N_test, output_cols]）
    # metrics : 包含MSE和R2的字典
  
    # 数据加载与预处理
    X_train = data_train[input_columns].values
    y_train = data_train[output_columns].values
    X_test = data_test[input_columns].values
    y_test = data_test[output_columns].values

    scaler = None    # 特征标准化(返回scaler用于后续预测)
  
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    

    
    # 模型选择与初始化
    model_mapping = {
        'DT': DecisionTreeRegressor(max_depth=max_depth, random_state=random_state),
        'RF': RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs, random_state=random_state),
        'SVM': MultiOutputRegressor(SVR(kernel=kernel, C=C, epsilon=epsilon), n_jobs=n_jobs),
        'MLP': MLPRegressor(hidden_layer_sizes=mlp_hidden_layers, max_iter=max_iter, random_state=random_state),
        'ET': ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs, random_state=random_state)
    }
    
    if model_type not in model_mapping:
        raise ValueError(f"Unsupported model type: {model_type}. Available options: {list(model_mapping.keys())}")
    
    model = model_mapping[model_type]
    
    # 模型训练
    print("开始训练模型...")
    with tqdm(total=1, desc="训练进度", unit="step") as pbar:  # 使用 tqdm 显示进度
        model.fit(X_train, y_train)
        pbar.update(1)  # 更新进度
    # 预测与评估
    y_pred = model.predict(X_test)
    
    
    # 计算指标
    mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
    metrics = {'MSE': mse, 'R2': r2}

    
    
    
    return model, scaler, y_test, y_pred, metrics

def ask_and_save_model(parent, model, default_name="model.pkl"):
    """
    训练后询问是否保存模型，并保存到用户指定路径
    :param parent: 父窗口self
    :param model: 训练好的模型对象
    :param default_name: 默认保存文件名
    """
    reply = QMessageBox.question(
        parent,
        "保存模型",
        "是否保存训练好的模型？",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )
    if reply == QMessageBox.Yes:
        file_path, _ = QFileDialog.getSaveFileName(
            parent,
            "保存模型文件",
            f"./trained_models/{default_name}",
            "模型文件 (*.pkl *.joblib);;所有文件 (*)"
        )
        if file_path:
            joblib.dump(model, file_path)
            if hasattr(parent, "lineEdit_state"):
                parent.lineEdit_state.setText("模型已保存 " )

#单输出画图可视化函数
def single_plot_and_evaluate(self, y_test, y_pred, method, data_test, 
                                output_columns, N_start_test, N_end_test,MSE,R2):
    """
    绘制真实值与预测值的散点图，计算评估指标，并更新界面控件。

    :param y_test: 测试集真实值
    :param y_pred: 测试集预测值
    :param method: 当前使用的算法名称
    :param data_test: 测试集数据
    :param output_columns: 输出特征列名
    :param N_start_test: 测试集起始索引
    :param N_end_test: 测试集结束索引
    """
    self.end_time = time.time()
    print(f"运行时间: {self.end_time - self.start_time:.2f} 秒")

    # 创建绘图
    self.figure = plt.figure(figsize=(7, 4), dpi=120)
    self.canvas = FigureCanvasQTAgg(self.figure)
    self.navi = NavigationToolbar(self.canvas, self.graphicsView)

    ax = self.figure.add_subplot(111)

    # 绘制真实值与预测值散点图
    scatter1 = ax.scatter(np.arange(len(y_test)), y_test, c='b', marker='o', s=10, label='True', alpha=0.8)
    scatter2 = ax.scatter(np.arange(len(y_test)), y_pred, c='r', marker='X', s=20, label='pred_' + method, alpha=0.8)

    # 绘制垂直连接线段
    for i in range(len(y_test)):
        ax.plot([i, i], [y_test[i], y_pred[i]],
                color='#2F5597', linestyle='-', linewidth=2.5, alpha=0.9, solid_capstyle='round', zorder=0)

    # 优化显示设置
    plt.grid(linestyle='--', alpha=0.5)
    plt.legend()
    ax.set_xlim(-5, len(y_test) + 5)
    ax.set_facecolor('#f8f9fa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    self.canvas.draw()

    # 更新图形到界面
    self.graphicscene = QGraphicsScene()
    self.graphicscene.addWidget(self.canvas)
    self.graphicsView.setScene(self.graphicscene)
    self.graphicsView.setDragMode(QGraphicsView.ScrollHandDrag)
    self.graphicsView.show()

    # 更新界面控件
    self.lineEdit_state.setText('Finish!')
    self.lineEdit_MSE.setText(str(round(MSE, 5)))
    self.lineEdit_R2.setText(str(round(R2, 5)))

    # 保存预测结果到 DataFrame
    self.data_save = pd.DataFrame(y_pred, index=data_test.index.values)
    self.lineEdit_Algorithm_name.setText(
        f'当前生成数据: {output_columns[0]} [ {N_start_test}:{N_end_test} ]'
    )
    return self.data_save
def multi_output_plot_and_evaluate(self, y_test, y_pred, method, data_test, 
                                   output_columns, N_start_test, N_end_test, MSE, R2):
    """
    绘制多输出特征的真实值与预测值的散点图，计算评估指标，并更新界面控件。

    :param y_test: 测试集真实值 (二维数组)
    :param y_pred: 测试集预测值 (二维数组)
    :param method: 当前使用的算法名称
    :param data_test: 测试集数据
    :param output_columns: 输出特征列名
    :param N_start_test: 测试集起始索引
    :param N_end_test: 测试集结束索引
    :param MSE: 均方误差
    :param R2: R² 分数
    """
    self.end_time = time.time()
    print(f"运行时间: {self.end_time - self.start_time:.2f} 秒")

       # 创建保存图像的目录
    output_dir = "MultiOutput_view"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
        # 清空之前的图像
    self.figures = []


    # 创建绘图
    self.figure = plt.figure(figsize=(10, 6), dpi=120)
    self.canvas = FigureCanvasQTAgg(self.figure)
    self.navi = NavigationToolbar(self.canvas, self.graphicsView)

    num_outputs = y_test.shape[1]  # 输出特征数量
    for i in range(num_outputs):
        ax = self.figure.add_subplot(num_outputs, 1, i + 1)  # 创建子图
        ax.scatter(np.arange(len(y_test[:, i])), y_test[:, i], c='b', marker='o', s=10, label='True', alpha=0.8)
        ax.scatter(np.arange(len(y_test[:, i])), y_pred[:, i], c='r', marker='X', s=20, label='Pred_' + method, alpha=0.8)

        # 绘制垂直连接线段
        for j in range(len(y_test[:, i])):
            ax.plot([j, j], [y_test[j, i], y_pred[j, i]],
                    color='#2F5597', linestyle='-', linewidth=2.5, alpha=0.9, solid_capstyle='round', zorder=0)

        # 优化显示设置
        ax.grid(linestyle='--', alpha=0.5)
        ax.legend()
        ax.set_xlim(-5, len(y_test[:, i]) + 5)
        ax.set_facecolor('#f8f9fa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f"Output: {output_columns[i]}")

    self.canvas.draw()

    # 更新图形到界面
    self.graphicscene = QGraphicsScene()
    self.graphicscene.addWidget(self.canvas)
    self.graphicsView.setScene(self.graphicscene)
    self.graphicsView.setDragMode(QGraphicsView.ScrollHandDrag)
    self.graphicsView.show()

    # 更新界面控件
    self.lineEdit_state.setText('Finish!')
    self.lineEdit_MSE.setText(str(round(MSE, 5)))
    self.lineEdit_R2.setText(str(round(R2, 5)))

    # 保存预测结果到 DataFrame
    self.data_save = pd.DataFrame(y_pred, index=data_test.index.values, columns=output_columns)
    self.lineEdit_Algorithm_name.setText(
        f'当前生成数据: {", ".join(output_columns)} [ {N_start_test}:{N_end_test} ]'
    )

#新的翻页多输出结果可视化
def Multi_output_plot_and_evaluate(self, y_test, y_pred, method, data_test, 
                                   output_columns, N_start_test, N_end_test, MSE, R2):
    """
    多输出回归模型的分页可视化函数
    
    参数:
        y_test: 测试集真实值 (形状 [N_test, n_outputs])
        y_pred: 测试集预测值 (形状 [N_test, n_outputs])
        method: 当前使用的算法名称
        data_test: 测试集数据
        output_columns: 输出特征列名列表
        N_start_test: 测试集起始索引
        N_end_test: 测试集结束索引
        MSE: 各输出变量的MSE列表
        R2: 各输出变量的R2列表
    """
    self.end_time = time.time()
    print(f"运行时间: {self.end_time - self.start_time:.2f} 秒")
    
    # 创建保存图像的目录
    output_dir = "MultiOutput_view/"+""+method+"_view"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
         # 清空文件夹内容
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或符号链接
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除子文件夹
            except Exception as e:
                print(f"无法删除文件 {file_path}。原因: {e}")
    
    # 清空之前的图像
    self.figures = []
    
    # 为每个输出变量创建并保存可视化图像
    for i, col in enumerate(output_columns):
        # 创建图像
        fig = plt.figure(figsize=(7, 4), dpi=120)
        ax = fig.add_subplot(111)
        
        # 绘制真实值与预测值散点图
        scatter1 = ax.scatter(np.arange(len(y_test[:, i])), y_test[:, i], 
                            c='b', marker='o', s=10, label='True', alpha=0.8)
        scatter2 = ax.scatter(np.arange(len(y_test[:, i])), y_pred[:, i], 
                            c='r', marker='X', s=20, label=f'Pred_{method}', alpha=0.8)
        
        # 绘制垂直连接线段
        for j in range(len(y_test[:, i])):
            ax.plot([j, j], [y_test[j, i], y_pred[j, i]],
                    color='#2F5597', linestyle='-', linewidth=1.5, alpha=0.6, zorder=0)
        
        # 设置子图标题和标签
        ax.set_title(f'Output: {col} (MSE: {MSE:.4f}, R2: {R2:.4f})', fontsize=12)
        ax.set_xlabel('Sample Index', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(linestyle='--', alpha=0.5)
        ax.legend()
        ax.set_xlim(-5, len(y_test[:, i]) + 5)
        ax.set_facecolor('#f8f9fa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 保存图像
        fig_path = os.path.join(output_dir, f"output_{col}.png")
        fig.savefig(fig_path)
        plt.close(fig)
        
        # 将图像路径添加到列表中
        self.figures.append(fig_path)
    
    # 初始化分页显示
    self.current_page = 0
    self.update_graphics_view()
    
    # 更新界面控件
    self.lineEdit_state.setText('Finish!')
    self.lineEdit_MSE.setText(str(round(MSE, 5)))
    self.lineEdit_R2.setText(str(round(R2, 5)))
    
    # 保存预测结果到 DataFrame
    self.data_save = pd.DataFrame(y_pred, 
                                index=data_test.index.values,
                                columns=[f"pred_{col}" for col in output_columns])
    self.lineEdit_Algorithm_name.setText(
        f'当前生成数据: {", ".join(output_columns)} [{N_start_test}:{N_end_test}]')
    return self.data_save

# 使用示例 ---------------------------------------------------
if __name__ == "__main__":
    # 示例数据
    data_train = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target1': np.random.rand(100),
        'target2': np.random.rand(100)
    })
    
    data_test = pd.DataFrame({
        'feature1': np.random.rand(50),
        'feature2': np.random.rand(50),
        'target1': np.random.rand(50),
        'target2': np.random.rand(50)
    })
    
    input_columns = ['feature1', 'feature2']
    output_columns = ['target1', 'target2']
    
    model, scaler, y_test, y_pred, metrics = multi_task_regression_predictor(
        data_train, data_test, input_columns, output_columns, model_type='RF'
    )
    
    print("模型训练完成")
    print("评估指标:", metrics)

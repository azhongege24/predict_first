import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import copy
from sklearn.metrics import r2_score
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView,QMessageBox, QFileDialog
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from tqdm import tqdm
import shutil
import joblib

#带有符号的对数变换
def signed_log1p(arr):
    arr = np.asarray(arr, dtype=np.float64)  # 强制转为 numpy 数组
    return np.sign(arr) * np.log1p(np.abs(arr))

def signed_expm1(arr):
    arr = np.asarray(arr, dtype=np.float64)  # 强制转为 numpy 数组
    return np.sign(arr) * np.expm1(np.abs(arr))

# 配套修改多任务回归预测函数，使其返回每个输出的独立指标
def multi_task_regression_predictor(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    input_columns: list,
    output_columns: list,
    model_type: str = 'RF',
    # 其他参数保持不变...
    scale_features: bool = True,
    random_state: int = 42,
    max_depth: int = 4,
    n_estimators=100,
    kernel='rbf',
    C=1.0,
    epsilon=0.1,
    n_jobs=-1,
    max_iter=500,
    alpha=0.0001,
    mlp_hidden_layers: tuple = (100, 50),
    mmoe_num_experts :int = 5,
    mmoe_expert_hidden :int = 64,
    mmoe_learning_rate : float = 0.001,
    mmoe_dropout_rate : float = 0.1,
    mmoe_epochs : int = 100,
    mmoe_batch_size : int = 32,
    mmoe_lambda_balance : float = 0.1,
):
    # 原有数据加载与预处理逻辑保持不变...
    X_train = data_train[input_columns].values
    y_train = data_train[output_columns].values
    X_test = data_test[input_columns].values
    y_test = data_test[output_columns].values
    
    # 处理output12的代码保持不变...
    if 'output12' in output_columns:
        idx = output_columns.index('output12')
        if y_train.ndim == 1:
            y_train = signed_log1p(y_train)
            y_test = signed_log1p(y_test)
        else:
            y_train[:, idx] = signed_log1p(y_train[:, idx])
            y_test[:, idx] = signed_log1p(y_test[:, idx])
    
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # 模型选择与训练逻辑保持不变...
    model_mapping = {
        'DT': DecisionTreeRegressor(max_depth=max_depth, random_state=random_state),
        'RF': RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs, random_state=random_state),
        'SVM': MultiOutputRegressor(SVR(kernel=kernel, C=C, epsilon=epsilon), n_jobs=n_jobs),
        'MLP': MLPRegressor(hidden_layer_sizes=mlp_hidden_layers, max_iter=max_iter,alpha=alpha, random_state=random_state),
        'ET': ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs, random_state=random_state),
        'MMoE': MMoERegressor(
            input_dim = len(input_columns),
            output_dim = len(output_columns),
            num_experts = mmoe_num_experts,
            expert_hidden = mmoe_expert_hidden,
            learning_rate = mmoe_learning_rate,
            dropout_rate = mmoe_dropout_rate,
            num_epochs = mmoe_epochs,
            batch_size = mmoe_batch_size,
            lambda_balance= mmoe_lambda_balance
        )
    }
    
    if model_type not in model_mapping:
        raise ValueError(f"Unsupported model type: {model_type}. Available options: {list(model_mapping.keys())}")
    
    model = model_mapping[model_type]
    
    # 模型训练逻辑保持不变...
    print("开始训练模型...")
    if model_type == 'MMoE':
        with tqdm(total = mmoe_epochs, desc ="MMoE训练进度", unit="epoch" ) as pbar:
            model.fit(X_train , y_train , verbose = False)
            pbar.update(mmoe_epochs)
    else:
        with tqdm(total=1, desc="训练进度", unit="step") as pbar:
            model.fit(X_train, y_train)
            pbar.update(1)
    
    # 预测逻辑保持不变...
    if model_type == 'MMoE':
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test)  
    
   
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
     # 关键修改：计算每个输出的独立指标
    n_tasks = y_test.shape[1] if y_test.ndim > 1 else 1
    if n_tasks == 1:
        # 单任务情况 - 直接计算指标
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        # 将指标放入列表以保持格式一致
        mse_list = [mse]
        r2_list = [r2]
        rmse_list = [rmse]
        mae_list = [mae]
    else:
        # 多任务情况 - 为每个任务单独计算指标
        mse_list = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(n_tasks)]
        r2_list = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(n_tasks)]
        rmse_list = [np.sqrt(mse) for mse in mse_list]
        mae_list = [mean_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(n_tasks)]
    
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
            if hasattr(model,'__class__') and model.__class__.name == 'MMoERegressor':
                # 保存pytorch模型
                torch.save({
                    'model_state_dict': model.model.state_dict(),
                    'scaler': model.scaler,
                    'input_dim' :model.input_dim,
                    'output_dim': model.output_dim,
                    'num_experts': model.num_experts,
                    'expert_hidden': model.expert_hidden,
                    'loss_history': model.loss_history
                },file_path.replace('.pkl','.pt'))
                if hasattr(parent, "lineEdit_state"):
                    parent.lineEdit_state.setText("模型已保存")
                
            else: 
                joblib.dump(model, file_path)
                if hasattr(parent, "lineEdit_state"):
                    parent.lineEdit_state.setText("模型已保存 " )

#新增加载模型代码功能
def load_model(model_path, model_type=None):
    """
    通用模型加载函数，支持多种模型类型
    
    :param model_path: 模型文件路径
    :param model_type: 模型类型 ('DT', 'RF', 'SVM', 'MLP', 'ET', 'MMoE')，
                      如果为None则尝试自动识别
    :return: 加载的模型对象
    """
    try:
        # 对于scikit-learn的模型，通常用joblib保存
        if model_type in ['DT', 'RF', 'SVM', 'MLP', 'ET'] or model_type is None:
            try:
                # 尝试用joblib加载（scikit-learn模型常用方式）
                model = joblib.load(model_path)
                
                # 验证模型类型是否匹配
                if model_type == 'DT' and not isinstance(model, DecisionTreeRegressor):
                    raise ValueError(f"模型类型不匹配，预期DecisionTreeRegressor，实际是{type(model)}")
                elif model_type == 'RF' and not isinstance(model, RandomForestRegressor):
                    raise ValueError(f"模型类型不匹配，预期RandomForestRegressor，实际是{type(model)}")
                elif model_type == 'SVM' and not isinstance(model, MultiOutputRegressor):
                    raise ValueError(f"模型类型不匹配，预期MultiOutputRegressor(SVR)，实际是{type(model)}")
                elif model_type == 'MLP' and not isinstance(model, MLPRegressor):
                    raise ValueError(f"模型类型不匹配，预期MLPRegressor，实际是{type(model)}")
                elif model_type == 'ET' and not isinstance(model, ExtraTreesRegressor):
                    raise ValueError(f"模型类型不匹配，预期ExtraTreesRegressor，实际是{type(model)}")
                
                print(f"成功加载{model_type if model_type else 'scikit-learn'}模型")
                return model
            except:
                # 如果joblib加载失败，尝试用torch加载（可能是MMoE模型）
                pass
        
        # 对于MMoE模型，用torch加载
        if model_type == 'MMoE' or model_type is None:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            # 创建MMoE模型实例
            model = MMoERegressor(
                input_dim=checkpoint['input_dim'],
                output_dim=checkpoint['output_dim'],
                num_experts=checkpoint['num_experts'],
                expert_hidden=checkpoint['expert_hidden']
            )
            
            # 加载模型状态
            model.model.load_state_dict(checkpoint['model_state_dict'])
            model.scaler = checkpoint.get('scaler')
            model.loss_history = checkpoint.get('loss_history', [])
            
            print("成功加载MMoE模型")
            return model
        
        raise ValueError(f"不支持的模型类型: {model_type}")
        
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return None
#过滤异常值
def filter_extreme(arr, threshold=1e10):
    arr = np.asarray(arr, dtype=np.float64)
    mask = np.isfinite(arr) & (np.abs(arr) < threshold)
    return arr[mask], mask

#单输出画图可视化函数
def single_plot_and_evaluate(self, y_test, y_pred, method, data_test, 
                                output_columns, N_start_test, N_end_test,MSE,RMSE,MAE,R2):
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
                color='#2F5597', linestyle='-', linewidth=2.5, alpha=0.5, solid_capstyle='round', zorder=0)

    # 新增：绘制理想预测线（y=x参考线）
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.6, label='理想预测线')
    
    
    # 优化显示设置
    plt.grid(linestyle='--', alpha=0.5)
    plt.legend(fontsize=10, loc='upper left')
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
    self.lineEdit_RMSE.setText(str(round(RMSE, 5)))  # 假设新增了RMSE控件
    self.lineEdit_MAE.setText(str(round(MAE, 5)))    # 假设新增了MAE控件
    self.lineEdit_R2.setText(str(round(R2, 5)))

    # 保存预测结果到 DataFrame
    self.data_save = pd.DataFrame(y_pred, index=data_test.index.values)
    # self.lineEdit_Algorithm_name.setText(
    #     f'当前生成数据: {output_columns[0]} [ {N_start_test}:{N_end_test} ]'
    # )
    return self.data_save

#新的翻页多输出结果可视化
def Multi_output_plot_and_evaluate(self, y_test, y_pred, method, data_test, 
                                   output_columns, N_start_test, N_end_test, 
                                   MSE_list, RMSE_list, MAE_list, R2_list):
    """
    多输出回归模型的分页可视化函数（支持每个输出显示独立指标）
    
    参数:
        y_test: 测试集真实值 (形状 [N_test, n_outputs])
        y_pred: 测试集预测值 (形状 [N_test, n_outputs])
        method: 当前使用的算法名称
        data_test: 测试集数据
        output_columns: 输出特征列名列表
        N_start_test: 测试集起始索引
        N_end_test: 测试集结束索引
        MSE_list: 各输出变量的MSE列表（长度=n_outputs）
        RMSE_list: 各输出变量的RMSE列表（长度=n_outputs）
        MAE_list: 各输出变量的MAE列表（长度=n_outputs）
        R2_list: 各输出变量的R2列表（长度=n_outputs）
    """
    self.end_time = time.time()
    print(f"运行时间: {self.end_time - self.start_time:.2f} 秒")
    
    # 校验输入维度（避免索引越界）
    n_outputs = len(output_columns)
    if len(MSE_list) != n_outputs or len(R2_list) != n_outputs:
        raise ValueError(f"指标列表长度与输出变量数不匹配: MSE列表长度={len(MSE_list)}, 输出变量数={n_outputs}")
    
    # 创建保存图像的目录
    output_dir = f"MultiOutput_view/{method}_view"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # 清空文件夹内容
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"无法删除文件 {file_path}。原因: {e}")
    
    # 清空之前的图像（保留则注释此行）
    self.figures = []
    
    # 计算整体指标（用于界面显示）
    overall_mse = np.mean(MSE_list)
    overall_rmse = np.mean(RMSE_list) if RMSE_list else np.sqrt(overall_mse)
    overall_mae = np.mean(MAE_list) if MAE_list else np.mean(np.abs(y_test - y_pred))
    overall_r2 = np.mean(R2_list)
    
    # 为每个输出变量创建并保存可视化图像
    for i, col in enumerate(output_columns):
        # 当前输出的独立指标
        current_mse = MSE_list[i]
        current_r2 = R2_list[i]
        current_rmse = RMSE_list[i] if RMSE_list else np.sqrt(current_mse)
        current_mae = MAE_list[i] if MAE_list else np.mean(np.abs(y_test[:, i] - y_pred[:, i]))
        
        # 创建图像
        fig = plt.figure(figsize=(7, 4), dpi=120)
        ax = fig.add_subplot(111)
        
        # 绘制真实值与预测值散点图
        ax.scatter(np.arange(len(y_test[:, i])), y_test[:, i], 
                  c='b', marker='o', s=10, label='真实值', alpha=0.8)
        ax.scatter(np.arange(len(y_test[:, i])), y_pred[:, i], 
                  c='r', marker='X', s=20, label=f'预测值_{method}', alpha=0.8)
        
        # 绘制垂直连接线段（向量方式更高效）
        x = np.arange(len(y_test[:, i]))
        ax.plot([x, x], [y_test[:, i], y_pred[:, i]],
                color='#2F5597', linestyle='-', linewidth=1.5, alpha=0.6, zorder=0)
        
        # 设置子图标题和标签（显示当前输出的独立指标）
        ax.set_title(
            f'输出变量: {col}\n'
            f'MSE: {current_mse:.4f}, RMSE: {current_rmse:.4f}\n'
            f'MAE: {current_mae:.4f}, R: {current_r2:.4f}',
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
        
        # 保存图像（自动调整布局避免标签截断）
        fig_path = os.path.join(output_dir, f"output_{col}.png")
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)
        
        # 将图像路径添加到列表中
        self.figures.append(fig_path)
    
    # 初始化分页显示
    self.current_page = 0
    self.update_graphics_view()
    
    # 更新界面控件（显示整体平均指标）
    self.lineEdit_state.setText('Finish!')
    self.lineEdit_MSE.setText(f"{overall_mse:.5f}")
    self.lineEdit_RMSE.setText(f"{overall_rmse:.5f}")
    self.lineEdit_MAE.setText(f"{overall_mae:.5f}")
    self.lineEdit_R2.setText(f"{overall_r2:.5f}")
    
    # 保存预测结果到 DataFrame（包含真实值便于对比）
    pred_df = pd.DataFrame(
        y_pred, 
        index=data_test.index.values,
        columns=[f"pred_{col}" for col in output_columns]
    )
    true_df = pd.DataFrame(
        y_test,
        index=data_test.index.values,
        columns=[f"true_{col}" for col in output_columns]
    )
    self.data_save = pd.concat([true_df, pred_df], axis=1)
    
    return self.data_save
#新增的MMOE多专家混合系统
class Expert(nn.Module):
    def __init__(self, input_dim ,hidden_dim, dropout_rate = 0.1):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            )
    def forward(self,x):
        return self.net(x)
class MMoE(nn.Module):
    def __init__(self, input_dim,output_dim,num_experts=5,num_tasks=1,
                 expert_hidden=64, dropout_rate =0.1):
        super(MMoE, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.input_dim = input_dim
        # self.out_dim = output_dim
        #专家网络
        self.experts = nn.ModuleList([
            Expert(input_dim,expert_hidden,dropout_rate) for _ in range(num_experts)
            ])
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim,num_experts),
                nn.Softmax(dim=1)
            ) for _ in range(num_tasks)
        ])
        # self.task_layers = nn.ModuleList([
        #     nn.Linear(expert_hidden,output_dim) for _ in range(num_tasks)
        # ])
        self.task_layers = nn.ModuleList([
            nn.Linear(expert_hidden, 1)  # 关键修改：输出维度固定为1
            for _ in range(num_tasks)
        ])
    
    def forward(self,x):
        #计算所有专家的输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts],dim = 1) # [batch, experts, hidden]
        #计算每个任务的，门控权重和最终输出
        outputs =[]
        cv_losses=[]
        for i in range(self.num_tasks):
            #计算门控权重
            gate_weights = self.gates[i](x) #[batch, experts]
            #计算加权专家输出
            weighted_expert =(expert_outputs * gate_weights.unsqueeze(-1)).sum(dim = 1) #[batch, hidden]
            #通过特定任务层
            task_output = self.task_layers[i](weighted_expert)#[batch, output_dim]
            outputs.append(task_output)
            #计算专家使用的变异系数损失
            importance = gate_weights.mean(dim = 0) #[experts]
            cv_loss = torch.std(importance)/(torch.mean(importance)+1e-8)
            cv_losses.append(cv_loss)
        
        # 对于单任务，直接返回输出和损失
        if self.num_tasks == 1:
            return outputs[0],cv_losses[0]
        else:
            # return outputs,cv_losses
            
             # 将输出列表拼接为张量 (batch_size, num_tasks, output_dim)
            outputs_tensor = torch.cat(outputs, dim=1)  # 假设output_dim=1，拼接后为(batch, num_tasks)
            # 将损失列表转换为张量并求和（或取平均）
            cv_losses_tensor = torch.stack(cv_losses).mean()  # 多任务损失取平均
            return outputs_tensor, cv_losses_tensor

class MMoERegressor:
    def __init__(self,input_dim,output_dim,num_experts = 5,expert_hidden = 64,
                 learning_rate = 0.001, dropout_rate = 0.1,
                 num_epochs = 100, batch_size =32,device = None,
                 lambda_balance = 0.1
                 ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.expert_hidden = expert_hidden
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device if device else ('cuda' if torch.cuda.is_available()else 'cpu')
        self.lambda_balance = lambda_balance
        self.model = None
        self.scaler = None
        self.loss_history = []
    
    def fit(self,X,y, verbose = True):
        #数据预处理
        if self.scaler is None:
            self.scaler = StandardScaler()
            x_scaled = self.scaler.fit_transform(X)
        else:
            x_scaled = self.scaler.transform(X)
    
        #转换为pytorch张量
        X_tensor = torch.FloatTensor(x_scaled).to(self.device)
        #处理y的形状
        if len(y.shape) ==1:
            y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
            num_tasks = 1
        else:
            y_tensor = torch.FloatTensor(y).to(self.device)
            num_tasks = y.shape[1]
        #创建数据加载器
        dataset = torch.utils.data.TensorDataset(X_tensor,y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=self.batch_size,shuffle=True)
        #初始化模型
        self.model = MMoE(input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_experts=self.num_experts,
            num_tasks=num_tasks,
            expert_hidden=self.expert_hidden,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        #定义优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(),lr = self.learning_rate)
        criterion = nn.MSELoss()
        
        # 训练循环
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch_X,batch_y in dataloader:
                optimizer.zero_grad()
                # 强制转换为Tensor，确保类型正确
                batch_X = batch_X if isinstance(batch_X, torch.Tensor) else torch.FloatTensor(batch_X).to(self.device)
                batch_y = batch_y if isinstance(batch_y, torch.Tensor) else torch.FloatTensor(batch_y).to(self.device)
                #前向传播
                predictions, cv_losses = self.model(batch_X)
                #计算损失（主损失和平衡损失）
                main_loss = criterion(predictions,batch_y)
                balance_loss = cv_losses *self.lambda_balance
                total_loss = main_loss + balance_loss
                #反向传播
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            avg_loss = epoch_loss/len(dataloader)
            self.loss_history.append(avg_loss)
            if verbose and (epoch + 1)%10 == 0:
                print(f'Epoch {epoch +1}/{self.num_epochs}, Loss:{avg_loss:.4f}')
        return self
    
    def predict(self,X):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        #数据预处理
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        #预测
        self.model.eval()
        with torch.no_grad():
            predictions, _ = self.model(X_tensor)
            return predictions.cpu().numpy()
    
    def get_loss_history(self):
        #返回训练历史损失
        return self.loss_history
    
    
    
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

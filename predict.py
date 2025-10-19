import subprocess
import time
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
import numpy as np
from ui2025 import Ui_MainWindow #我新创建的界面类
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton,QMessageBox, QFileDialog, QGraphicsScene, QGraphicsView, QWidget, QCheckBox, QListWidgetItem
from PyQt5.QtCore import Qt ,QSettings,QTimer
from PyQt5.QtGui import QPixmap
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
import os
import scipy.io as sio
import torch
import joblib
from ALL_Algorithms.VA_data_handle import preview_VAdata
from ALL_Algorithms.VA_data_handle import analyze_VA_psd
from ALL_Algorithms.VA_data_handle import save_psd_result_util
from ALL_Algorithms.VA_para import Ui_VA_para
from ALL_Algorithms.Load_model_para import Ui_Load_model_para
from ALL_Algorithms.Load_pretrained_model import select_pretrained_model_path
from ALL_Algorithms.algorithms1_DT_para import Ui_DT_para
from ALL_Algorithms.algorithms2_RF_para import Ui_RF_para
from ALL_Algorithms.algorithms3_SVM_para import Ui_SVM_para   
from ALL_Algorithms.algorithms4_MLP_para import Ui_MLP_para   
from ALL_Algorithms.algorithms5_ET_para import Ui_ET_para 
from ALL_Algorithms.algorithms6_GL_para import Ui_GL_para  
from ALL_Algorithms.algorithms7_MTW_para import Ui_MTW_para
from ALL_Algorithms.algorithms8_REMTW_para import Ui_REMTW_para
from ALL_Algorithms.algorithms9_MMoE_para import Ui_MMoE_para
from ALL_Algorithms.algorithms10_GP_para import Ui_GP_para
from ALL_Algorithms.Dataset_handle import Ui_dataset_handle
from PyQt5.QtCore import pyqtSlot
from ALL_Algorithms.Algorithms import multi_task_regression_predictor
from ALL_Algorithms.Algorithms import ask_and_save_model
from ALL_Algorithms.Algorithms import single_plot_and_evaluate
from ALL_Algorithms.Algorithms import Multi_output_plot_and_evaluate
from ALL_Algorithms.Group_Lasso import group_lasso_predictor
from ALL_Algorithms.Group_Lasso import group_lasso_plot_and_evaluate
from ALL_Algorithms.MTW import MTW_Lasso
from ALL_Algorithms.MTW import mtw_plot_and_evaluate
from ALL_Algorithms.ReMTW import REMTW_Lasso
from ALL_Algorithms.ReMTW import remtw_plot_and_evaluate
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
global max_depth, random_state,n_estimators,kernel, C, epsilon,scale_features
global hidden_layer_sizes, max_iter,method,n_jobs,alpha,beta,tol
global mmoe_num_experts,mmoe_expert_hidden,mmoe_learning_rate,mmoe_dropout_rate
global mmoe_epochs,mmoe_batch_size,mmoe_lambda_balance,mmoe_scale_features
method = 'NONE'  # 初始化方法为NONE
# 读取输入参数
VA_data_path = ""  # 设置一个默认值
class POP_VA_para(QMainWindow, Ui_VA_para, Ui_MainWindow):
    def __init__(self, parent=None):
        super(POP_VA_para, self).__init__()
        self.setupUi(self)
        self.parent_window = parent#保存主窗口的引用
        self.lastSelectedPath = ""
    
    def Confirm(self):
         # 读取输入参数
        global sampling_rate,VA_data_path,VA_outpath,num_groups
        sampling_rate = int(self.spinBox_sampling_rate.text())
        VA_data_path = str(self.lineEdit_VA_data_path.text())
        if self.parent_window:
            self.parent_window.VA_data_path = VA_data_path  # 将路径传递给主窗口
            self.parent_window.lineEdit_Algorithm_name.setText("Vibration Analysis")
            self.parent_window.lineEdit_state.setText("正在进行振动分析")
            self.parent_window.preview_VA_data()  # 预览数据
        
        print("sampling_rate:", sampling_rate) 
        print("VA_data_path:", VA_data_path)

    
    def open_VA_folder(self):
        '''新内容取消的时候不会改变linedit'''
        file_filter = "Data Files (*.csv *.xls *.xlsx *.mat);;Excel Files (*.xls *.xlsx);;MATLAB Files (*.mat);;CSV Files (*.csv);;All Files(*.*)"
        initial_dir = self.lastSelectedPath if self.lastSelectedPath else "data/"
        self.fileName, _ = QFileDialog.getOpenFileName(
            self, "选取文件", initial_dir, file_filter
        )

        if not self.fileName:  # 用户取消选择
            return  # 直接返回，不改变lineEdit的内容

        # 更新上次路径并显示到lineEdit
        self.lastSelectedPath = self.fileName
        self.lineEdit_VA_data_path.setText(self.fileName)
        print("选择的文件:", self.fileName)
        if not os.path.isfile(self.fileName):
            self.lineEdit_VA_data_path.setText("File path doesn't exist")
            return

    def save_VA_folder(self):
        '''新内容取消的时候不会改变lineEdit'''
        # 将初始目录设置为 data 文件夹
        initial_dir = os.path.join(os.getcwd(), "data")  # 获取当前工作目录下的 data 文件夹路径
        if not os.path.exists(initial_dir):  # 如果 data 文件夹不存在，则创建
            os.makedirs(initial_dir)

        self.fileName = QFileDialog.getExistingDirectory(
            self, "选取文件夹", initial_dir
        )

        if not self.fileName:  # 用户取消选择
            return 
        # 更新上次路径并显示到 lineEdit
        self.lastSelectedPath = self.fileName
        self.lineEdit_VA_outputpath.setText(self.fileName)
        print("选择输出的文件夹:", self.fileName)

class POP_Load_model_para(QMainWindow, Ui_Load_model_para, Ui_MainWindow):
    def __init__(self, parent=None):
        super(POP_Load_model_para, self).__init__()
        self.setupUi(self)
        self.parent_window = parent
        self.selected_model_path = None
        self.predict_data = None
        self.lastSelectedPath = ""
    def Confirm(self):
        global loaded_model_path
        if self.parent_window:
            self.parent_window.lineEdit_Algorithm_name.setText("已加载预训练模型")
        loaded_model_path = self.selected_model_path  # 这里只存路径
        self.parent_window.predict_with_loaded_model()
        print("selected_model_path:", loaded_model_path)

    def load_pretrained_model(self):
        # 只选择路径，不加载模型
        path = select_pretrained_model_path(self)
        self.selected_model_path = path
        print("Selected model path:", self.selected_model_path)
        if self.parent_window is not None:
            self.parent_window.selected_model_path = self.selected_model_path
            
    def load_predict_data(self):
        file_filter = "Data Files (*.csv *.xls *.xlsx *.mat);;Excel Files (*.xls *.xlsx);;MATLAB Files (*.mat);;CSV Files (*.csv);;All Files(*.*)"
        initial_dir = self.lastSelectedPath if self.lastSelectedPath else "data/"
        file_path, _ = QFileDialog.getOpenFileName(self, "选择需要预测的新数据文件", initial_dir, file_filter)
        if not file_path:
            return
        try:
            if file_path.endswith(('.xls', '.xlsx')):
                self.predict_data = pd.read_excel(file_path)
            elif file_path.endswith('.mat'):
                mat_data = sio.loadmat(file_path)
                all_keys = list(mat_data.keys())
                feature_keys = all_keys[3:]
                df = pd.DataFrame({
                    key: mat_data[key].squeeze()  # 压缩单维度，如 (n,1)→n
                    for key in feature_keys
                })
                columns = df.columns.tolist()

                new_columns = columns[1:] + [columns[0]]

                self.predict_data = df[new_columns]

            else:  # 默认处理CSV
                self.predict_data = pd.read_csv(file_path,encoding='utf-8')
            #公共数据处理流程   
            self.predict_data = self.predict_data.dropna()
            self.predict_data = self.predict_data[self.spinBox_predict_start.value():self.spinBox_predict_end.value()]
            self.predict_data.columns = self.predict_data.columns.astype(str)
            self.predict_columns = self.predict_data.columns.tolist()#确保是字符串
            self.predict_input_columns = [col for col in self.predict_columns if"input" in col.lower()]
            self.predict_output_columns = [col for col in self.predict_columns if"output" in col.lower()]
            
            if not self.predict_input_columns:
                print("数据中未找到以'input'命名的输入特征列")
            if not self.predict_output_columns:
                print("数据中未找到以'output'命名的输出特征列")
            #显示数据信息
            if self.parent_window:
                self.parent_window.listWidget_inputfeature.clear()
                self.parent_window.listWidget_outputfeature.clear()
                self.parent_window.add_listitem(self.predict_input_columns, self.parent_window.listWidget_inputfeature)
                self.parent_window.add_listitem(self.predict_output_columns, self.parent_window.listWidget_outputfeature)
                
                self.parent_window.predict_data = self.predict_data
                self.parent_window.predict_input_cols = self.predict_input_columns  # 输入特征列
                self.parent_window.predict_output_cols = self.predict_output_columns  # 真实值列
                
                self.shape = self.predict_data.shape
                self.parent_window.lineEdit_dataset_nums.setText(f'({self.shape[0]} Samples * {self.shape[1]} Features)')
                self.parent_window.lineEdit_state.setText("新数据加载成功（包含真实值，可进行评估）")
                
                self.spinBox_predict_end.setValue(self.shape[0]*0.9)
                self.parent_window.data_load = 1
            print(len(self.predict_output_columns))
            print("新数据 shape:", self.predict_data.shape)
            print("输入特征列:", self.predict_input_columns)
            print("真实值列:", self.predict_output_columns)
        except Exception as e:
            QMessageBox.warning(self, "加载失败", str(e))


class POP_DatasetHandleWindow(QMainWindow, Ui_dataset_handle, Ui_MainWindow):
    def __init__(self, parent=None):
        super(POP_DatasetHandleWindow, self).__init__()
        self.setupUi(self)
        self.datasets = []         # 存储每次试验的DataFrame
        self.merged_data = None    # 合并后的总数据集
        self.parent_window = parent
        
        self.input_files = []      # 存储输入特征文件路径
        self.output_files = []     # 存储输出特征文件路径
        self.lastSelectedPath = None  # 上次选择的路径

        # 绑定按钮（确保UI控件名与代码一致）
        self.pushButton_add_input_features.clicked.connect(self.add_input_features)
        self.pushButton_add_output_features.clicked.connect(self.add_output_features)
        self.pushButton_save_file.clicked.connect(self.save_combined_data)
        self.pushButton_clear_file_lists.clicked.connect(self.clear_file_lists)
        
        # 初始化列表控件
        self.listWidget_input_files.clear()
        self.listWidget_output_files.clear()

    def add_input_features(self):
        """添加输入特征文件（支持多选，空格分隔的.txt）"""
        file_filter = "文本文件 (*.txt);;所有文件 (*.*)"
        initial_dir = self.lastSelectedPath if self.lastSelectedPath else "data/"
        files, _ = QFileDialog.getOpenFileNames(self, "选择输入特征文件", initial_dir, file_filter)
        
        if files:
            self.lastSelectedPath = os.path.dirname(files[0])
            for file in files:
                if file not in self.input_files:
                    self.input_files.append(file)
                    self.listWidget_input_files.addItem(os.path.basename(file))
        
        # 更新状态显示
        self.lineEdit_status.setText(f"已加载 {len(self.input_files)} 个输入特征文件")
        self.update_combined_info()

    def add_output_features(self):
        """添加输出特征文件（支持多选，空格分隔的多列.txt）"""
        file_filter = "文本文件 (*.txt);;所有文件 (*.*)"
        initial_dir = self.lastSelectedPath if self.lastSelectedPath else "data/"
        files, _ = QFileDialog.getOpenFileNames(self, "选择输出特征文件", initial_dir, file_filter)
        
        if files:
            self.lastSelectedPath = os.path.dirname(files[0])
            for file in files:
                if file not in self.output_files:
                    self.output_files.append(file)
                    self.listWidget_output_files.addItem(os.path.basename(file))
        
        # 更新状态显示
        self.lineEdit_status_2.setText(f"已加载 {len(self.output_files)} 个输出特征文件")
        self.update_combined_info()

    def update_combined_info(self):
        """更新合并信息（修正输出特征列数计算）"""
        if self.input_files and self.output_files:
            try:
                # 读取输入文件（指定空格分隔，避免多列解析错误）
                sample_df = pd.read_csv(self.input_files[0], header=None, sep="\s+")
                sample_count = len(sample_df)
                input_count = len(self.input_files)
                
                # 计算所有输出文件的总列数（修正列表推导式求和）
                output_count = sum([pd.read_csv(f, header=None, sep="\s+").shape[1] for f in self.output_files])
                
                self.lineEdit_info.setText(
                    f"预计合并结果: {sample_count} 样本 × {input_count + output_count} 特征 "
                    f"(输入: {input_count}, 输出: {output_count})"
                )
            except Exception as e:
                self.lineEdit_info.setText(f"预览失败：{str(e)[:50]}...")
        else:
            self.lineEdit_info.setText("请选择输入和输出特征文件")

    def combine_data(self):
        """合并输入输出数据（核心修复：补全列表添加、分隔符、行数校验）"""
        if not self.input_files or not self.output_files:
            QMessageBox.warning(self, "警告", "请先选择输入和输出特征文件！")
            return None
        
        try:
            # ---------------------- 1. 读取输入文件（修复：添加到input_dfs列表）----------------------
            input_dfs = []
            for i, file_path in enumerate(self.input_files, 1):
                # 关键：指定sep="\s+"（匹配任意空格），避免多列数据解析为1列
                df = pd.read_csv(file_path, header=None, sep="\s+")
                # 若输入文件是多列，仅取第一列（按原逻辑保留，可根据需求调整）
                if df.shape[1] > 1:
                    df = df.iloc[:, 0]
                df.name = f"input{i}"
                input_dfs.append(df)  # 修复：将处理后的输入数据加入列表

            # ---------------------- 2. 读取输出文件（修复：加入all_dfs校验行数）----------------------
            output_dfs = []
            output_counter = 1
            for file_path in self.output_files:
                # 关键：指定sep="\s+"，正确解析多列输出
                df = pd.read_csv(file_path, header=None, sep="\s+")
                # 拆分多列为单独特征（如output1、output2...）
                for col_idx in range(df.shape[1]):
                    col_df = df.iloc[:, col_idx].reset_index(drop=True)  # 重置索引避免行数错位
                    col_df.name = f"output{output_counter}"
                    output_dfs.append(col_df)
                    output_counter += 1

            # ---------------------- 3. 校验所有数据行数一致性（修复：包含输出文件）----------------------
            all_dfs = input_dfs + output_dfs  # 修复：加入输出数据参与行数校验
            row_counts = [len(df) for df in all_dfs]
            
            if len(set(row_counts)) > 1:
                min_rows = min(row_counts)
                # 截取到最小行数（避免索引越界）
                input_dfs = [df.head(min_rows).reset_index(drop=True) for df in input_dfs]
                output_dfs = [df.head(min_rows).reset_index(drop=True) for df in output_dfs]
                warning_msg = f"警告: 文件行数不一致，已截取到最小行数: {min_rows}"
                self.lineEdit_info.setText(warning_msg)

            # ---------------------- 4. 合并数据（确保索引对齐）----------------------
            combined_df = pd.DataFrame()
            # 添加输入特征
            for df in input_dfs:
                combined_df[df.name] = df
            # 添加输出特征
            for df in output_dfs:
                combined_df[df.name] = df

            return combined_df

        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据合并失败:\n{str(e)}")
            return None

    def save_combined_data(self):
        """保存合并数据（增加异常值剔除功能）"""
        combined_df = self.combine_data()
        if combined_df is None:
            return

        # ----------- 异常值剔除部分 -----------
        # 1. 剔除包含 #NAME? 的行
        combined_df = combined_df[~combined_df.apply(lambda row: row.astype(str).str.contains('#NAME\?').any(), axis=1)]
        # 2. 剔除 inf/-inf/NaN 行
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.dropna(axis=0, how='any')

        # 选择保存路径
        file_filter = "CSV文件 (*.csv);;Excel文件 (*.xlsx);;所有文件 (*.*)"
        initial_dir = self.lastSelectedPath if self.lastSelectedPath else "data/"
        save_path, _ = QFileDialog.getSaveFileName(self, "保存合并数据", initial_dir, file_filter)

        if not save_path:
            return

        try:
            if save_path.endswith(('.xls', '.xlsx')):
                combined_df.to_excel(save_path, index=False, engine="openpyxl")
            else:
                combined_df.to_csv(save_path, index=False, float_format="%.6f")

            self.lastSelectedPath = os.path.dirname(save_path)

            output_total_cols = sum([pd.read_csv(f, header=None, sep="\s+").shape[1] for f in self.output_files])
            success_msg = (
                f"数据保存成功!\n"
                f"文件: {os.path.basename(save_path)}\n"
                f"数据形状: {combined_df.shape}\n"
                f"输入特征: {len(self.input_files)} 列\n"
                f"输出特征: {output_total_cols} 列"
            )
            QMessageBox.information(self, "成功", success_msg)
            self.lineEdit_info.setText(f"数据已保存: {os.path.basename(save_path)}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"文件保存失败:\n{str(e)}")

    def clear_file_lists(self):
        """清空文件列表"""
        self.input_files.clear()
        self.output_files.clear()
        self.listWidget_input_files.clear()
        self.listWidget_output_files.clear()
        self.lineEdit_info.setText("请选择输入和输出特征文件")
        self.lineEdit_status.setText("就绪")
        self.lineEdit_status_2.setText("就绪")   
                           
class POP_DT_para(QMainWindow, Ui_DT_para, Ui_MainWindow):
    def __init__(self, parent=None):
        super(POP_DT_para, self).__init__()
        self.setupUi(self)
        self.parent_window = parent#保存主窗口的引用
    
    def Confirm(self):
         # 读取输入参数
        global max_depth, random_state,method,scale_features
        scale_features = self.comboBox_scale_features.currentText()=="True"
        max_depth = self.spinBox_max_depth.text()
        random_state = self.spinBox_random_state.text()
        method = 'DT'
        if self.parent_window:
            self.parent_window.lineEdit_Algorithm_name.setText("Decision Tree")
        self.parent_window.All_Methods_Begin()
        print("max_depth:", max_depth) 
        print("random_state:", random_state)
        print("scale_features:", scale_features)
        
class POP_RF_para(QMainWindow, Ui_RF_para, Ui_MainWindow):
    def __init__(self,parent=None):
        super(POP_RF_para, self).__init__()
        self.setupUi(self)
        self.parent_window = parent#保存主窗口的引用
    def Confirm(self):
        # 读取输入参数
        global max_depth, random_state,n_estimators,method,scale_features
        scale_features = self.comboBox_scale_features.currentText()=="True"
        max_depth = self.spinBox_max_depth.text()
        random_state = self.spinBox_random_state.text()
        n_estimators = self.spinBox_n_estimators.text()
        method = 'RF'
        if self.parent_window:
            self.parent_window.lineEdit_Algorithm_name.setText("Random Forest")
        self.parent_window.All_Methods_Begin()
        print("n_estimators:", n_estimators)
        print("max_depth:", max_depth)
        print("random_state:", random_state)
        print("scale_features:", scale_features)

class POP_SVM_para(QMainWindow, Ui_SVM_para, Ui_MainWindow):
    def __init__(self,parent=None):
        super(POP_SVM_para, self).__init__()
        self.setupUi(self)
        self.parent_window = parent#保存主窗口的引用
    def Confirm(self):
        # 读取输入参数
        global kernel, C, epsilon,n_jobs,method,random_state,scale_features
        kernel = self.comboBox_kernel.currentText()
        C = self.spinBox_C.text()
        epsilon = self.doubleSpinBox_epsilon.text()
        n_jobs = self.spinBox_n_jobs.text()
        random_state = self.spinBox_random_state.text()
        scale_features = self.comboBox_scale_features.currentText()=="True"
        method = 'SVM'
        if self.parent_window:
            self.parent_window.lineEdit_Algorithm_name.setText("Support Vector Machine")
        self.parent_window.All_Methods_Begin()
        print("kernel:", kernel)
        print("C:", C)
        print("epsilon:", epsilon)
        print("n_jobs:", n_jobs)
        print("random_state:", random_state)
        print("scale_features:", scale_features)
         
class POP_MLP_para(QMainWindow, Ui_MLP_para, Ui_MainWindow):
    def __init__(self,parent=None):
        super(POP_MLP_para, self).__init__()
        self.setupUi(self)
        self.parent_window = parent#保存主窗口的引用
    
    def Confirm(self):
         # 读取输入参数
        global hidden_layer_sizes, max_iter,random_state,method,scale_features, mlp_alpha
        input_text = self.lineEdit_hidden_layer_sizes.text()
        hidden_layer_sizes = tuple(map(int, input_text.split(','))) # 解析为元组
        random_state = self.spinBox_random_state.text()
        scale_features = self.comboBox_scale_features.currentText()=="True"
        method = 'MLP'
        mlp_alpha = self.doubleSpinBox_mlp_alpha.text()  # 获取MLP的alpha参数
        if self.parent_window:
            self.parent_window.lineEdit_Algorithm_name.setText("Multi-layer Perceptron")
        max_iter = self.spinBox_max_iter.text()
        self.parent_window.All_Methods_Begin()
        print("random_state:", random_state)
        print("hidden_layer_sizes:", hidden_layer_sizes)
        print("max_iter:", max_iter)   
        print("scale_features:", scale_features) 

class POP_ET_para(QMainWindow, Ui_ET_para, Ui_MainWindow):
    def __init__(self,parent=None):
        super(POP_ET_para, self).__init__()
        self.setupUi(self)  
        self.parent_window = parent#保存主窗口的引用
    
    def Confirm(self):
        # 读取输入参数
        global n_estimators,max_depth,n_jobs,random_state,method,scale_features
        n_estimators = self.spinBox_n_estimators.text()
        max_depth = self.spinBox_max_depth.text()
        n_jobs = self.spinBox_n_jobs.text()
        random_state = self.spinBox_random_state.text()
        scale_features = self.comboBox_scale_features.currentText()=="True"
        method = 'ET'   
        if self.parent_window:
            self.parent_window.lineEdit_Algorithm_name.setText("Extra Trees")   
        self.parent_window.All_Methods_Begin()
        print("n_estimators:", n_estimators)
        print("max_depth:", max_depth)  
        print("n_jobs:", n_jobs)
        print("random_state:", random_state)
        print("scale_features:", scale_features)

class POP_GL_para(QMainWindow, Ui_GL_para, Ui_MainWindow):
    def __init__(self, parent=None):
        super(POP_GL_para, self).__init__()
        self.setupUi(self)
        self.parent_window = parent#保存主窗口的引用
    
    def Confirm(self):
         # 读取输入参数
        global  random_state,alpha,method,max_iter,tol
        alpha = self.doubleSpinBox_alpha.text()
        max_iter = self.spinBox_max_iter.text()
        tol = self.doubleSpinBox_tol.text()
        random_state = self.spinBox_random_state.text()
        method = 'GL'
        if self.parent_window:
            self.parent_window.lineEdit_Algorithm_name.setText("Group Lasso")
        self.parent_window.All_Methods_Begin()
        print("alpha:", alpha) 
        print("random_state:", random_state)
        print("max_iter:", max_iter)
        print("tol:", tol)

class POP_MTW_para(QMainWindow, Ui_MTW_para, Ui_MainWindow):#mtw算法改
    def __init__(self,parent=None):
        super(POP_MTW_para, self).__init__()
        self.setupUi(self)
        self.parent_window = parent#保存主窗口的引用
    
    def Confirm(self):
         # 读取输入参数
        global alpha,beta,random_state,method,max_iter,tol
        
        alpha = self.doubleSpinBox_alpha.text()
        beta = self.doubleSpinBox_beta.text()
        random_state = self.spinBox_random_state.text()
        max_iter = self.spinBox_max_iter.text()
        tol = self.doubleSpinBox_tol.text()
        method = 'MTW'
        if self.parent_window:
            self.parent_window.lineEdit_Algorithm_name.setText("Multitask Wasserstein ")
        self.parent_window.All_Methods_Begin()
        print("random_state:", random_state)
        print("alpha:", alpha)
        print("beta:", beta)
        print("max_iter:", max_iter)
        print("tol:", tol)

class POP_REMTW_para(QMainWindow, Ui_REMTW_para, Ui_MainWindow):#remtw算法改
    def __init__(self,parent=None):
        super(POP_REMTW_para, self).__init__()
        self.setupUi(self)
        self.parent_window = parent#保存主窗口的引用
    
    def Confirm(self):
         # 读取输入参数
        global alpha,beta,random_state,method,tol,max_iter
        
        alpha = self.doubleSpinBox_alpha.text()
        beta = self.doubleSpinBox_beta.text()
        tol = self.doubleSpinBox_tol.text()
        max_iter = self.spinBox_max_iter.text()
        random_state = self.spinBox_random_state.text()
        method = 'REMTW'
        if self.parent_window:
            self.parent_window.lineEdit_Algorithm_name.setText("Reweighted Multitask Wasserstein ")
        self.parent_window.All_Methods_Begin()
        print("random_state:", random_state)
        print("alpha:", alpha)
        print("beta:", beta)
        print("max_iter:", max_iter)
        print("tol:", tol)
     
class POP_MMoE_para(QMainWindow, Ui_MMoE_para, Ui_MainWindow):#remtw算法改
    def __init__(self,parent=None):
        super(POP_MMoE_para, self).__init__()
        self.setupUi(self)
        self.parent_window = parent#保存主窗口的引用
    
    def Confirm(self):
         # 读取输入参数
        global alpha,beta,random_state,method,tol,max_iter,mmoe_num_experts,mmoe_expert_hidden
        global mmoe_learning_rate,mmoe_dropout_rate,mmoe_epochs,mmoe_batch_size,mmoe_lambda_balance,mmoe_scale_features
        alpha = self.doubleSpinBox_alpha.text()
        max_iter = self.spinBox_max_iter.text()
        random_state = self.spinBox_random_state.text()
        mmoe_num_experts = self.spinBox_mmoe_num_experts.text()
        mmoe_expert_hidden = self.spinBox_mmoe_expert_hidden.text()  # 专家网络隐藏层大小（整数）
        mmoe_learning_rate = self.doubleSpinBox_mmoe_learning_rate.text()  # 学习率（浮点数）
        mmoe_dropout_rate = self.doubleSpinBox_mmoe_dropout_rate.text()  # Dropout率（浮点数）
        mmoe_epochs = self.spinBox_mmoe_epochs.text()  # 训练轮数（整数）
        mmoe_batch_size = self.spinBox_mmoe_batch_size.text()  # 批处理大小（整数）
        mmoe_lambda_balance = self.doubleSpinBox_mmoe_lambda_balance.text()  # 平衡系数（浮点数）
        mmoe_scale_features = self.comboBox_mmoe_scale_features.currentText()=="True"
        method = 'MMoE'
        if self.parent_window:
            self.parent_window.lineEdit_Algorithm_name.setText("Multi-gate Mixture-of-Experts")
        self.parent_window.All_Methods_Begin()
        # 打印所有参数值，方便调试
        print(f"算法名称: {method}")
        print(f"alpha: {alpha}")
        print(f"max_iter: {max_iter}")
        print(f"random_state: {random_state}")
        print(f"专家数量(mmoe_num_experts): {mmoe_num_experts}")
        print(f"专家网络隐藏层大小(mmoe_expert_hidden): {mmoe_expert_hidden}")
        print(f"学习率(mmoe_learning_rate): {mmoe_learning_rate}")
        print(f"Dropout率(mmoe_dropout_rate): {mmoe_dropout_rate}")
        print(f"训练轮数(mmoe_epochs): {mmoe_epochs}")
        print(f"批处理大小(mmoe_batch_size): {mmoe_batch_size}")
        print(f"平衡系数(mmoe_lambda_balance): {mmoe_lambda_balance}")
        print(f"是否标准化输入特征(mmoe_scale_features): {mmoe_scale_features}")

class POP_GP_para(QMainWindow, Ui_GP_para, Ui_MainWindow): 
    def __init__(self,parent=None):
        super(POP_GP_para, self).__init__()
        self.setupUi(self)
        self.parent_window = parent#保存主窗口的引用
    def Confirm(self):
        global learning_rate,training_iterations,method
        learning_rate = self.doubleSpinBox_learning_rate.text()
        training_iterations = self.spinBox_training_iterations.text()
        method = 'GP'
        if self.parent_window:
            self.parent_window.lineEdit_Algorithm_name.setText("GassuProcessing")
        self.parent_window.All_Methods_Begin()
        
   
class MyMainWindow(QMainWindow, Ui_MainWindow):  # 继承 QMainWindow类和 Ui_MainWindow界面类
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)  # 初始化父类
        self.setupUi(self)  # 继承 Ui_MainWindow 界面类
        self.fileName = ''
        self.new_model = 0
        self.data_load = 0
        self.predict_data = None
        self.predict_input_cols = []
        self.predict_output_cols = []
        self.graphicscene = QGraphicsScene()
        self.lastSelectedPath = ""
        self.method = 'NONE'
        self.psd_results = None
        self.selected_model_path = None
        self.trained_model = None  # 用于存储训练好的模型

        self.current_page = 0
        self.figures = []  # 存储所有图表的列表
        # 在UI初始化代码中添加


        # 初始化 VibrationAnalyzer 为 None，等待用户输入采样率后再实例化
        self.VA_data_path = ""



        # 连接按钮信号
        
        self.pushButton_top.clicked.connect(self.show_previous_page)
        self.pushButton_bottom.clicked.connect(self.show_next_page)
        # self.pushButton_preview_VA_data.clicked.connect(self.preview_VA_data)  这个现在没用了，直接导入振动数据文件的时候，确认便就会预览
        self.pushButton_psd_analysis.clicked.connect(self.handle_vibration_analysis)
        self.pushButton_save_psd.clicked.connect(self.save_psd_result)
        self.pushButton_dataset.clicked.connect(self.AL_dataset_handle)  # 打开数据集处理窗口
        self.pushButton_save_pretrained_model.clicked.connect(
                    lambda: self.ask_and_save_model(self.trained_model, method)
                ) # 保存预训练模型

    
    def ask_and_save_model(self, trained_model, method):
        try:
            # 定义文件过滤器
            file_filter = "模型文件 (*.pkl);;所有文件 (*.*)"
            
            # 获取初始目录（如果存在）
            base_dir = self.lastSelectedPath if hasattr(self, 'lastSelectedPath') else "data/"
            
            # 构造默认文件名：method + .pkl
            default_filename = f"{method}.pkl"
            # 拼接完整的初始文件路径（目录+文件名）
            initial_file_path = os.path.join(base_dir, default_filename)
            
            # 打开保存对话框，设置默认文件名
            save_path, _ = QFileDialog.getSaveFileName(
                self, 
                "保存预训练模型", 
                initial_file_path,  # 这里传入包含默认文件名的路径
                file_filter
            )
            
            if not save_path:  # 用户取消保存
                return
            
            # 保存模型
            joblib.dump(trained_model, save_path)
            QMessageBox.information(self, "保存成功", f"模型已保存到: {save_path}")
            
            # 可选：更新最后选择的路径（用于下次默认目录）
            self.lastSelectedPath = os.path.dirname(save_path)
            
        except Exception as e:
            QMessageBox.warning(self, "保存模型失败", str(e))
            
    
    def handle_vibration_analysis(self):#测试阶段
        """
        振动分析按钮的槽函数
        """


        try:
            # 调用 VibrationAnalyzer 的方法
            result = analyze_VA_psd(self, VA_data_path,fs=sampling_rate)
            if result is not None:
                self.psd_results = result['psd_data']  # 获取 PSD 数据
                self.lineEdit_state.setText("功率谱分析成功")
            else:
                self.lineEdit_state.setText("分析失败")
        except Exception as e:
            print(f"振动分析失败: {str(e)}")   

    def preview_VA_data(self):
        """
        预览振动数据
        """

        try:
            # 调用 VibrationAnalyzer 的方法
            preview_VAdata(self, VA_data_path)
            self.lineEdit_state.setText("数据预览成功")
        except Exception as e:
            print(f"数据预览失败: {str(e)}")
    # 主窗口类内
    def save_psd_result(self):
        save_psd_result_util(self, self.psd_results)   



    def show_previous_page(self):
        """显示上一页"""
        if self.current_page > 0:
            self.current_page -= 1
            self.update_graphics_view()

    def show_next_page(self):
        """显示下一页"""
        if self.current_page < len(self.figures) - 1:
            self.current_page += 1
            self.update_graphics_view()

    def update_graphics_view(self):
        """更新 graphicsView 中显示的内容"""
        if self.figures:
            self.graphicscene.clear()  # 清空当前内容
            image_path = self.figures[self.current_page]
            pixmap = QPixmap(image_path)  # 加载图像
            self.graphicscene.addPixmap(pixmap)  # 将图像添加到场景
            self.graphicsView.setScene(self.graphicscene)
            self.graphicsView.show()



    @pyqtSlot()
    def AL_VA_para(self):
    
        self.ui_pop = POP_VA_para(self)
        self.ui_pop.show()

    def AL_load_model_para(self):
    
        self.ui_pop = POP_Load_model_para(self)
        self.ui_pop.show()

    def AL_dataset_handle(self):
        """
        打开数据集处理窗口
        """
        self.ui_pop = POP_DatasetHandleWindow(self)
        self.ui_pop.show()
    
    def AL_DT_para(self):
      
        self.ui_pop = POP_DT_para(self)
        self.ui_pop.show()
 
    def AL_RF_para(self):
      
        self.ui_pop = POP_RF_para(self)
        self.ui_pop.show()

    def AL_SVM_para(self):
        
            self.ui_pop = POP_SVM_para(self)
            self.ui_pop.show()  
    
    def AL_MLP_para(self):
        
            self.ui_pop = POP_MLP_para(self)
            self.ui_pop.show()  

    def AL_ET_para(self):
        
            self.ui_pop = POP_ET_para(self)
            self.ui_pop.show() 

    def AL_GL_para(self):
        
            self.ui_pop = POP_GL_para(self)
            self.ui_pop.show()

    def AL_MTW_para(self):  
            
                self.ui_pop = POP_MTW_para(self)
                self.ui_pop.show()  

    def AL_REMTW_para(self):  
            
                self.ui_pop = POP_REMTW_para(self)
                self.ui_pop.show()

    def AL_MMoE_para(self):  
            
                self.ui_pop = POP_MMoE_para(self)
                self.ui_pop.show()
    
    def AL_GP_para(self):  
        
            self.ui_pop = POP_GP_para(self)
            self.ui_pop.show()

    def get_gpu_util(self):
    # 调用 nvidia-smi 获取利用率（返回纯数字）
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        try:
            return int(result.stdout.strip())  # 转换为整数（如 78）
        except:
            return 0  # 获取失败时默认返回0
    
    def save(self):
        # 保存功能保持不变，可以自动识别保存格式
        self.file_save, _ = QFileDialog.getSaveFileName(
            self, "保存文件", "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;MATLAB Files (*.mat)"
        )

        if self.file_save:
            try:
                if self.file_save.endswith('.mat'):
                    # 将DataFrame转换为MATLAB兼容格式
                    mat_data = {col: self.data_save[col].values for col in self.data.columns}
                    sio.savemat(self.file_save, mat_data)
                elif self.file_save.endswith(('.xls', '.xlsx')):
                    self.data_save.to_excel(self.file_save, index=True)
                else:  # 默认保存CSV
                    self.data_save.to_csv(self.file_save, index=True)

                self.lineEdit_Algorithm_name.setText('Data saved successfully')
            except Exception as e:
                self.lineEdit_Algorithm_name.setText(f"Save failed: {str(e)}")#尝试一下，三种格式的接入   

    def openfolder(self):
        '''新内容取消的时候不会改变linedit'''
        file_filter = "Data Files (*.csv *.xls *.xlsx *.mat);;Excel Files (*.xls *.xlsx);;MATLAB Files (*.mat);;CSV Files (*.csv);;All Files(*.*)"
        # 使用上次路径作为初始目录，如果不存在则使用"data/"
        initial_dir = self.lastSelectedPath if self.lastSelectedPath else "data/"
        self.fileName, _ = QFileDialog.getOpenFileName(
            self, "选取文件", initial_dir, file_filter
        )

        if not self.fileName:  # 用户取消选择
            return  # 直接返回，不改变lineEdit的内容

        # 更新上次路径并显示到lineEdit
        self.lastSelectedPath = self.fileName
        self.lineEdit_dataset_file.setText(self.fileName)

        print("选择的文件:", self.fileName)

        if not os.path.isfile(self.fileName):
            self.lineEdit_dataset_file.setText("File path doesn't exist")
            return

        try:
            # 根据文件扩展名选择读取方式
            if self.fileName.endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(self.fileName)
            elif self.fileName.endswith('.mat'):
                mat_data = sio.loadmat(self.fileName)
                all_keys = list(mat_data.keys())
                feature_keys = all_keys[3:]
                df = pd.DataFrame({
                    key: mat_data[key].squeeze()  # 压缩单维度，如 (n,1)→n
                    for key in feature_keys
                })
                columns = df.columns.tolist()

                new_columns = columns[1:] + [columns[0]]
                self.data = df[new_columns]

            else:  # 默认处理CSV
                self.data = pd.read_csv(self.fileName, encoding='utf-8')

            # 公共数据处理流程
            self.data = self.data.dropna()  # 删除缺失值
            self.data.columns = self.data.columns.astype(str)  # 确保列名为字符串
            self.columns = self.data.columns.tolist()  # 获取所有列名列表

            # 关键修改：根据列名前缀区分输入和输出特征
            # 筛选出输入特征列（包含"input"前缀）
            input_columns = [col for col in self.columns if "input" in col.lower()]
            # 筛选出输出特征列（包含"output"前缀）
            output_columns = [col for col in self.columns if "output" in col.lower()]

            # 如果没有找到符合命名规则的列，给出警告
            if not input_columns:
                print("警告：未找到以'input'命名的输入特征列")
            if not output_columns:
                print("警告：未找到以'output'命名的输出特征列")

            # 显示数据信息
            self.listWidget_inputfeature.clear()
            self.listWidget_outputfeature.clear()
            # 使用新的分类方式添加到列表控件
            self.add_listitem(input_columns, self.listWidget_inputfeature)
            self.add_listitem(output_columns, self.listWidget_outputfeature)
            
            self.shape = self.data.shape
            self.lineEdit_dataset_nums.setText(f'({self.shape[0]} Samples * {self.shape[1]} Features)')

            self.spinBox_train_end.setValue(self.shape[0] * 0.9)
            self.spinBox_test_start.setValue(self.shape[0] * 0.9 + 1)
            self.spinBox_test_end.setValue(self.shape[0])
            self.data_load = 1

        except Exception as e:
            self.lineEdit_dataset_nums.setText(f"Error: {str(e)}")
            return   
    
    def add_listitem(self, columns, list):
        """
        :param list: 要插入的选项文字数据列表 list[str] eg：['城市','小区','小区ID']
        """
        data_list = columns
        print(data_list)
        for i in data_list:
            # box = QCheckBox(i)  # 实例化一
            #
            # # 个QCheckBox，吧文字传进去
            # box.setChecked(True)
            item = QListWidgetItem(i)  # 实例化一个Item，QListWidget，不能直接加入QCheckBox
            item.setCheckState(Qt.Checked)
            item.checkState()
            list.addItem(item) 
    
    def change_CheckBox_input(self):
        count = self.listWidget_inputfeature.count()
        count_isChecked = 0
        for index in range(count):
            item = self.listWidget_inputfeature.item(index)
            if item.checkState() == Qt.Checked:
                count_isChecked += 1
        if count_isChecked == count:
            for index in range(count):
                item = self.listWidget_inputfeature.item(index)
                item.setCheckState(Qt.Unchecked)
        else:
            for index in range(count):
                item = self.listWidget_inputfeature.item(index)
                item.setCheckState(Qt.Checked)

    def change_CheckBox_output(self):
        count = self.listWidget_outputfeature.count()
        count_isChecked = 0
        for index in range(count):
            item = self.listWidget_outputfeature.item(index)
            if item.checkState() == Qt.Checked:
                count_isChecked += 1
        if count_isChecked == count:
            for index in range(count):
                item = self.listWidget_outputfeature.item(index)
                item.setCheckState(Qt.Unchecked)
        else:
            for index in range(count):
                item = self.listWidget_outputfeature.item(index)
                item.setCheckState(Qt.Checked)
    
    # 在主窗口类（如Ui_MainWindow的子类）中添加
    def predict_with_loaded_model(self):
        if not hasattr(self, 'selected_model_path') or not self.selected_model_path:
            QMessageBox.warning(self, "警告", "请先选择预训练模型")
            return
        if not hasattr(self, 'predict_data') or self.predict_data is None:
            QMessageBox.warning(self, "警告", "请先加载预测数据")
            return
        if not hasattr(self, 'predict_input_cols') or not hasattr(self, 'predict_output_cols'):
            QMessageBox.warning(self, "警告", "数据中未找到有效的输入/输出列")
            return
        
        try:
            # 1. 加载模型并判断类型
            from ALL_Algorithms.Algorithms import load_model
            from mutar import GroupLasso, ReMTW, MTW  # 导入特殊模型类用于类型判断
            model = load_model(self.selected_model_path)
            if model is None:
                raise ValueError("模型加载失败")
            self.start_time = time.time()

            # 2. 准备输入特征和真实值
            predict_data = self.predict_data
            self.predict_input_columns, flag1 = self.get_input()
            self.predict_output_columns, flag2 = self.get_output()
            self.predict_output_cols = self.predict_output_columns
            X_2d = predict_data[self.predict_input_columns].values  # 基础2D输入 (样本×特征)
            y_test = predict_data[self.predict_output_columns].values  # 真实值 (样本×任务)
            n_tasks = len(self.predict_output_columns)  # 任务数量

            # 3. 根据模型类型处理输入格式并预测
            model_type = None
            y_pred = None

            # 处理Group Lasso模型 (需要3D输入)
            if isinstance(model, GroupLasso):
                model_type = "GroupLasso"
                # 转换为3D格式 (任务×样本×特征)
                X_3d = np.repeat(X_2d[None, :, :], n_tasks, axis=0)
                y_pred = model.predict(X_3d).T  # 预测后转置为 (样本×任务)

            # 处理ReMTW模型 (需要3D输入)
            elif isinstance(model, ReMTW):
                model_type = "ReMTW"
                X_3d = np.repeat(X_2d[None, :, :], n_tasks, axis=0)
                y_pred = model.predict(X_3d).T

            # 处理MTW模型 (需要3D输入)
            elif isinstance(model, MTW):
                model_type = "MTW"
                X_3d = np.repeat(X_2d[None, :, :], n_tasks, axis=0)
                y_pred = model.predict(X_3d).T

            # 普通模型 (2D输入)
            else:
                model_type = "Normal"
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_2d)
                else:
                    raise ValueError("加载的模型不支持predict方法")

            # 确保预测结果维度正确
            if y_test.ndim == 1:
                y_test = y_test.reshape(-1, 1)
                y_pred = y_pred.reshape(-1, 1)
            n_tasks = y_test.shape[1]  # 重新确认任务数

            # 4. 计算评估指标 (与原逻辑一致，确保兼容所有模型)
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            mse_list = []
            r2_list = []
            rmse_list = []
            mae_list = []
            
            for i in range(n_tasks):
                y_test_task = y_test[:, i] if n_tasks > 1 else y_test.ravel()
                y_pred_task = y_pred[:, i] if n_tasks > 1 else y_pred.ravel()
                
                mse_list.append(mean_squared_error(y_test_task, y_pred_task))
                r2_list.append(r2_score(y_test_task, y_pred_task))
                rmse_list.append(np.sqrt(mse_list[-1]))
                mae_list.append(mean_absolute_error(y_test_task, y_pred_task))
            
            overall_mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
            overall_r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
            overall_rmse = np.sqrt(overall_mse)
            overall_mae = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
            
            metrics = {
                'MSE': overall_mse, 'MSE_list': mse_list,
                'R2': overall_r2, 'R2_list': r2_list,
                'RMSE': overall_rmse, 'RMSE_list': rmse_list,
                'MAE': overall_mae, 'MAE_list': mae_list
            }

            # 计算分贝偏差指标
            epsilon = 1e-8
            y_test_pos = y_test + epsilon
            y_pred_pos = y_pred + epsilon
            db_diff = 20 * np.log10(y_pred_pos / y_test_pos)

            db_within_3_ratio_list = []
            for i in range(n_tasks):
                within_3 = np.abs(db_diff[:, i]) <= 3 if n_tasks > 1 else np.abs(db_diff) <= 3
                db_within_3_ratio_list.append(np.mean(within_3))
            db_within_3_ratio = np.mean(db_within_3_ratio_list)

            total_db_deviation = np.sum(np.abs(db_diff))
            total_db_deviation_per_feature = [
                np.sum(np.abs(db_diff[:, i])) if n_tasks > 1 else np.sum(np.abs(db_diff))
                for i in range(n_tasks)
            ]

            metrics.update({
                'db_within_3_ratio': db_within_3_ratio,
                'db_within_3_ratio_list': db_within_3_ratio_list,
                'total_db_deviation': total_db_deviation,
                'total_db_deviation_per_feature': total_db_deviation_per_feature
            })

            # 5. 可视化结果 (根据模型类型选择对应函数)
            method_name = model_type  # 使用模型类型作为方法名
            data_test_index = predict_data.index  # 测试集索引

            # Group Lasso可视化
            if model_type == "GroupLasso":
                
                group_lasso_plot_and_evaluate(
                    self,
                    coef_shared=model.coef_shared_,
                    coef_specific=model.coef_specific_,
                    method=method_name,
                    input_columns=self.predict_input_columns,
                    output_columns=self.predict_output_columns,
                    MSE=metrics['MSE'],
                    MSE_list=metrics['MSE_list'],
                    RMSE=metrics['RMSE'],
                    RMSE_list=metrics['RMSE_list'],
                    MAE=metrics['MAE'],
                    MAE_list=metrics['MAE_list'],
                    R2=metrics['R2'],
                    R2_list=metrics['R2_list'],
                    db_within_3_ratio_list=metrics['db_within_3_ratio_list'],
                    total_db_deviation_per_feature=metrics['total_db_deviation_per_feature'],
                    y_test=y_test,
                    y_pred=y_pred,
                    data_test_index=data_test_index
                )

            # ReMTW可视化
            elif model_type == "ReMTW":
                
                remtw_plot_and_evaluate(
                    self,
                    remtw_model=model,
                    method=method_name,
                    input_columns=self.predict_input_columns,
                    output_columns=self.predict_output_columns,
                    MSE=metrics['MSE'],
                    MSE_list=metrics['MSE_list'],
                    RMSE=metrics['RMSE'],
                    RMSE_list=metrics['RMSE_list'],
                    MAE=metrics['MAE'],
                    MAE_list=metrics['MAE_list'],
                    R2=metrics['R2'],
                    R2_list=metrics['R2_list'],
                    db_within_3_ratio_list=metrics['db_within_3_ratio_list'],
                    total_db_deviation_per_feature=metrics['total_db_deviation_per_feature'],
                    y_test=y_test,
                    y_pred=y_pred,
                    data_test_index=data_test_index
                )

            # MTW可视化
            elif model_type == "MTW":
                mtw_plot_and_evaluate(
                    self,
                    mtw_model=model,
                    method=method_name,
                    input_columns=self.predict_input_columns,
                    output_columns=self.predict_output_columns,
                    MSE=metrics['MSE'],
                    MSE_list=metrics['MSE_list'],
                    RMSE=metrics['RMSE'],
                    RMSE_list=metrics['RMSE_list'],
                    MAE=metrics['MAE'],
                    MAE_list=metrics['MAE_list'],
                    R2=metrics['R2'],
                    R2_list=metrics['R2_list'],
                    db_within_3_ratio_list=metrics['db_within_3_ratio_list'],
                    total_db_deviation_per_feature=metrics['total_db_deviation_per_feature'],
                    y_test=y_test,
                    y_pred=y_pred,
                    data_test_index=data_test_index
                )

            # 普通模型可视化
            else:
                if y_test.ndim == 1 or y_test.shape[1] == 1:
                    # 单输出可视化
                    single_plot_and_evaluate(
                        self,
                        y_test=y_test.ravel(),
                        y_pred=y_pred.ravel(),
                        method=method_name,
                        data_test=predict_data,
                        output_columns=self.predict_output_cols,
                        N_start_test=0,
                        N_end_test=len(y_test),
                        MSE=overall_mse,
                        RMSE=overall_rmse,
                        MAE=overall_mae,
                        R2=overall_r2,
                        db_within_3_ratio=metrics['db_within_3_ratio'],
                        total_db_deviation=metrics['total_db_deviation']
                    )
                else:
                    # 多输出可视化
                    Multi_output_plot_and_evaluate(
                        self,
                        y_test=y_test,
                        y_pred=y_pred,
                        method=method_name,
                        data_test=predict_data,
                        output_columns=self.predict_output_cols,
                        N_start_test=0,
                        N_end_test=len(y_test),
                        MSE_list=metrics['MSE_list'],
                        RMSE_list=metrics['RMSE_list'],
                        MAE_list=metrics['MAE_list'],
                        R2_list=metrics['R2_list'],
                        db_within_3_ratio_list=metrics['db_within_3_ratio_list'],
                        total_db_deviation_per_feature=metrics['total_db_deviation_per_feature']
                    )

        except Exception as e:
            QMessageBox.critical(self, "错误", f"预测过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
                
        
    def All_Methods_Begin(self):
        global method

        if self.data_load == 0:
            self.lineEdit_Algorithm_name.setText("请在右侧选择所需算法")
            self.lineEdit_state.setText('')
            return

        if method == 'NONE':
                    self.lineEdit_Algorithm_name.setText("Please select an Algorithm!!!")
                    self.lineEdit_state.setText('')
                    return
    

        if self.checkBox_percentage.isChecked():
            self.shape[0] = self.data.shape
            N_start_train = round(self.shape[0] * int(self.spinBox_train_start.text()) / 100)
            N_end_train = round(self.shape[0] * int(self.spinBox_train_end.text()) / 100)
            N_start_test = round(self.shape[0] * int(self.spinBox_test_start.text()) / 100)
            N_end_test = round(self.shape[0] * int(self.spinBox_test_end.text()) / 100)
        else:
            N_start_train = int(self.spinBox_train_start.text())
            N_end_train = int(self.spinBox_train_end.text())
            N_start_test = int(self.spinBox_test_start.text())
            N_end_test = int(self.spinBox_test_end.text())
        if (N_start_train < N_end_train) & (N_end_train <= self.shape[0])\
            & (N_start_test < N_end_test) & (N_end_test <= self.shape[0]):
            pass
        else:
            self.lineEdit_Algorithm_name.setText("Data index is illegal or beyond range!!!")
            self.lineEdit_state.setText('')
            return

        print('ok now')


        self.N_train = N_end_train - N_start_train
        self.N_test = N_end_test - N_start_test
        self.start_time = time.time()
        # self.mytoolBar.clear()
        self.lineEdit_DEVICE.clear()

        #数据统一处理
        data = self.data
        input_columns, flag1 = self.get_input()
        output_columns, flag2 = self.get_output()
        if flag1 == 0 | flag2 == 0:
            self.lineEdit_Algorithm_name.setText("Select at least 1 input and 1 output!")
            self.lineEdit_state.setText('')
            return
        data_X = data[input_columns]   # 选择输入特征
        data_y = data[output_columns]  # 选择输出特征   
        data_filter = pd.concat([data_X, data_y], axis=1)  # 合并数据
        data_train = data_filter[N_start_train : N_end_train]
        data_test = data_filter[N_start_test : N_end_test]   
        print('check')


        
        if method == 'DT':
            
            self.new_model = 1
            model,_, y_test, y_pred, metrics = multi_task_regression_predictor(
                data_train,
                data_test,
                input_columns,
                output_columns,
                model_type='DT',
                scale_features=scale_features,
                random_state=int(random_state),
                max_depth=int(max_depth),
            )
            
            self.trained_model = model  # 保存训练好的模型
            # ask_and_save_model(self,method,default_name='trained_model'+'_'+str(method)+'.pkl')    

            # `model` 是训练好的模型，`y_pred` 是预测结果
            print("预测结果:", y_pred)
            print("真实值:", y_test)
            print("评估指标:", metrics)
            # 检查测试数据

            if len(output_columns) > 1:
                self.data_save=Multi_output_plot_and_evaluate(self,y_test, 
                                               y_pred, method,
                                                data_test, 
                                                output_columns, 
                                                N_start_test,
                                                N_end_test,
                                                MSE_list=metrics['MSE_list'],  # 传递每个输出的MSE列表
                                                RMSE_list=metrics['RMSE_list'],  # 传递每个输出的RMSE列表
                                                MAE_list=metrics['MAE_list'],  # 传递每个输出的MAE列表
                                                R2_list=metrics['R2_list'], # 传递每个输出的R2列表)
                                                db_within_3_ratio_list = metrics['db_within_3_ratio_list'],
                                                total_db_deviation_per_feature = metrics['total_db_deviation_per_feature']
                                                )
                
            if len(output_columns) == 1:
                self.data_save=single_plot_and_evaluate(self,y_test, y_pred,
                                        method, data_test,
                                        output_columns, N_start_test, 
                                        N_end_test,
                                        metrics['MSE'],
                                        metrics['RMSE'],
                                        metrics['MAE'],
                                        metrics['R2'],
                                        db_within_3_ratio=metrics['db_within_3_ratio'],
                                        total_db_deviation=metrics['total_db_deviation']
                                        )
            self.lineEdit_DEVICE.setText("CPU")
            #这个是为了防止弹出保存模型的窗口而设置的延迟2秒功能  
            # QTimer.singleShot(2000, lambda: ask_and_save_model(self, model, default_name='trained_model_' + str(method) + '.pkl'))

        if method =='RF':
            self.new_model = 1
            model,_, y_test, y_pred, metrics = multi_task_regression_predictor(
                data_train,
                data_test,
                input_columns,
                output_columns,
                model_type='RF',
                scale_features=scale_features,
                random_state=int(random_state),
                max_depth=int(max_depth),
                n_estimators=int(n_estimators),
            )
            self.trained_model = model  # 保存训练好的模型
            # `model` 是训练好的模型，`y_pred` 是预测结果
            print("预测结果:", y_pred)
            print("真实值:", y_test)
            print("评估指标:", metrics)
            if len(output_columns) > 1:
                self.data_save=Multi_output_plot_and_evaluate(self,y_test, 
                                               y_pred, method,
                                                data_test, 
                                                output_columns, 
                                                N_start_test,
                                                N_end_test,
                                                MSE_list=metrics['MSE_list'],  # 传递每个输出的MSE列表
                                                RMSE_list=metrics['RMSE_list'],  # 传递每个输出的RMSE列表
                                                MAE_list=metrics['MAE_list'],  # 传递每个输出的MAE列表
                                                R2_list=metrics['R2_list'],  # 传递每个输出的R2列表)
                                                db_within_3_ratio_list = metrics['db_within_3_ratio_list'],
                                                total_db_deviation_per_feature = metrics['total_db_deviation_per_feature']
                )
            if len(output_columns) == 1:
                self.data_save=single_plot_and_evaluate(self,y_test, y_pred, method,
                                          data_test, output_columns, 
                                          N_start_test, N_end_test,
                                          metrics['MSE'],
                                        metrics['RMSE'],
                                        metrics['MAE'],
                                        metrics['R2'],
                                        db_within_3_ratio=metrics['db_within_3_ratio'],
                                        total_db_deviation=metrics['total_db_deviation'])
            self.lineEdit_DEVICE.setText("CPU")
            #这个是为了防止弹出保存模型的窗口而设置的延迟2秒功能  
            # QTimer.singleShot(2000, lambda: ask_and_save_model(self, model, default_name='trained_model_' + str(method) + '.pkl'))
   

        if method =='SVM':
            self.new_model = 1
            model,_, y_test, y_pred, metrics = multi_task_regression_predictor(
                data_train,
                data_test,
                input_columns,
                output_columns,
                model_type='SVM',
                scale_features=scale_features,
                random_state=int(random_state),
                kernel=str(kernel),
                C=float(C),
                epsilon=float(epsilon),
                
                n_jobs=n_jobs,
                max_iter=-1,#SVM不需要这个参数
            )
            self.trained_model = model  # 保存训练好的模型
            # `model` 是训练好的模型，`y_pred` 是预测结果
            print("预测结果:", y_pred)
            print("真实值:", y_test)
            print("评估指标:", metrics)
            if len(output_columns) > 1:
                self.data_save=Multi_output_plot_and_evaluate(self,y_test, 
                                               y_pred, method,
                                                data_test, 
                                                output_columns, 
                                                N_start_test,
                                                N_end_test,
                                                MSE_list=metrics['MSE_list'],  # 传递每个输出的MSE列表
                                                RMSE_list=metrics['RMSE_list'],  # 传递每个输出的RMSE列表
                                                MAE_list=metrics['MAE_list'],  # 传递每个输出的MAE列表
                                                R2_list=metrics['R2_list'],  # 传递每个输出的R2列表)
                                                db_within_3_ratio_list = metrics['db_within_3_ratio_list'],
                                                total_db_deviation_per_feature = metrics['total_db_deviation_per_feature']
                )
            if len(output_columns) == 1:
                self.data_save=single_plot_and_evaluate(self,y_test, y_pred, method, 
                                         data_test, output_columns,
                                           N_start_test, N_end_test,
                                           metrics['MSE'],
                                            metrics['RMSE'],
                                            metrics['MAE'],
                                            metrics['R2'],
                                            db_within_3_ratio=metrics['db_within_3_ratio'],
                                        total_db_deviation=metrics['total_db_deviation'])
            self.lineEdit_DEVICE.setText("CPU")
            #这个是为了防止弹出保存模型的窗口而设置的延迟2秒功能  
            # QTimer.singleShot(2000, lambda: ask_and_save_model(self, model, default_name='trained_model_' + str(method) + '.pkl'))

        if method == 'ET':
            self.new_model = 1
            model,_, y_test, y_pred, metrics = multi_task_regression_predictor(
                data_train,
                data_test,
                input_columns,
                output_columns,
                model_type='ET',
                scale_features=scale_features,
                n_jobs=int(n_jobs),
                random_state=int(random_state),
                max_depth=int(max_depth),
                n_estimators=int(n_estimators),
            )
            self.trained_model = model  # 保存训练好的模型
            # `model` 是训练好的模型，`y_pred` 是预测结果
            print("预测结果:", y_pred)
            print("真实值:", y_test)
            print("评估指标:", metrics)
            if len(output_columns) > 1:
                self.data_save=Multi_output_plot_and_evaluate(self,y_test, 
                                               y_pred, method,
                                                data_test, 
                                                output_columns, 
                                                N_start_test,
                                                N_end_test,
                                                MSE_list=metrics['MSE_list'],  # 传递每个输出的MSE列表
                                                RMSE_list=metrics['RMSE_list'],  # 传递每个输出的RMSE列表
                                                MAE_list=metrics['MAE_list'],  # 传递每个输出的MAE列表
                                                R2_list=metrics['R2_list'],  # 传递每个输出的R2列表)
                                                db_within_3_ratio_list = metrics['db_within_3_ratio_list'],
                                                total_db_deviation_per_feature = metrics['total_db_deviation_per_feature']
                )
            if len(output_columns) == 1:
                self.data_save=single_plot_and_evaluate(self,y_test, y_pred, method, data_test,
                                           output_columns, N_start_test, N_end_test,
                                           metrics['MSE'],
                                            metrics['RMSE'],
                                            metrics['MAE'],
                                            metrics['R2'],
                                            db_within_3_ratio=metrics['db_within_3_ratio'],
                                        total_db_deviation=metrics['total_db_deviation'])
            self.lineEdit_DEVICE.setText("CPU")
            #这个是为了防止弹出保存模型的窗口而设置的延迟2秒功能  
            # QTimer.singleShot(2000, lambda: ask_and_save_model(self, model, default_name='trained_model_' + str(method) + '.pkl'))


        if method == 'MLP':
            self.new_model = 1
            model,_, y_test, y_pred, metrics = multi_task_regression_predictor(
                data_train,
                data_test,
                input_columns,
                output_columns,
                model_type='MLP',
                scale_features=scale_features,
                random_state=int(random_state),
                max_iter=int(max_iter),
                alpha=float(mlp_alpha),
                mlp_hidden_layers=tuple(hidden_layer_sizes),
                
            )
            self.trained_model = model  # 保存训练好的模型
            # `model` 是训练好的模型，`y_pred` 是预测结果
            print("预测结果:", y_pred)
            print("真实值:", y_test)
            print("评估指标:", metrics)
            if len(output_columns) > 1:
                self.data_save=Multi_output_plot_and_evaluate(self,y_test, 
                                               y_pred, method,
                                                data_test, 
                                                output_columns, 
                                                N_start_test,
                                                N_end_test,
                                                MSE_list=metrics['MSE_list'],  # 传递每个输出的MSE列表
                                                RMSE_list=metrics['RMSE_list'],  # 传递每个输出的RMSE列表
                                                MAE_list=metrics['MAE_list'],  # 传递每个输出的MAE列表
                                                R2_list=metrics['R2_list'],  # 传递每个输出的R2列表)
                                                db_within_3_ratio_list = metrics['db_within_3_ratio_list'],
                                                total_db_deviation_per_feature = metrics['total_db_deviation_per_feature']
                )
            if len(output_columns) == 1:    
                self.data_save=single_plot_and_evaluate(self,y_test, y_pred, method,
                                                         data_test, output_columns, 
                                                         N_start_test, N_end_test,
                                                         metrics['MSE'],
                                                        metrics['RMSE'],
                                                        metrics['MAE'],
                                                        metrics['R2'],db_within_3_ratio=metrics['db_within_3_ratio'],
                                                        total_db_deviation=metrics['total_db_deviation'])

            self.lineEdit_DEVICE.setText("CPU")
            #这个是为了防止弹出保存模型的窗口而设置的延迟2秒功能  
            # QTimer.singleShot(2000, lambda: ask_and_save_model(self, model, default_name='trained_model_' + str(method) + '.pkl'))

        
        if method =='GP':
            self.new_model = 1
            model,_, y_test, y_pred, metrics = multi_task_regression_predictor(
                data_train,
                data_test,
                input_columns,
                output_columns,
                model_type='GP',
                gp_learning_rate=float(learning_rate),
                gp_training_iterations=int(training_iterations),
                
            )
            self.trained_model = model  # 保存训练好的模型
            # `model` 是训练好的模型，`y_pred` 是预测结果
            print("预测结果:", y_pred)
            print("真实值:", y_test)
            print("评估指标:", metrics)
            if len(output_columns) > 1:
                self.data_save=Multi_output_plot_and_evaluate(self,y_test, 
                                               y_pred, method,
                                                data_test, 
                                                output_columns, 
                                                N_start_test,
                                                N_end_test,
                                                MSE_list=metrics['MSE_list'],  # 传递每个输出的MSE列表
                                                RMSE_list=metrics['RMSE_list'],  # 传递每个输出的RMSE列表
                                                MAE_list=metrics['MAE_list'],  # 传递每个输出的MAE列表
                                                R2_list=metrics['R2_list'],  # 传递每个输出的R2列表)
                                                db_within_3_ratio_list = metrics['db_within_3_ratio_list'],
                                                total_db_deviation_per_feature = metrics['total_db_deviation_per_feature']
                )
            if len(output_columns) == 1:    
                self.data_save=single_plot_and_evaluate(self,y_test, y_pred, method,
                                                         data_test, output_columns, 
                                                         N_start_test, N_end_test,
                                                         metrics['MSE'],
                                                        metrics['RMSE'],
                                                        metrics['MAE'],
                                                        metrics['R2'],db_within_3_ratio=metrics['db_within_3_ratio'],
                                                        total_db_deviation=metrics['total_db_deviation'])

            self.lineEdit_DEVICE.setText("CPU")
            
            

        
        if method == 'GL':
            self.new_model = 1
            model, X_test, y_test, y_pred, metrics,data_index = group_lasso_predictor(
                data_train=data_train,
                data_test=data_test,
                input_columns=input_columns,
                output_columns=output_columns,
                alpha=float(alpha),
                random_state=int(random_state),
                max_iter=int(max_iter),
                tol=float(tol),
                show_plots=False,
                show_prints=True
            )
            self.trained_model = model  # 保存训练好的模型
            # 打印结果
            print("预测结果:", y_pred)
            print("真实值:", y_test)
            print("评估指标:", metrics)

            # 可视化结果
            group_lasso_plot_and_evaluate(self,
                    coef_shared=model.coef_shared_,
                    coef_specific=model.coef_specific_,
                    method="Group Lasso",
                    input_columns=input_columns,
                    output_columns=output_columns,
                    MSE=metrics['MSE'],
                    RMSE = metrics['RMSE'],
                    MAE = metrics['MAE'],
                    R2=metrics['R2'],
                    MSE_list=metrics['MSE_list'],
                    MAE_list=metrics['MAE_list'],
                    RMSE_list=metrics['RMSE_list'],
                    R2_list=metrics['R2_list'],
                    y_test=y_test, 
                    y_pred=y_pred,
                    db_within_3_ratio_list = metrics['db_within_3_ratio_list'],
                    total_db_deviation_per_feature = metrics['total_db_deviation_per_feature'],
                    data_test_index=data_index
                )
            self.lineEdit_DEVICE.setText("CPU")
    
        if method == 'MTW':
            self.new_model = 1
            model, X_test, y_test, y_pred, metrics,data_index = MTW_Lasso(
                data_train=data_train,
                data_test=data_test,
                input_columns=input_columns,
                output_columns=output_columns,
                alpha=float(alpha),
                beta=float(beta),
                max_iter=int(max_iter),
                tol=float(tol),
                model_type='MTW',
                gpu =True,              
            )
            self.trained_model = model  # 保存训练好的模型
            # 打印结果
            print("预测结果:", y_pred)
            print("真实值:", y_test)
            print("评估指标:", metrics)

            # 可视化结果
            mtw_plot_and_evaluate(self,
                    mtw_model=model,
                    method="MTW Lasso",
                    input_columns=input_columns,
                    output_columns=output_columns,
                    MSE=metrics['MSE'],
                    RMSE = metrics['RMSE'],
                    MAE = metrics['MAE'],
                    R2=metrics['R2'],
                    MSE_list=metrics['MSE_list'],
                    MAE_list=metrics['MAE_list'],
                    RMSE_list=metrics['RMSE_list'],
                    R2_list=metrics['R2_list'],
                    db_within_3_ratio_list = metrics['db_within_3_ratio_list'],
                    total_db_deviation_per_feature = metrics['total_db_deviation_per_feature'],
                    data_test_index=data_index,
                    y_test=y_test, 
                    y_pred=y_pred
                )
            self.lineEdit_DEVICE.setText("GPU")
        
        if method == 'REMTW':
            self.new_model = 1
            model, X_test, y_test, y_pred, metrics,data_index = REMTW_Lasso(
                data_train=data_train,
                data_test=data_test,
                input_columns=input_columns,
                output_columns=output_columns,
                alpha=float(alpha),
                beta=float(beta),
                tol =float(tol),
                max_iter=int(max_iter),
                model_type='REMTW',
                gpu =True,              
            )
            self.trained_model = model  # 保存训练好的模型
            # 打印结果
            print("预测结果:", y_pred)
            print("真实值:", y_test)
            print("评估指标:", metrics)

            # 可视化结果
            remtw_plot_and_evaluate(self,
                    remtw_model=model,
                    method="ReMTW Lasso",
                    input_columns=input_columns,
                    output_columns=output_columns,
                    MSE=metrics['MSE'],
                    RMSE = metrics['RMSE'],
                    MAE = metrics['MAE'],
                    R2=metrics['R2'],
                    MSE_list=metrics['MSE_list'],
                    MAE_list=metrics['MAE_list'],
                    RMSE_list=metrics['RMSE_list'],
                    R2_list=metrics['R2_list'],
                    data_test_index=data_index,
                    db_within_3_ratio_list = metrics['db_within_3_ratio_list'],
                    total_db_deviation_per_feature = metrics['total_db_deviation_per_feature'],
                    y_test=y_test, 
                    y_pred=y_pred
                    
                )
            self.lineEdit_DEVICE.setText("GPU")
        if method == 'MMoE':
            self.new_model = 1
            model, mmoe_scale_features, y_test, y_pred, metrics = multi_task_regression_predictor(
                data_train,
                data_test,
                input_columns,
                output_columns,
                model_type='MMoE',
                scale_features=False,
                random_state=int(random_state),
                max_iter=int(max_iter),
                alpha=float(alpha),
                # 下面参数可根据你的界面设置传入
                mmoe_num_experts=int(mmoe_num_experts),
                mmoe_expert_hidden=int(mmoe_expert_hidden),
                mmoe_learning_rate=float(mmoe_learning_rate),
                mmoe_dropout_rate=float(mmoe_dropout_rate),
                mmoe_epochs=int(mmoe_epochs),
                mmoe_batch_size=int(mmoe_batch_size),
                mmoe_lambda_balance=float(mmoe_lambda_balance)
            )
            self.trained_model = model  # 保存训练好的模型
            print("MMoE预测结果:", y_pred)
            print("MMoE真实值:", y_test)
            print("MMoE评估指标:", metrics)
            # 多输出绘图
            if len(output_columns) > 1:
                self.data_save=Multi_output_plot_and_evaluate(self,y_test, 
                                               y_pred, method,
                                                data_test, 
                                                output_columns, 
                                                N_start_test,
                                                N_end_test,
                                                MSE_list=metrics['MSE_list'],  # 传递每个输出的MSE列表
                                                RMSE_list=metrics['RMSE_list'],  # 传递每个输出的RMSE列表
                                                MAE_list=metrics['MAE_list'],  # 传递每个输出的MAE列表
                                                R2_list=metrics['R2_list'],  # 传递每个输出的R2列表)
                                                db_within_3_ratio_list = metrics['db_within_3_ratio_list'],
                                                total_db_deviation_per_feature = metrics['total_db_deviation_per_feature']
                )
            else:
                self.data_save = single_plot_and_evaluate(
                    self, y_test, y_pred, method, data_test,
                    output_columns, N_start_test, N_end_test,
                    metrics['MSE'],
                    metrics['RMSE'],
                    metrics['MAE'],
                    metrics['R2'],
                    db_within_3_ratio=metrics['db_within_3_ratio'],
                    total_db_deviation=metrics['total_db_deviation']
                )
            self.lineEdit_DEVICE.setText("GPU" if torch.cuda.is_available() else "CPU")
    #单输出绘图的时候调用此函数，进行可视化展示,前五个算法的单输出画图展示

    def get_input(self):#取得输入的特征
        # 获取输入特征列表和标志位
    
        flag = 0
        count = self.listWidget_inputfeature.count()  # 得到QListWidget的总个数
        item_list = [self.listWidget_inputfeature.item(i)
                    for i in range(count)]  # 得到QListWidget里面所有QListWidgetItem
        chooses = []  # 存放被选择的数据
        for it in item_list:  # type QCheckBox
            if it.checkState():
                flag = 1
                chooses.append(it.text())
        return chooses, flag

    def get_output(self):#取得输出的特征
        # 获取输出特征列表和标志位
        flag = 0
        count = self.listWidget_outputfeature.count()  # 得到QListWidget的总个数
        item_list = [self.listWidget_outputfeature.item(i)
                    for i in range(count)]  # 得到QListWidget里面所有QListWidgetItem
        chooses = []  # 存放被选择的数据
        for it in item_list:
            if it.checkState():
                flag = 1
                chooses.append(it.text())
        return chooses, flag  
    
    def change_percentage(self):
        box = self.checkBox_percentage#换成复选框
        if box.isChecked():
            self.spinBox_train_start.setSingleStep(1)
            self.spinBox_train_end.setSingleStep(1)
            self.spinBox_test_start.setSingleStep(1)
            self.spinBox_test_end.setSingleStep(1)
            if self.data_load == 1:
                N_start_train = int(self.spinBox_train_start.text())
                N_end_train = int(self.spinBox_train_end.text())
                N_start_test = int(self.spinBox_test_start.text())
                N_end_test = int(self.spinBox_test_end.text())

                N_start_train_p = round(N_start_train/self.shape[0]*100)
                N_end_train_p = round(N_end_train/self.shape[0]*100)
                N_start_test_p = round(N_start_test/self.shape[0]*100)
                N_end_test_p = round(N_end_test/self.shape[0]*100)
                self.spinBox_train_start.setValue( N_start_train_p)#设置值
                self.spinBox_train_end.setValue( N_end_train_p)
                self.spinBox_test_start.setValue( N_start_test_p)
                self.spinBox_test_end.setValue( N_end_test_p)


        else:
            self.spinBox_train_start.setSingleStep(1000)
            self.spinBox_train_end.setSingleStep(1000)
            self.spinBox_test_start.setSingleStep(1000)
            self.spinBox_test_end.setSingleStep(1000)
            if self.data_load == 1:
                N_start_train_p = int(self.spinBox_train_start.text())
                N_end_train_p = int(self.spinBox_train_end.text())
                N_start_test_p = int(self.spinBox_test_start.text())
                N_end_test_p = int(self.spinBox_test_end.text())

                N_start_train = round(self.shape[0]*N_start_train_p/100)
                N_end_train = round(self.shape[0]*N_end_train_p/100)
                N_start_test = round(self.shape[0]*N_start_test_p/100)
                N_end_test = round(self.shape[0]*N_end_test_p/100)
                self.spinBox_train_start.setValue(N_start_train)
                self.spinBox_train_end.setValue( N_end_train)
                self.spinBox_test_start.setValue(N_start_test)
                self.spinBox_test_end.setValue(N_end_test)     




if __name__ == '__main__':
    app = QApplication(sys.argv)  # 在 QApplication 方法中使用，创建应用程序对象
    myWin = MyMainWindow()  # 实例化 MyMainWindow 类，创建主窗口
    myWin.show()  # 在桌面显示控件 myWin
    sys.exit(app.exec_())  # 结束进程，退出程序
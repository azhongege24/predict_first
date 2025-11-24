import subprocess
import time
import json
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
import numpy as np
from ui2025 import Ui_MainWindow #我新创建的界面类
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow,QProgressDialog, QPushButton,QMessageBox, QFileDialog, QGraphicsScene, QGraphicsView, QWidget, QCheckBox, QListWidgetItem
from PyQt5.QtCore import Qt ,QSettings,QTimer
from PyQt5.QtGui import QPixmap
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
import os
import scipy.io as sio
import torch
import joblib
import webbrowser
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
from ALL_Algorithms.algorithms11_LR_para import Ui_LR_para
from ALL_Algorithms.help_para import Ui_help_para
from ALL_Algorithms.Dataset_handle import Ui_dataset_handle
from ALL_Algorithms.Other import Ui_Other
from ALL_Algorithms.OtherLogic import POP_Other_para
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
# 在文件顶部添加VA分析模块导入
from ALL_Algorithms.VA_ANALYSIS.vibration_analyzer import VibrationAnalysisController
from ALL_Algorithms.VA_ANALYSIS.vibration_data_loader import VibrationDataLoader
from ALL_Algorithms.VA_ANALYSIS.power_spectrum_analyzer import PowerSpectrumAnalyzer
from ALL_Algorithms.VA_method_para import Ui_VA_method_para
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
global max_depth, random_state,n_estimators,kernel, C, epsilon,scale_features
global hidden_layer_sizes, max_iter,method,n_jobs,alpha,beta,tol
global mmoe_num_experts,mmoe_expert_hidden,mmoe_learning_rate,mmoe_dropout_rate
global mmoe_epochs,mmoe_batch_size,mmoe_lambda_balance,mmoe_scale_features,fit_intercept
method = 'NONE'  # 初始化方法为NONE
# 读取输入参数

class POP_VA_para(QMainWindow, Ui_VA_para, Ui_MainWindow):
    def __init__(self, parent=None):
        super(POP_VA_para, self).__init__()
        self.setupUi(self)
        self.parent_window = parent  # 保存主窗口的引用
        
        # 初始化VA分析控制器
        self.va_controller = VibrationAnalysisController()
        self.multi_file_analysis_results = []  # 新增：存储多个文件的分析结果  
        self.current_file_paths = []  # 新增：支持多个文件      
        # 当前数据和分析结果
        self.current_file_path = None
        self.current_time_data = None
        self.current_signal_data = None
        self.analysis_results = None
        self.save_directory = None
                # 分页显示相关属性
        self.current_page = 0
        self.total_pages = 0
        self.results_per_page = 2  # 每页显示2个功率谱图
        # 创建matplotlib图形画布
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        # 将matplotlib画布添加到graphicsView中
        scene = QGraphicsScene()
        scene.addWidget(self.canvas)
        self.graphicsView.setScene(scene)
        
        # 连接按钮信号
        # self.pushButton_browe_data_file.clicked.connect(self.browse_data_file)
        self.pushButton_browse_multiple_files.clicked.connect(self.browse_multiple_files)  # 新增：多文件选择按钮
        self.pushButton_psd_analysis_multiple.clicked.connect(self.perform_psd_analysis_multiple)  # 新增：多文件分析按钮
        self.pushButton_select.clicked.connect(self.select_output_directory)
        self.pushButton_preview_data_file.clicked.connect(self.preview_data)
        # self.pushButton_psd_analysis.clicked.connect(self.perform_psd_analysis)
        # self.pushButton_save_data.clicked.connect(self.save_analysis_results)
        self.pushButton_save_image.clicked.connect(self.save_analysis_images)  # 添加保存图片按钮连接
        self.pushButton_save_multiple_data.clicked.connect(self.save_multiple_analysis_results)  # 新增：多文件保存按钮
        self.pushButton_set_para.clicked.connect(self.AL_VA_method_para)
        self.pushButton_help.clicked.connect(self.show_help)
        self.pushButton_top.clicked.connect(self.show_previous_page)
        self.pushButton_bottom.clicked.connect(self.show_next_page)

        # 启用滚轮事件
        self.graphicsView.wheelEvent = self.graphics_view_wheel_event        

    def AL_VA_method_para(self):
      
        self.ui_pop = POP_VA_method_para(self)
        self.ui_pop.show()   
    def browse_data_file(self):
        """浏览数据文件"""
        file_filter = "数据文件 (*.txt *.mat);;文本文件 (*.txt);;MAT文件 (*.mat);;所有文件 (*.*)"
        initial_dir = "data/" if os.path.exists("data/") else "./"
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择振动数据文件", initial_dir, file_filter
        )
        
        if file_path:
            self.current_file_path = file_path
            self.lineEdit.setText(file_path)
            
            try:
                # 尝试加载数据以验证文件格式
                time_data, signal_data = self.va_controller.data_loader.load_data(file_path)
                self.current_time_data = time_data
                self.current_signal_data = signal_data
                
                # 解析文件信息
                file_name = os.path.basename(file_path)
                channel, direction = self.va_controller.data_loader.parse_channel_info(file_name)
                
                info_msg = f"文件加载成功\n"
                info_msg += f"数据点数: {len(time_data)}\n"
                info_msg += f"时间范围: {time_data[0]:.3f} - {time_data[-1]:.3f}秒\n"
                if channel:
                    info_msg += f"通道: {channel}\n"
                if direction:
                    info_msg += f"方向: {direction}"
                
                QMessageBox.information(self, "文件信息", info_msg)
                
            except Exception as e:
                QMessageBox.warning(self, "加载失败", f"文件加载失败: {str(e)}")
                self.current_file_path = None
                self.current_time_data = None
                self.current_signal_data = None
    
    def select_output_directory(self):
        """选择输出目录"""
        initial_dir = "results/" if os.path.exists("results/") else "./"
        directory = QFileDialog.getExistingDirectory(
            self, "选择结果保存目录", initial_dir
        )
        
        if directory:
            self.save_directory = directory
            self.lineEdit_2.setText(directory)
    
    def preview_data(self):
        """预览数据 - 显示时域和频域预览"""
        # 检查是否有可用的数据
        if self.current_time_data is None or self.current_signal_data is None:
            # 如果没有单文件数据，检查是否有多个文件
            if not self.current_file_paths:
                QMessageBox.warning(self, "警告", "请先加载数据文件！")
                return
            else:
                # 如果有多个文件但没有加载数据，尝试加载第一个文件
                try:
                    first_file_path = self.current_file_paths[0]
                    time_data, signal_data = self.va_controller.data_loader.load_data(first_file_path)
                    self.current_time_data = time_data
                    self.current_signal_data = signal_data
                    self.current_file_path = first_file_path
                except Exception as e:
                    QMessageBox.warning(self, "加载失败", f"无法加载第一个文件用于预览: {str(e)}")
                    return                
                
        try:
            
            # 清空当前图形
            self.figure.clear()
            
            # 创建子图
            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212)
            
            # 时域图
            ax1.plot(self.current_time_data, self.current_signal_data, 'b-', linewidth=1)
            ax1.set_xlabel('时间 (s)')
            ax1.set_ylabel('振动幅值')
            ax1.set_title('时域信号')
            ax1.grid(True, alpha=0.3)
            self.set_tick_font(ax1)
            # 频域预览（使用简单的FFT）
            fs = 1.0 / np.mean(np.diff(self.current_time_data))
            n = len(self.current_signal_data)
            freq = np.fft.fftfreq(n, 1/fs)[:n//2]
            fft_spectrum = np.abs(np.fft.fft(self.current_signal_data))[:n//2]
            
            ax2.plot(freq, fft_spectrum, 'r-', linewidth=1)
            ax2.set_xlabel('频率 (Hz)')
            ax2.set_ylabel('幅值')
            ax2.set_title('频域预览（FFT）')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, min(fs/2, 2000))  # 限制显示到2000Hz
            ax2.set_yscale('log')

            # 调整布局
            self.set_tick_font(ax2)
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.warning(self, "预览失败", f"数据预览失败: {str(e)}")
            
    def browse_multiple_files(self):
        """浏览多个数据文件"""
        file_filter = "数据文件 (*.txt *.mat);;文本文件 (*.txt);;MAT文件 (*.mat);;所有文件 (*.*)"
        initial_dir = "data/" if os.path.exists("data/") else "./"
        
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择多个振动数据文件", initial_dir, file_filter
        )
        
        if file_paths:
            self.current_file_paths = file_paths
            # 在界面上显示文件列表 - 显示第一个文件名和文件总数
            if len(file_paths) == 1:
                file_name = os.path.basename(file_paths[0])
                self.lineEdit.setText(f"已选择: {file_name}")
            else:
                first_file_name = os.path.basename(file_paths[0])
                self.lineEdit.setText(f"已选择: {first_file_name} 等 {len(file_paths)} 个文件")
            try:
                first_file_path = file_paths[0]
                time_data, signal_data = self.va_controller.data_loader.load_data(first_file_path)
                self.current_time_data = time_data
                self.current_signal_data = signal_data
                self.current_file_path = first_file_path
            except Exception as e:
                print(f"加载第一个文件用于预览失败: {str(e)}")
                self.current_time_data = None
                self.current_signal_data = None
                self.current_file_path = None
            
            
            # 显示文件信息
            info_msg = f"已选择 {len(file_paths)} 个文件：\n\n"
            for i, file_path in enumerate(file_paths, 1):
                file_name = os.path.basename(file_path)
                channel, direction = self.va_controller.data_loader.parse_channel_info(file_name)
                info_msg += f"{i}. {file_name}\n"
                if channel:
                    info_msg += f"   通道: {channel}\n"
                if direction:
                    info_msg += f"   方向: {direction}\n"
                info_msg += "\n"
            
            QMessageBox.information(self, "文件选择完成", info_msg)

    def perform_psd_analysis_multiple(self):
        """执行多文件功率谱分析"""
        if not self.current_file_paths:
            QMessageBox.warning(self, "警告", "请先选择多个数据文件！")
            return
        
        try:
            # 清空之前的多文件分析结果
            self.multi_file_analysis_results = []
            total_segments = 0
            
            # 创建进度对话框
            progress = QProgressDialog("正在分析多个文件...", "取消", 0, len(self.current_file_paths), self)
            progress.setWindowTitle("功率谱分析进度")
            progress.setWindowModality(Qt.WindowModal)
            
            # 分析每个文件
            for i, file_path in enumerate(self.current_file_paths):
                progress.setValue(i)
                progress.setLabelText(f"正在分析文件 {i+1}/{len(self.current_file_paths)}: {os.path.basename(file_path)}")
                
                if progress.wasCanceled():
                    break
                
                try:
                    # 分析单个文件
                    analysis_result = self.va_controller.analyze_file(
                        file_path, 
                        product_code="Unknown",
                        serial_number="Unknown"
                    )
                    
                    # 生成结构化特征集
                    feature_dataset = self.va_controller.generate_feature_dataset(analysis_result)
                    
                    # 存储分析结果
                    self.multi_file_analysis_results.append({
                        'file_path': file_path,
                        'analysis_result': analysis_result,
                        'feature_dataset': feature_dataset,
                        'file_name': os.path.basename(file_path),
                        'channel': analysis_result['channel'],
                        'direction': analysis_result['direction'],
                        'num_segments': len(analysis_result['results'])
                    })
                    
                    total_segments += len(analysis_result['results'])
                    
                except Exception as e:
                    print(f"分析文件 {file_path} 失败: {str(e)}")
                    continue
            
            progress.setValue(len(self.current_file_paths))
            
            # 显示分析结果摘要
            if self.multi_file_analysis_results:
                summary_msg = f"多文件功率谱分析完成！\n\n"
                summary_msg += f"成功分析文件数: {len(self.multi_file_analysis_results)}/{len(self.current_file_paths)}\n"
                summary_msg += f"总分段数量: {total_segments}\n\n"
                summary_msg += "各文件分析结果:\n"
                
                for result in self.multi_file_analysis_results:
                    summary_msg += f"- {result['file_name']}: {result['num_segments']} 个分段\n"
                
                QMessageBox.information(self, "分析完成", summary_msg)
                
                # 显示第一个文件的分析结果
                self.analysis_results = self.multi_file_analysis_results[0]['analysis_result']
                self.display_analysis_results()
            else:
                QMessageBox.warning(self, "分析失败", "所有文件分析都失败了！")
                
        except Exception as e:
            QMessageBox.warning(self, "分析失败", f"多文件功率谱分析失败: {str(e)}")

    def generate_descriptive_filename(self, multi_file_results):
        """基于导入的文件名生成描述性的文件名"""
        if not multi_file_results:
            return "MULTI_FILE_STRUCTURED_DATASET"
        
        # 提取所有通道和方向信息
        channels = set()
        directions = set()
        file_names = []
        
        for result in multi_file_results:
            channel = result.get('channel', '')
            direction = result.get('direction', '')
            file_name = result.get('file_name', '')
            
            if channel:
                channels.add(channel)
            if direction:
                directions.add(direction)
            if file_name:
                file_names.append(file_name)
        
        # 构建描述性文件名
        if len(channels) == 1 and len(directions) > 1:
            # 同一通道，多个方向：通道名称C1位置X、Y、Z向振动
            channel_name = list(channels)[0]
            direction_list = sorted(list(directions))
            direction_str = '、'.join(direction_list)
            return f"{channel_name}位置{direction_str}向振动_STRUCTURED_DATASET"
        
        elif len(channels) > 1 and len(directions) == 1:
            # 多个通道，同一方向：通道名称C1、C2位置X向振动
            channel_list = sorted(list(channels))
            channel_str = '、'.join(channel_list)
            direction_name = list(directions)[0]
            return f"{channel_str}位置{direction_name}向振动_STRUCTURED_DATASET"
        
        elif len(channels) == 1 and len(directions) == 1:
            # 单一通道和方向：通道名称C1位置X向振动
            channel_name = list(channels)[0]
            direction_name = list(directions)[0]
            return f"{channel_name}位置{direction_name}向振动_STRUCTURED_DATASET"
        
        else:
            # 复杂情况：使用文件数量描述
            num_files = len(multi_file_results)
            return f"{num_files}文件合并_STRUCTURED_DATASET"


    def save_multiple_analysis_results(self):
        """保存多文件分析结果"""
        if not self.multi_file_analysis_results:
            QMessageBox.warning(self, "警告", "请先执行多文件功率谱分析！")
            return
        
        if not self.save_directory:
            QMessageBox.warning(self, "警告", "请先选择输出目录！")
            return
        
        try:
            # 询问保存方式
            reply = QMessageBox.question(
                self, 
                "选择保存方式", 
                "请选择多文件分析结果的保存方式：\n\n"
                "• 合并保存：将所有文件的功率谱分段合并保存为单个结构化数据集文件\n"
                "• 分别保存：将每个文件的功率谱分段分别保存为单独的结构化数据集文件\n\n"
                "点击'是'合并保存，点击'否'分别保存",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Cancel:
                return
            
            saved_files = []
            
            if reply == QMessageBox.Yes:
                # 合并保存所有文件的分析结果
                # 询问保存格式
                format_reply = QMessageBox.question(
                    self,
                    "选择保存格式",
                    "请选择合并数据集的保存格式：\n\n"
                    "• MATLAB格式 (.mat)：适合MATLAB分析\n"
                    "• CSV格式 (.csv)：适合Excel和Python分析\n\n"
                    "点击'是'保存为MATLAB格式，点击'否'保存为CSV格式",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                format_type = 'mat' if format_reply == QMessageBox.Yes else 'csv'
                
                # 合并所有文件的功率谱结果
                all_results = []
                file_info = []
                
                for i, multi_result in enumerate(self.multi_file_analysis_results):
                    analysis_result = multi_result['analysis_result']
                    file_name = multi_result['file_name']
                    channel = multi_result['channel']
                    direction = multi_result['direction']
                    
                    # 为每个分段添加文件标识信息
                    for j, result in enumerate(analysis_result['results']):
                        # 复制结果并添加文件信息
                        extended_result = result.copy()
                        extended_result['source_file'] = file_name
                        extended_result['source_channel'] = channel
                        extended_result['source_direction'] = direction
                        extended_result['file_index'] = i
                        extended_result['segment_index'] = j
                        all_results.append(extended_result)
                
                # 构建基础文件名 - 使用新的描述性命名逻辑
                base_filename = self.generate_descriptive_filename(self.multi_file_analysis_results)
                full_base_path = os.path.join(self.save_directory, base_filename)
                
                # 保存合并的结构化数据集
                saved_file = self.va_controller.result_saver.save_structured_dataset(
                    all_results,
                    product_code="MultiFile",
                    serial_number="Combined",
                    channel="Multiple",
                    direction="Combined",
                    base_path=full_base_path,
                    format=format_type,
                    additional_info={
                        'num_files': len(self.multi_file_analysis_results),
                        'total_segments': len(all_results),
                        'source_files': [r['file_name'] for r in self.multi_file_analysis_results],
                        'analysis_method': self.va_controller.current_params['method']
                    }
                )
                saved_files.append(saved_file)
                
                # 如果是CSV格式，还会生成元数据文件
                if format_type == 'csv':
                    meta_file = f"{full_base_path}_metadata.txt"
                    if os.path.exists(meta_file):
                        saved_files.append(meta_file)
                
                file_list = "\n".join(saved_files)
                QMessageBox.information(
                    self, 
                    "保存成功", 
                    f"多文件合并数据集已保存！\n\n"
                    f"文件信息：\n"
                    f"- 文件数量：{len(self.multi_file_analysis_results)}\n"
                    f"- 总分段数量：{len(all_results)}\n"
                    f"- 保存格式：{format_type.upper()}\n\n"
                    f"保存文件：\n{file_list}"
                )
                
            else:
                # 分别保存每个文件的分析结果
                total_saved = 0
                
                for multi_result in self.multi_file_analysis_results:
                    analysis_result = multi_result['analysis_result']
                    file_name = multi_result['file_name']
                    channel = multi_result['channel']
                    direction = multi_result['direction']
                    
                    # 询问每个文件的保存格式
                    format_reply = QMessageBox.question(
                        self,
                        f"选择 {file_name} 的保存格式",
                        f"请选择文件 {file_name} 的保存格式：\n\n"
                        "• MATLAB格式 (.mat)：适合MATLAB分析\n"
                        "• CSV格式 (.csv)：适合Excel和Python分析\n\n"
                        "点击'是'保存为MATLAB格式，点击'否'保存为CSV格式",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )
                    
                    format_type = 'mat' if format_reply == QMessageBox.Yes else 'csv'
                    
                    # 构建基础文件名 - 使用新的描述性命名逻辑
                    if channel and direction:
                        # 使用通道和方向信息构建文件名
                        base_filename = f"{channel}位置{direction}向振动_STRUCTURED_DATASET"
                    else:
                        # 回退到原始逻辑
                        base_filename = f"{file_name.replace('.', '_')}_STRUCTURED_DATASET"
                        
                    full_base_path = os.path.join(self.save_directory, base_filename)
                    
                    # 保存单个文件的结构化数据集
                    saved_file = self.va_controller.result_saver.save_structured_dataset(
                        analysis_result['results'],
                        product_code="Unknown",
                        serial_number="Unknown",
                        channel=channel,
                        direction=direction,
                        base_path=full_base_path,
                        format=format_type,
                        additional_info={
                            'source_file': file_name,
                            'num_segments': len(analysis_result['results']),
                            'analysis_method': self.va_controller.current_params['method']
                        }
                    )
                    saved_files.append(saved_file)
                    total_saved += 1
                
                QMessageBox.information(
                    self, 
                    "保存成功", 
                    f"多文件分析结果已分别保存！\n\n"
                    f"成功保存文件数：{total_saved}/{len(self.multi_file_analysis_results)}\n"
                    f"共保存了 {len(saved_files)} 个数据集文件"
                )
            
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"多文件结果保存失败: {str(e)}")
            
    def set_tick_font(self,ax):#解决对数符号问题
        '''
        对指定坐标轴的刻度字体进行设置，确保上标符号显示正常
        :param ax: 坐标轴对象（如ax1、ax2）
        :return: None
        '''
        # 使用DejaVu Sans字体，支持上标和负号
        tick_font = font_manager.FontProperties(family='DejaVu Sans', size=8)  # 大小可根据需要调整
        # 设置x轴刻度字体
        for labelx in ax.get_xticklabels():
            labelx.set_fontproperties(tick_font)
        # 设置y轴刻度字体
        for labely in ax.get_yticklabels():
            labely.set_fontproperties(tick_font)
        # 仅对需要整数刻度的轴启用（例如频域图x轴）
        # 这里不强制设置，避免影响时域图的时间轴（可能为小数） 
    def perform_psd_analysis(self):
        """执行功率谱分析"""
        if self.current_time_data is None or self.current_signal_data is None:
            QMessageBox.warning(self, "警告", "请先加载数据文件！")
            return
        
        try:
            # 获取文件信息
            file_name = os.path.basename(self.current_file_path)
            channel, direction = self.va_controller.data_loader.parse_channel_info(file_name)
            
            # 执行分析（使用默认参数）
            self.analysis_results = self.va_controller.analyze_file(
                self.current_file_path, 
                product_code="Unknown",
                serial_number="Unknown"
            )
            
            
            # 生成结构化特征集
            self.feature_dataset = self.va_controller.generate_feature_dataset(self.analysis_results)
            print(f"生成特征数据集: 形状 {self.feature_dataset.shape}")
            # 显示分析结果
            self.display_analysis_results()
            
            QMessageBox.information(self, "分析完成", 
                              f"功率谱分析已完成！\n生成特征数据集: {self.feature_dataset.shape[0]}段 x {self.feature_dataset.shape[1]}特征")
            
        except Exception as e:
            QMessageBox.warning(self, "分析失败", f"功率谱分析失败: {str(e)}")
    
    def display_analysis_results(self):
        """显示分析结果 - 分页显示"""
        if not self.analysis_results:
            return
        
        try:
            results = self.analysis_results['results']
            num_segments = len(results)
            
            # 计算总页数
            self.total_pages = (num_segments + self.results_per_page - 1) // self.results_per_page
            
            # 确保当前页在有效范围内
            if self.current_page >= self.total_pages:
                self.current_page = self.total_pages - 1
            if self.current_page < 0:
                self.current_page = 0
            
            # 清空当前图形
            self.figure.clear()
            
            # 计算当前页显示的分段范围
            start_idx = self.current_page * self.results_per_page
            end_idx = min(start_idx + self.results_per_page, num_segments)
            
            # 为当前页的分段创建子图
            num_to_show = end_idx - start_idx
            
            if num_to_show > 0:
                # 垂直排列显示当前页的分段
                for i in range(num_to_show):
                    result_idx = start_idx + i
                    result = results[result_idx]
                    
                    ax = self.figure.add_subplot(num_to_show, 1, i+1)
                    ax.plot(result['frequency'], result['power_spectrum'], 'b-', linewidth=1.5)
                    ax.set_xlabel('频率 (Hz)')
                    ax.set_ylabel('功率谱密度')
                    ax.set_title(f"时间段 {result_idx+1}/{num_segments}: {result['time_range'][0]:.1f}-{result['time_range'][1]:.1f}秒")
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                    ax.set_xscale('log')  # 新增：设置X轴为对数坐
                    self.set_tick_font(ax)
                # 调整布局，避免tight_layout警告
                self.figure.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, 
                                          hspace=0.5, wspace=0.3)
                
                # 添加总标题和页码信息
                self.figure.suptitle(
                    f"功率谱分析结果 - {self.analysis_results['channel']} {self.analysis_results['direction']} 第 {self.current_page + 1}/{self.total_pages} 页 (显示 {start_idx+1}-{end_idx} 段，共 {num_segments} 段)",
                    fontsize=12
                )
            else:
                # 没有数据可显示
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, '没有数据可显示', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
            self.canvas.draw()
            
            # 更新翻页按钮状态
            self.update_page_buttons()
            
        except Exception as e:
            QMessageBox.warning(self, "显示失败", f"结果显示失败: {str(e)}")
    def save_analysis_images(self):
        """保存分析结果图片"""
        if not self.analysis_results:
            QMessageBox.warning(self, "警告", "请先进行功率谱分析！")
            return
        
        if not self.save_directory:
            QMessageBox.warning(self, "警告", "请先选择保存目录！")
            return
        
        try:
            # 获取分析结果信息
            results = self.analysis_results['results']
            channel = self.analysis_results['channel']
            direction = self.analysis_results['direction']
            num_segments = len(results)
            
            # 创建保存目录
            save_dir = os.path.join(self.save_directory, "VA_view")
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存当前显示的页面图片
            if self.current_page >= 0 and self.current_page < self.total_pages:
                # 计算当前页显示的分段范围
                start_idx = self.current_page * self.results_per_page
                end_idx = min(start_idx + self.results_per_page, num_segments)
                
                # 生成文件名
                filename = f"{channel}_{direction}_page_{self.current_page + 1}_of_{self.total_pages}_segments_{start_idx + 1}_to_{end_idx}.png"
                file_path = os.path.join(save_dir, filename)
                
                # 保存当前显示的图片
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                
                # 询问是否保存所有页面图片
                reply = QMessageBox.question(self, "保存成功", 
                                           f"当前页面图片已保存到:\n{file_path}\n\n是否保存所有页面图片？",
                                           QMessageBox.Yes | QMessageBox.No)
                
                if reply == QMessageBox.Yes:
                    # 保存所有页面图片
                    for page in range(self.total_pages):
                        # 计算当前页显示的分段范围
                        start_idx = page * self.results_per_page
                        end_idx = min(start_idx + self.results_per_page, num_segments)
                        
                        # 生成文件名
                        filename = f"{channel}_{direction}_page_{page + 1}_of_{self.total_pages}_segments_{start_idx + 1}_to_{end_idx}.png"
                        file_path = os.path.join(save_dir, filename)
                        
                        # 创建临时图形来保存该页面
                        temp_fig = plt.figure(figsize=(10, 8))
                        
                        # 计算当前页显示的分段数量
                        num_to_show = end_idx - start_idx
                        
                        if num_to_show > 0:
                            # 垂直排列显示当前页的分段
                            for i in range(num_to_show):
                                result_idx = start_idx + i
                                result = results[result_idx]
                                
                                ax = temp_fig.add_subplot(num_to_show, 1, i+1)
                                ax.plot(result['frequency'], result['power_spectrum'], 'b-', linewidth=1.5)
                                ax.set_xlabel('频率 (Hz)')
                                ax.set_ylabel('功率谱密度')
                                ax.set_title(f"时间段 {result_idx+1}/{num_segments}: {result['time_range'][0]:.1f}-{result['time_range'][1]:.1f}秒")
                                ax.grid(True, alpha=0.3)
                                ax.set_yscale('log')
                                ax.set_xscale('log')  # 新增：设置X轴为对数坐
                                self.set_tick_font(ax)
                            # 调整布局
                            temp_fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, 
                                                   hspace=0.5, wspace=0.3)
                            
                            # 添加总标题和页码信息
                            temp_fig.suptitle(
                                f"功率谱分析结果 - {channel} {direction} 第 {page + 1}/{self.total_pages} 页 (显示 {start_idx+1}-{end_idx} 段，共 {num_segments} 段)",
                                fontsize=12
                            )
                        
                        # 保存图片
                        temp_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                        plt.close(temp_fig)
                    
                    QMessageBox.information(self, "保存完成", 
                                          f"所有页面图片已保存到:\n{save_dir}\n共保存了 {self.total_pages} 个图片文件")
            
            else:
                QMessageBox.warning(self, "保存失败", "没有有效的分析结果可保存")
                
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"图片保存失败: {str(e)}")


    def show_previous_page(self):
        """显示上一页"""
        if self.current_page > 0:
            self.current_page -= 1
            self.display_analysis_results()

    def show_next_page(self):
        """显示下一页"""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.display_analysis_results()

    def graphics_view_wheel_event(self, event):
        """滚轮事件处理 - 实现快速翻页功能"""
        if not self.analysis_results or self.total_pages <= 1:
            # 没有分析结果或只有一页，不处理滚轮事件
            event.ignore()
            return
        
        # 获取滚轮滚动的角度增量
        delta = event.angleDelta().y()
        
        if delta > 0:
            # 向上滚动滚轮 - 上一页
            if self.current_page > 0:
                self.current_page -= 1
                self.display_analysis_results()
        elif delta < 0:
            # 向下滚动滚轮 - 下一页
            if self.current_page < self.total_pages - 1:
                self.current_page += 1
                self.display_analysis_results()
        
        # 接受事件处理，阻止事件继续传播
        event.accept()
    

    def update_page_buttons(self):
        """更新翻页按钮状态"""
        if hasattr(self, 'pushButton_prev'):
            self.pushButton_prev.setEnabled(self.current_page > 0)
        if hasattr(self, 'pushButton_next'):
            self.pushButton_next.setEnabled(self.current_page < self.total_pages - 1)
        
        # 如果没有翻页按钮，在状态栏显示提示信息
        if not hasattr(self, 'pushButton_prev') and not hasattr(self, 'pushButton_next'):
            # 在图形标题中显示翻页提示
            if self.total_pages > 1:
                print(f"提示: 使用键盘左右箭头键翻页 (第 {self.current_page + 1}/{self.total_pages} 页)")    
    
    
    
    def save_analysis_results(self):
        """保存分析结果"""
        if not self.analysis_results:
            QMessageBox.warning(self, "警告", "请先执行功率谱分析！")
            return
        
        if not self.save_directory:
            QMessageBox.warning(self, "警告", "请先选择输出目录！")
            return
        
        try:
            # 获取文件信息
            file_name = os.path.basename(self.current_file_path)
            channel, direction = self.va_controller.data_loader.parse_channel_info(file_name)
            
            # 询问用户保存方式
            reply = QMessageBox.question(
                self, 
                "选择保存方式", 
                "请选择保存方式：\n\n"
                "• 单个文件：将所有功率谱分段保存为单个结构化数据集文件\n"
                "• 多个文件：将每个功率谱分段保存为单独的文件\n\n"
                "点击'是'保存为单个结构化数据集，点击'否'保存为多个单独文件",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Cancel:
                return
            
            saved_files = []
            
            if reply == QMessageBox.Yes:
                # 保存为结构化数据集
                # 询问保存格式
                format_reply = QMessageBox.question(
                    self,
                    "选择保存格式",
                    "请选择结构化数据集的保存格式：\n\n"
                    "• MATLAB格式 (.mat)：适合MATLAB分析\n"
                    "• CSV格式 (.csv)：适合Excel和Python分析\n\n"
                    "点击'是'保存为MATLAB格式，点击'否'保存为CSV格式",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                format_type = 'mat' if format_reply == QMessageBox.Yes else 'csv'
                
                # 构建基础文件名
                base_filename = f"{self.analysis_results['product_code']}_{self.analysis_results['serial_number']}"
                if channel:
                    base_filename += f"_{channel}"
                if direction:
                    base_filename += f"_{direction}"
                
                full_base_path = os.path.join(self.save_directory, base_filename)
                
                # 保存结构化数据集
                saved_file = self.va_controller.result_saver.save_structured_dataset(
                    self.analysis_results['results'],
                    product_code=self.analysis_results['product_code'],
                    serial_number=self.analysis_results['serial_number'],
                    channel=channel,
                    direction=direction,
                    base_path=full_base_path,
                    format=format_type
                )
                saved_files.append(saved_file)
                
                # 如果是CSV格式，还会生成元数据文件
                if format_type == 'csv':
                    meta_file = f"{full_base_path}_metadata.txt"
                    if os.path.exists(meta_file):
                        saved_files.append(meta_file)
                
                file_list = "\n".join(saved_files)
                QMessageBox.information(
                    self, 
                    "保存成功", 
                    f"结构化数据集已保存！\n\n"
                    f"文件信息：\n"
                    f"- 分段数量：{len(self.analysis_results['results'])}\n"
                    f"- 每段点数：{len(self.analysis_results['results'][0]['power_spectrum']) if self.analysis_results['results'] else 0}\n"
                    f"- 保存格式：{format_type.upper()}\n\n"
                    f"保存文件：\n{file_list}"
                )
                
            else:
                # 保存为多个单独文件
                saved_files = self.va_controller.save_analysis_results(
                    self.analysis_results,
                    self.save_directory
                )
                
                file_list = "\n".join(saved_files)
                QMessageBox.information(
                    self, 
                    "保存成功", 
                    f"分析结果已保存到以下文件：\n\n"
                    f"共保存了 {len(saved_files)} 个分段文件\n\n"
                    f"{file_list}"
                )
            
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"结果保存失败: {str(e)}")
    
 
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """振动分析工具使用说明:

1. 浏览数据文件 - 选择振动数据文件(.txt或.mat格式)
2. 选择输出目录 - 设置分析结果保存位置
3. 预览数据 - 显示时域和频域预览图
4. 功率谱分析 - 执行完整的功率谱分析
5. 保存结果 - 保存分析结果到指定目录
6. 参数设置 - 查看当前分析参数
7. 帮助 - 显示此帮助信息

支持的数据格式:
- .txt文件: 两列数据(时间, 振动值)
- .mat文件: 包含's'变量的MAT文件

文件命名建议:
- 包含通道和方向信息，如"A通道X方向振动.txt"
"""
        QMessageBox.information(self, "使用帮助", help_text)
    
class POP_VA_method_para(QMainWindow, Ui_VA_method_para, Ui_MainWindow):
    def __init__(self, parent=None):
        super(POP_VA_method_para, self).__init__()
        self.setupUi(self)
        self.parent_window = parent#保存主窗口的引用
        
        
        # 定义参数配置文件路径
        self.config_dir = os.path.join(os.getcwd(), "config")
        self.config_path = os.path.join(self.config_dir, "va_params.json")
        # 连接信号
        self.comboBox_segment_mode.currentTextChanged.connect(self.on_segment_mode_changed)
        # 连接时间参数变化信号，用于验证时间间隔
        self.doubleSpinBox_start_time.valueChanged.connect(self.validate_segment_duration)
        self.doubleSpinBox_end_time.valueChanged.connect(self.validate_segment_duration)
        self.doubleSpinBox_segment_duration.valueChanged.connect(self.validate_segment_duration)
        # 初始状态设置
        self.load_params()  # 加载历史参数
        
        # 初始状态设置
        self.on_segment_mode_changed(self.comboBox_segment_mode.currentText())
        
    def validate_segment_duration(self):
        """验证时间间隔不超过开始时间和结束时间的差值"""
        if self.comboBox_segment_mode.currentText() == "时间分段":
            start_time = self.doubleSpinBox_start_time.value()
            end_time = self.doubleSpinBox_end_time.value()
            segment_duration = self.doubleSpinBox_segment_duration.value()
            
            # 计算时间范围
            time_range = end_time - start_time
            
            if time_range <= 0:
                # 开始时间大于等于结束时间，显示警告
                self.doubleSpinBox_segment_duration.setStyleSheet("QDoubleSpinBox { background-color: #FFCCCC; }")
                self.doubleSpinBox_segment_duration.setToolTip("开始时间必须小于结束时间")
                return False
            elif segment_duration > time_range:
                # 时间间隔超过时间范围，显示警告并自动调整
                self.doubleSpinBox_segment_duration.setStyleSheet("QDoubleSpinBox { background-color: #FFCCCC; }")
                self.doubleSpinBox_segment_duration.setToolTip(f"时间间隔不能超过时间范围({time_range:.2f}s)")
                
                # 自动调整为最大允许值
                self.doubleSpinBox_segment_duration.blockSignals(True)  # 防止递归调用
                self.doubleSpinBox_segment_duration.setValue(time_range)
                self.doubleSpinBox_segment_duration.blockSignals(False)
                return False
            else:
                # 时间间隔在合理范围内，恢复正常样式
                self.doubleSpinBox_segment_duration.setStyleSheet("")
                self.doubleSpinBox_segment_duration.setToolTip("")
                return True
        return True

    def on_segment_mode_changed(self, mode):
        """根据分段模式启用/禁用相关控件"""
        if mode == "时间分段":
            # 时间段模式：启用开始/终止时间，禁用分段数量
            self.doubleSpinBox_start_time.setEnabled(True)
            self.doubleSpinBox_end_time.setEnabled(True)
            self.doubleSpinBox_segment_duration.setEnabled(True)
            self.spinBox_amounts.setEnabled(False)
        else:  # 固定长度模式
            # 固定长度模式：禁用开始/终止时间，启用分段数量
            self.doubleSpinBox_start_time.setEnabled(False)
            self.doubleSpinBox_end_time.setEnabled(False)
            self.doubleSpinBox_segment_duration.setEnabled(False)
            self.spinBox_amounts.setEnabled(True)
    
    
    def save_params(self):
        """保存当前参数到JSON文件"""
        # 收集所有需要记忆的参数
        params = {
            "analysis_method": self.comboBox_analysis_method.currentText(),
            "window_funtion": self.comboBox_window_funtion.currentText(),
            "overlap_rate": float(self.doubleSpinBox_overlap_rate.value()),
            "frequence": self.spinBox_frequence.value(),
            "number_psd": self.spinBox_number_psd.value(),
            "segment_mode": self.comboBox_segment_mode.currentText(),
            # 时间分段参数
            "start_time": float(self.doubleSpinBox_start_time.value()),
            "end_time": float(self.doubleSpinBox_end_time.value()),
            "segment_duration": float(self.doubleSpinBox_segment_duration.value()),
            # 固定长度参数
            "amounts": self.spinBox_amounts.value()
        }
        
        try:
            # 确保配置目录存在
            if not os.path.exists(self.config_dir):
                os.makedirs(self.config_dir)
            
            # 写入JSON文件
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(params, f, ensure_ascii=False, indent=4)
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"参数记忆失败：{str(e)}")
    
    def load_params(self):
        """从JSON文件加载历史参数"""
        if not os.path.exists(self.config_path):
            return  # 无历史参数，使用默认值
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            
            # 恢复参数到控件（按实际控件名称对应）
            self.comboBox_analysis_method.setCurrentText(params.get("analysis_method", ""))
            self.comboBox_window_funtion.setCurrentText(params.get("window_funtion", ""))
            self.doubleSpinBox_overlap_rate.setValue(params.get("overlap_rate", 0.5))
            self.spinBox_frequence.setValue(params.get("frequence", 5000))
            self.spinBox_number_psd.setValue(params.get("number_psd", 160))
            self.comboBox_segment_mode.setCurrentText(params.get("segment_mode", "时间分段"))
            
            # 恢复时间分段参数
            self.doubleSpinBox_start_time.setValue(params.get("start_time", 0.0))
            self.doubleSpinBox_end_time.setValue(params.get("end_time", 100.0))
            self.doubleSpinBox_segment_duration.setValue(params.get("segment_duration", 0.01))
            
            # 恢复固定长度参数
            self.spinBox_amounts.setValue(params.get("amounts", 1024))
            
        except Exception as e:
            QMessageBox.warning(self, "加载失败", f"历史参数加载失败：{str(e)}")
            
    def Confirm(self):
         # 读取输入参数
        global analysis_method, window_funtion, overlap_rate, frequence, amounts,number_psd
        analysis_method = self.comboBox_analysis_method.currentText()
        window_funtion = self.comboBox_window_funtion.currentText()
        overlap_rate = float(self.doubleSpinBox_overlap_rate.text())
        frequence = int(self.spinBox_frequence.text())
        amounts = int(self.spinBox_amounts.text())
        number_psd = int(self.spinBox_number_psd.text())
        # 读取分段模式相关参数
        segment_mode = self.comboBox_segment_mode.currentText()
        
        if segment_mode == "时间分段":
            # 时间段模式：使用开始/终止时间
            start_time = float(self.doubleSpinBox_start_time.text())
            end_time = float(self.doubleSpinBox_end_time.text())
            num_segments = None  # 自动计算分段数量
            segment_duration = float(self.doubleSpinBox_segment_duration.text())
        else:
            # 固定长度模式：使用分段数量
            start_time = None
            end_time = None
            amounts = int(self.spinBox_amounts.text())
            
            num_segments = None
            segment_duration = None  # 或者可以添加每段时长设置
        
        print("analysis_method:", analysis_method) 
        print("window_funtion:", window_funtion)
        print("overlap_rate:", overlap_rate)
        print("frequence:", frequence)
        print("amounts:", amounts)
        print("segment_mode:", segment_mode)
        print("start_time:", start_time)
        print("end_time:", end_time)
        print("num_segments:", num_segments)
        
        # 将参数传递给VA分析控制器
        if self.parent_window and hasattr(self.parent_window, 'va_controller'):
            # 根据分段模式设置nperseg参数
            if segment_mode == "时间分段":
                # 时间分段模式：nperseg应该为None，使用segment_duration
                nperseg_param = None
            else:
                # 固定长度模式：使用amounts作为nperseg
                nperseg_param = amounts
            # 设置分析参数
            self.parent_window.va_controller.set_analysis_params(
                method=analysis_method,
                window=window_funtion,
                overlap_ratio=overlap_rate,
                fs=frequence,
                nperseg=nperseg_param,
                start_time=start_time,
                end_time=end_time,
                num_segments=num_segments,
                segment_duration=segment_duration,
                number_psd=number_psd
            )

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
        """添加输入特征文件（支持多选）"""
        file_filter = "数据文件 (*.csv *.txt);;CSV文件 (*.csv);;文本文件 (*.txt);;所有文件 (*.*)"
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
        """添加输出特征文件（支持多选）"""
        file_filter = "数据文件 (*.csv *.txt);;CSV文件 (*.csv);;文本文件 (*.txt);;所有文件 (*.*)"
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
        """更新合并信息"""
        if self.input_files and self.output_files:
            try:
                # 读取输入文件（带标题行）
                input_df = pd.read_csv(self.input_files[0], sep="\s+|\t|,", engine='python')
                input_cols = input_df.shape[1]
                
                # 读取输出文件（带标题行）
                output_df = pd.read_csv(self.output_files[0], sep="\s+|\t|,", engine='python')
                # 计算P1到P160的列数
                p_cols = sum(1 for col in output_df.columns if col.startswith('P'))
                
                sample_count = min(len(input_df), len(output_df))
                
                self.lineEdit_info.setText(
                    f"预计合并结果: {sample_count} 样本 × {input_cols + p_cols} 特征 "
                    f"(输入: {input_cols}, 输出: {p_cols})"
                )
            except Exception as e:
                self.lineEdit_info.setText(f"预览失败：{str(e)[:50]}...")
        else:
            self.lineEdit_info.setText("请选择输入和输出特征文件")

    def combine_data(self):
        """合并输入输出数据（基于时间对齐）"""
        if not self.input_files or not self.output_files:
            QMessageBox.warning(self, "警告", "请先选择输入和输出特征文件！")
            return None
        
        try:
            # 读取所有输入文件并合并
            input_dfs = []
            for file_path in self.input_files:
                # 支持多种分隔符（空格、制表符、逗号）
                df = pd.read_csv(file_path, sep="\s+|\t|,", engine='python')
                # 确保包含必要的时间列
                required_cols = ['start_time', 'end_time']
                if not all(col in df.columns for col in required_cols):
                    QMessageBox.warning(self, "格式错误", f"输入文件 {os.path.basename(file_path)} 缺少必要的时间列")
                    return None
                input_dfs.append(df)
            
            # 合并所有输入文件
            input_combined = pd.concat(input_dfs, ignore_index=True)
            
            # 读取所有输出文件并合并
            output_dfs = []
            for file_path in self.output_files:
                df = pd.read_csv(file_path, sep="\s+|\t|,", engine='python')
                # 确保包含必要的时间列
                required_cols = ['start_time', 'end_time']
                if not all(col in df.columns for col in required_cols):
                    QMessageBox.warning(self, "格式错误", f"输出文件 {os.path.basename(file_path)} 缺少必要的时间列")
                    return None
                
                # 重命名P1-P160列，添加output后缀
                p_cols = [col for col in df.columns if col.startswith('P')]
                for col in p_cols:
                    # 提取数字部分，如P1→1
                    num = col[1:]
                    df.rename(columns={col: f"P{num}_output{num}"}, inplace=True)
                
                output_dfs.append(df)
            
            # 合并所有输出文件
            output_combined = pd.concat(output_dfs, ignore_index=True)
            
            # 改进的时间匹配策略：使用时间窗口中心点进行匹配
            # 计算每个时间窗口的中心点
            input_combined['time_center'] = (input_combined['start_time'] + input_combined['end_time']) / 2
            output_combined['time_center'] = (output_combined['start_time'] + output_combined['end_time']) / 2
            
            # 使用最近邻匹配，容忍0.1秒的时间差
            from scipy.spatial import cKDTree
            
            # 创建时间中心点的KD树
            input_centers = input_combined[['time_center']].values
            output_centers = output_combined[['time_center']].values
            
            # 构建KD树进行最近邻匹配
            tree = cKDTree(output_centers)
            distances, indices = tree.query(input_centers, k=1, distance_upper_bound=0.1)  # 容忍0.1秒的时间差
            
            # 创建匹配结果
            matched_indices = []
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                if dist < 0.1:  # 只保留时间差小于0.1秒的匹配
                    matched_indices.append((i, idx))
            
            if not matched_indices:
                QMessageBox.warning(self, "合并警告", "未找到时间匹配的记录，请检查文件时间范围是否一致")
                return None
            
            # 构建合并结果
            merged_rows = []
            for input_idx, output_idx in matched_indices:
                input_row = input_combined.iloc[input_idx].copy()
                output_row = output_combined.iloc[output_idx].copy()
                
                # 合并行，保留输入特征和输出特征
                merged_row = pd.concat([input_row, output_row.drop(['start_time', 'end_time', 'time_center'])], axis=0)
                merged_rows.append(merged_row)
            
            merged_df = pd.DataFrame(merged_rows)
            
            # 移除不需要的列（如segment_id）
            cols_to_drop = ['segment_id']
            merged_df = merged_df.drop(columns=[col for col in cols_to_drop if col in merged_df.columns])
            
            # 检查合并结果
            if merged_df.empty:
                QMessageBox.warning(self, "合并警告", "未找到时间匹配的记录，请检查文件时间范围是否一致")
                return None
                
            return merged_df

        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据合并失败:\n{str(e)}")
            return None

    def save_combined_data(self):
        """保存合并数据"""
        combined_df = self.combine_data()
        if combined_df is None:
            return
        # 在数据清洗后、保存前添加
        print("输出列数值统计：")
        print(combined_df[[col for col in combined_df.columns if col.startswith('P')]].describe())
        # 数据清洗
        # 1. 剔除包含 #NAME? 的行
        combined_df = combined_df[~combined_df.apply(lambda row: row.astype(str).str.contains('#NAME\?').any(), axis=1)]
        # 2. 剔除 inf/-inf/NaN 行
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.dropna(axis=0, how='any')

        # 3. 清洗中文字符
        def clean_chinese_characters(value):
            """清洗中文字符，将'1道'、'1类'等转换为'1'"""
            if isinstance(value, str):
                import re
                # 匹配以数字开头，后面跟着中文字符的模式
                pattern = r'^(\d+)[\u4e00-\u9fff]+$'
                match = re.match(pattern, value.strip())
                if match:
                    return match.group(1)
                # 如果是纯中文字符，返回NaN
                elif re.match(r'^[\u4e00-\u9fff]+$', value.strip()):
                    return np.nan
            return value
        
        # 对DataFrame的每一列应用中文清洗
        for col in combined_df.columns:
            combined_df[col] = combined_df[col].apply(clean_chinese_characters)

        # 再次剔除可能产生的NaN值
        combined_df = combined_df.dropna(axis=0, how='any')
    
        # 选择保存路径
        file_filter = "CSV文件 (*.csv);;Excel文件 (*.xlsx);;所有文件 (*.*)"
        initial_dir = self.lastSelectedPath if self.lastSelectedPath else "data/"
        save_path, _ = QFileDialog.getSaveFileName(self, "保存合并数据", initial_dir, file_filter)

        if not save_path:
            return

        try:
            # 计算输入和输出特征列数
            input_cols_count = sum(1 for col in combined_df.columns if not col.startswith('P'))
            output_cols_count = sum(1 for col in combined_df.columns if col.startswith('P'))
            
            if save_path.endswith(('.xls', '.xlsx')):
                # Excel保存：保留15位有效数字（避免科学计数法自动转换）
                combined_df.to_excel(save_path, index=False, engine="openpyxl", float_format="%.15g")
            else:
                # CSV保存：用科学计数法保留15位有效数字，确保极小值不被截断
                combined_df.to_csv(save_path, index=False, float_format="%.15g")  # 关键修改：调整float_format

            self.lastSelectedPath = os.path.dirname(save_path)

            success_msg = (
                f"数据保存成功!\n"
                f"文件: {os.path.basename(save_path)}\n"
                f"数据形状: {combined_df.shape}\n"
                f"输入特征: {input_cols_count} 列\n"
                f"输出特征: {output_cols_count} 列"
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
class POP_LR_para(QMainWindow, Ui_LR_para, Ui_MainWindow):
    def __init__(self, parent=None):
        super(POP_LR_para, self).__init__()
        self.setupUi(self)
        self.parent_window = parent  # 保存主窗口的引用

    def Confirm(self):
        # 读取输入参数
        global random_state, scale_features, method,fit_intercept
        scale_features = self.comboBox_scale_features.currentText() == "True"
        random_state = self.spinBox_random_state.value()
        fit_intercept = self.comboBox_fit_intercept.currentText() == "True"  # 新增fit_intercept参数读取
        method = 'LR'
        if self.parent_window:
            self.parent_window.lineEdit_Algorithm_name.setText("Linear Regression")
        self.parent_window.All_Methods_Begin()
        print("random_state:", random_state)
        print("scale_features:", scale_features)
 
 

class POP_help_para(QMainWindow, Ui_help_para, Ui_MainWindow):
    def __init__(self, parent=None):
        super(POP_help_para, self).__init__(parent)  # 初始化父类
        self.setupUi(self)  # 继承 Ui_MainWindow 界面类
        self.parent_window = parent#保存主窗口的引用

        self.pushButton_func_1.clicked.connect(self.open_file_1)
        self.pushButton_func_2.clicked.connect(self.open_file_2)
        self.pushButton_func_3.clicked.connect(self.open_file_3)
        self.pushButton_func_4.clicked.connect(self.open_file_4)
        self.pushButton_func_5.clicked.connect(self.open_file_5)
        self.pushButton_principle.clicked.connect(self.open_principle)
    def open_file_1(self):
        webbrowser.open("docs\振动分析使用说明.pdf")
    def open_file_2(self):
        webbrowser.open("docs\其他参数使用说明.pdf")
    def open_file_3(self):
        webbrowser.open("docs\数据集合成使用说明.pdf")
    def open_file_4(self):
        webbrowser.open("docs\训练验模使用说明.pdf")
    def open_file_5(self):
        webbrowser.open("docs\预测使用说明.pdf")
    def open_principle(self):
        webbrowser.open("docs\算法原理说明文档.pdf")
    
    def Confirm(self):
        pass
       
   
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
        
        
        # 进度条相关变量
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_value = 0
        self.progress_max = 100
        self.is_training = False
        
        # 初始化进度条
        self.progressBar.setValue(0)
        self.progressBar.setVisible(False)




        # 连接按钮信号
        
        self.pushButton_top.clicked.connect(self.show_previous_page)
        self.pushButton_bottom.clicked.connect(self.show_next_page)
        self.pushButton_help.clicked.connect(self.AL_help_para) # 打开帮助说明窗口
        self.pushButton_dataset.clicked.connect(self.AL_dataset_handle)  # 打开数据集合成处理窗口
        self.pushButton_save_pretrained_model.clicked.connect(
                    lambda: self.ask_and_save_model(self.trained_model, method)
                ) # 保存预训练模型
    def ask_and_save_model(self,model, method):
         ask_and_save_model(self, model, f"{method}_model.pkl")
         
         
    #进度条相关函数   
    def start_progress_indicator(self, max_value=100):
        """开始进度指示器"""
        self.progress_value = 0
        self.progress_max = max_value
        self.progressBar.setMaximum(max_value)
        self.progressBar.setValue(0)
        self.progressBar.setVisible(True)
        self.is_training = True
        
        # 启动定时器，每100ms更新一次进度
        self.progress_timer.start(100)
    
    def stop_progress_indicator(self):
        """停止进度指示器"""
        self.progress_timer.stop()
        self.progressBar.setValue(self.progress_max)
        self.is_training = False
        
        # 延迟隐藏进度条，让用户看到完成状态
        QTimer.singleShot(1000, lambda: self.progressBar.setVisible(False))
    
    def update_progress(self):
        """更新进度条显示"""
        if self.is_training:
            # 模拟进度增长，实际应用中应该根据实际训练进度更新
            if self.progress_value < self.progress_max:
                self.progress_value += 1
                self.progressBar.setValue(self.progress_value)
            else:
                self.stop_progress_indicator()
    
    def set_progress_value(self, value):
        """设置进度条的具体值"""
        if self.is_training:
            self.progress_value = min(value, self.progress_max)
            self.progressBar.setValue(self.progress_value)        




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
    def AL_Other_para(self):
    
        self.ui_pop = POP_Other_para(self)
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
        self.clear_interface()
        self.ui_pop.show()
 
    def AL_RF_para(self):
      
        self.ui_pop = POP_RF_para(self)
        self.clear_interface()
        self.ui_pop.show()

    def AL_SVM_para(self):
        
            self.ui_pop = POP_SVM_para(self)
            self.clear_interface()
            self.ui_pop.show()  
    
    def AL_MLP_para(self):
        
            self.ui_pop = POP_MLP_para(self)
            self.clear_interface()
            self.ui_pop.show()  

    def AL_ET_para(self):
        
            self.ui_pop = POP_ET_para(self)
            self.clear_interface()
            self.ui_pop.show() 

    def AL_GL_para(self):
        
            self.ui_pop = POP_GL_para(self)
            self.clear_interface()
            self.ui_pop.show()

    def AL_MTW_para(self):  
            
                self.ui_pop = POP_MTW_para(self)
                self.clear_interface()
                self.ui_pop.show()  

    def AL_REMTW_para(self):  
            
                self.ui_pop = POP_REMTW_para(self)
                self.clear_interface()
                self.ui_pop.show()

    def AL_MMoE_para(self):  
            
                self.ui_pop = POP_MMoE_para(self)
                self.clear_interface()
                self.ui_pop.show()
    
    def AL_GP_para(self):  
        
            self.ui_pop = POP_GP_para(self)
            self.clear_interface()
            self.ui_pop.show()

    def AL_LR_para(self):  # 添加线性回归参数窗口调用方法
        self.ui_pop = POP_LR_para(self)
        self.clear_interface()
        self.ui_pop.show()
    
    def AL_help_para(self):
        self.ui_pop = POP_help_para(self)
        self.clear_interface()
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
            
            # 关键修改：检查并保留时间列信息
            time_columns = ['start_time', 'end_time']
            self.time_columns_present = [col for col in time_columns if col in self.columns]            
            

            # 关键修改：根据列名前缀区分输入和输出特征
            # 筛选出输入特征列（包含"input"前缀）
            input_columns = [col for col in self.columns if "input" in col.lower()]
            # 筛选出输出特征列（包含"output"前缀）
            output_columns = [col for col in self.columns if "output" in col.lower()]
            
            # 关键修改：从输入特征中排除时间列，避免重复
            input_columns = [col for col in input_columns if col not in time_columns]            
            

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

        self.start_progress_indicator(100)
                # 定义MMoE专用的进度回调函数
        def mmoe_progress_callback(progress, message):
            # 将MMoE内部进度映射到整体进度 (40%-90%)
            mapped_progress = 40 + (progress * 0.5)  # 40%到90%的范围
            self.set_progress_value(int(mapped_progress))
        
        if self.data_load == 0:
            self.lineEdit_Algorithm_name.setText("请在右侧选择所需算法")
            self.lineEdit_state.setText('')
            return

        if method == 'NONE':
                    self.lineEdit_Algorithm_name.setText("Please select an Algorithm!!!")
                    self.lineEdit_state.setText('')
                    self.stop_progress_indicator()  # 停止进度指示
                    return
    

        if self.checkBox_percentage.isChecked():

            total_rows = self.data.shape[0]
            N_start_train = round(total_rows * int(self.spinBox_train_start.text()) / 100)
            N_end_train = round(total_rows * int(self.spinBox_train_end.text()) / 100)
            N_start_test = round(total_rows * int(self.spinBox_test_start.text()) / 100)
            N_end_test = round(total_rows * int(self.spinBox_test_end.text()) / 100)

            
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
            self.stop_progress_indicator()  # 停止进度指示
            return

        print('ok now')


        self.N_train = N_end_train - N_start_train
        self.N_test = N_end_test - N_start_test
        self.start_time = time.time()
      
        self.lineEdit_DEVICE.clear()

        #数据统一处理
        data = self.data
        input_columns, flag1 = self.get_input()
        output_columns, flag2 = self.get_output()
        if flag1 == 0 | flag2 == 0:
            self.lineEdit_Algorithm_name.setText("Select at least 1 input and 1 output!")
            self.lineEdit_state.setText('')
            self.stop_progress_indicator()  # 停止进度指示
            return

        # 更新进度到10%
        self.set_progress_value(10)


        data_X = data[input_columns]   # 选择输入特征
        data_y = data[output_columns]  # 选择输出特征   
        # 关键修改：检查并保留时间列
        time_columns = ['start_time', 'end_time']
        time_columns_present = [col for col in time_columns if col in data.columns]
        
        if len(time_columns_present) == 2:
            # 如果存在时间列，将它们包含在数据中
            data_time = data[time_columns_present]
            data_filter = pd.concat([data_X, data_y, data_time], axis=1)
            print(f"数据包含时间列: {time_columns_present}")
        else:
            # 如果没有时间列，使用原来的合并方式
            data_filter = pd.concat([data_X, data_y], axis=1)
            print("警告：数据缺少时间列，预测结果将不包含时间中心点")
        
        # 更新进度到20%
        self.set_progress_value(20)
        
                
        # 正确的shuffle逻辑：先打乱，再划分---shuffle
        if hasattr(self, 'shuffle_yes_or_no') and self.shuffle_yes_or_no.isChecked():
            # 如果启用shuffle，先打乱整个数据集
            data_filter = data_filter.sample(frac=1, random_state=int(random_state)).reset_index(drop=True)
            print("数据已随机打乱")
            
            # 打乱后，无论是否使用百分比模式，都应该从0开始划分
            # 因为数据已经打乱，原来的索引范围不再有意义
            if self.checkBox_percentage.isChecked():
                # 百分比模式：使用百分比计算新的索引范围
                N_start_train = 0
                N_end_train = round(len(data_filter) * int(self.spinBox_train_end.text()) / 100)
                N_start_test = N_end_train
                N_end_test = len(data_filter)
            else:
                # 绝对索引模式：重置为0开始的连续范围
                N_start_train = 0
                N_end_train = N_end_train - N_start_train  # 保持训练集大小不变
                N_start_test = N_end_train
                N_end_test = N_start_test + (N_end_test - N_start_test)  # 保持验证集大小不变

        # 更新进度到30%
        self.set_progress_value(30)
 
        data_train = data_filter[N_start_train : N_end_train]
        data_test = data_filter[N_start_test : N_end_test]   
        print('check')

        # 关键修改：确保测试集数据包含时间列信息
        time_columns_present_in_test = [col for col in time_columns if col in data_test.columns]
        
        if len(time_columns_present_in_test) == 2:
            print(f"测试集数据包含时间列: {time_columns_present_in_test}")
        else:
            print("警告：测试集数据缺少时间列，预测结果将不包含时间中心点")


        # 更新进度到40%
        self.set_progress_value(40)       
        
        if method == 'LR':  # 添加线性回归算法实现
            self.new_model = 1
            
            self.set_progress_value(50)
            
            model, _, y_test, y_pred, metrics = multi_task_regression_predictor(
                data_train,
                data_test,
                input_columns,
                output_columns,
                model_type='LR',
                scale_features=scale_features,
                random_state=int(random_state),
                fit_intercept=str(fit_intercept)
            )
            self.set_progress_value(80)
            
            self.trained_model = model  # 保存训练好的模型
            print("预测结果:", y_pred)
            print("真实值:", y_test)
            print("评估指标:", metrics)
            
            if len(output_columns) > 1:
                self.data_save = Multi_output_plot_and_evaluate(
                    self, y_test, y_pred, method, data_test,
                    output_columns, N_start_test, N_end_test,
                    MSE_list=metrics['MSE_list'],
                    RMSE_list=metrics['RMSE_list'],
                    MAE_list=metrics['MAE_list'],
                    R2_list=metrics['R2_list'],
                    db_within_3_ratio_list=metrics['db_within_3_ratio_list'],
                    total_db_deviation_per_feature=metrics['total_db_deviation_per_feature']
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
            self.lineEdit_DEVICE.setText("CPU")       

        
        if method == 'DT':
            
            self.new_model = 1
            self.set_progress_value(50)
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
            self.set_progress_value(80)
            self.trained_model = model  # 保存训练好的模型
            

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
            

        if method =='RF':
            self.new_model = 1
            self.set_progress_value(50)
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
            self.set_progress_value(80)
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
            
   

        if method =='SVM':
            self.new_model = 1
            self.set_progress_value(50)
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
                
                n_jobs=int(n_jobs),
                max_iter=-1,#SVM不需要这个参数
            )
            self.set_progress_value(80)
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
            

        if method == 'ET':
            self.new_model = 1
            self.set_progress_value(50)
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
            self.set_progress_value(80)
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
            


        if method == 'MLP':
            self.new_model = 1
            self.set_progress_value(50)
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
            self.set_progress_value(80)
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
            

        
        if method =='GP':
            self.new_model = 1
            self.set_progress_value(50)
            model,_, y_test, y_pred, metrics = multi_task_regression_predictor(
                data_train,
                data_test,
                input_columns,
                output_columns,
                model_type='GP',
                gp_learning_rate=float(learning_rate),
                gp_training_iterations=int(training_iterations),
                
            )
            self.set_progress_value(80)
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

            self.lineEdit_DEVICE.setText("GPU")
            
            

        
        if method == 'GL':
            self.new_model = 1
            self.set_progress_value(50)
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
            self.set_progress_value(80)
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
            self.set_progress_value(50)
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
            self.set_progress_value(80)
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
            self.set_progress_value(50)
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
            self.set_progress_value(80)
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
                mmoe_lambda_balance=float(mmoe_lambda_balance),
                progress_callback=mmoe_progress_callback  # 传递进度回调
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
            
        self.set_progress_value(100)
        self.stop_progress_indicator()      
    

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

    def clear_interface(self):
        """清空界面控件内容的工具函数"""
        print("正在清空界面...")
        
        # 清空文本框
        self.lineEdit_Algorithm_name.clear()
        self.lineEdit_DEVICE.clear()
        self.lineEdit_MSE.clear()
        self.lineEdit_RMSE.clear()
        self.lineEdit_MAE.clear()
        self.lineEdit_R2.clear()
        self.lineEdit_state.clear()
        self.lineEdit_dataset_nums.clear()
        self.lineEdit_db_within_3_ratio.clear()
        # self.lineEdit_total_db_deviation.clear()
        
        # 清空图表区域
        self.graphicscene.clear()
        self.graphicsView.setScene(self.graphicscene)
        
        # 清空分页数据
        self.current_page = 0
        self.figures = []
        
        print("界面清空完成")


if __name__ == '__main__':
    app = QApplication(sys.argv)  # 在 QApplication 方法中使用，创建应用程序对象
    myWin = MyMainWindow()  # 实例化 MyMainWindow 类，创建主窗口
    myWin.show()  # 在桌面显示控件 myWin
    sys.exit(app.exec_())  # 结束进程，退出程序
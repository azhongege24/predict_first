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
        num_groups = int(self.spinBox_num_groups.text())
        VA_data_path = str(self.lineEdit_VA_data_path.text())
        if self.parent_window:
            self.parent_window.VA_data_path = VA_data_path  # 将路径传递给主窗口
            self.parent_window.lineEdit_Algorithm_name.setText("Vibration Analysis")
            self.parent_window.lineEdit_state.setText("正在进行振动分析")

        print("sampling_rate:", sampling_rate) 
        print("num_groups:", num_groups)
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

    def Confirm(self):
        global loaded_model_path
        if self.parent_window:
            self.parent_window.lineEdit_Algorithm_name.setText("加载预训练模型")
        loaded_model_path = self.selected_model_path  # 这里只存路径
        print("selected_model_path:", loaded_model_path)

    def load_pretrained_model(self):
        # 只选择路径，不加载模型
        path = select_pretrained_model_path(self)
        self.selected_model_path = path
        print("Selected model path:", self.selected_model_path)
        if self.parent_window is not None:
            self.parent_window.selected_model_path = self.selected_model_path
                
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
        
        print("random_state:", random_state)
        print("alpha:", alpha)
        print("beta:", beta)
        print("max_iter:", max_iter)
        print("tol:", tol)
        
class MyMainWindow(QMainWindow, Ui_MainWindow):  # 继承 QMainWindow类和 Ui_MainWindow界面类
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)  # 初始化父类
        self.setupUi(self)  # 继承 Ui_MainWindow 界面类
        self.fileName = ''
        self.new_model = 0
        self.data_load = 0
        self.graphicscene = QGraphicsScene()
        self.lastSelectedPath = ""
        self.method = 'NONE'
        self.psd_results = None
        self.selected_model_path = None

        self.current_page = 0
        self.figures = []  # 存储所有图表的列表
        # 在UI初始化代码中添加


        # 初始化 VibrationAnalyzer 为 None，等待用户输入采样率后再实例化
        self.VA_data_path = ""



        # 连接按钮信号
        self.pushButton_top.clicked.connect(self.show_previous_page)
        self.pushButton_bottom.clicked.connect(self.show_next_page)
        self.pushButton_preview_VA_data.clicked.connect(self.preview_VA_data)
        self.pushButton_psd_analysis.clicked.connect(self.handle_vibration_analysis)
        self.pushButton_save_psd.clicked.connect(self.save_psd_result)
        
    
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
        # 使用上次路径作为初始目录，如果不存在则使用"../"
        # initial_dir = self.lastSelectedPath if self.lastSelectedPath else "./"原来的打开路径
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
                self.data = pd.read_csv(self.fileName,encoding='utf-8')

            # 公共数据处理流程
            self.data = self.data.dropna()#删除缺失值
            # self.columns = self.data.columns.tolist()
            self.data.columns = self.data.columns.astype(str)
            self.columns = self.data.columns.tolist()  # 确保是字符串列表

            # 显示数据信息
            self.listWidget_inputfeature.clear()
            self.listWidget_outputfeature.clear()
            self.add_listitem(self.columns[:-1], self.listWidget_inputfeature)
            self.add_listitem(self.columns[-1:], self.listWidget_outputfeature)
            self.shape = self.data.shape
            self.lineEdit_dataset_nums.setText(f'({self.shape[0]} Samples * {self.shape[1]} Features)')

            self.spinBox_train_end.setValue(self.shape[0]*0.9)
            self.spinBox_test_start.setValue(self.shape[0]*0.9+1)
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
            # ask_and_save_model(self,method,default_name='trained_model'+'_'+str(method)+'.pkl')    

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
                                                  metrics['MSE'],
                                                  metrics['R2'])
            if len(output_columns) == 1:
                self.data_save=single_plot_and_evaluate(self,y_test, y_pred,
                                        method, data_test,
                                        output_columns, N_start_test, 
                                        N_end_test,metrics['MSE'],
                                        metrics['R2'])
            self.lineEdit_DEVICE.setText("CPU")
            #这个是为了防止弹出保存模型的窗口而设置的延迟2秒功能  
            QTimer.singleShot(2000, lambda: ask_and_save_model(self, model, default_name='trained_model_' + str(method) + '.pkl'))

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
                                                  metrics['MSE'],
                                                  metrics['R2'])
            if len(output_columns) == 1:
                self.data_save=single_plot_and_evaluate(self,y_test, y_pred, method,
                                          data_test, output_columns, 
                                          N_start_test, N_end_test,
                                          metrics['MSE'],metrics['R2'])
            self.lineEdit_DEVICE.setText("CPU")
            #这个是为了防止弹出保存模型的窗口而设置的延迟2秒功能  
            QTimer.singleShot(2000, lambda: ask_and_save_model(self, model, default_name='trained_model_' + str(method) + '.pkl'))
   

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
                                                  metrics['MSE'],
                                                  metrics['R2'])
            if len(output_columns) == 1:
                self.data_save=single_plot_and_evaluate(self,y_test, y_pred, method, 
                                         data_test, output_columns,
                                           N_start_test, N_end_test,
                                           metrics['MSE'],metrics['R2'])
            self.lineEdit_DEVICE.setText("CPU")
            #这个是为了防止弹出保存模型的窗口而设置的延迟2秒功能  
            QTimer.singleShot(2000, lambda: ask_and_save_model(self, model, default_name='trained_model_' + str(method) + '.pkl'))




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
                                                  metrics['MSE'],
                                                  metrics['R2'])
            if len(output_columns) == 1:    
                self.data_save=single_plot_and_evaluate(self,y_test, y_pred, method,
                                                         data_test, output_columns, 
                                                         N_start_test, N_end_test,
                                                         metrics['MSE'],metrics['R2'])
            self.lineEdit_DEVICE.setText("CPU")
            #这个是为了防止弹出保存模型的窗口而设置的延迟2秒功能  
            QTimer.singleShot(2000, lambda: ask_and_save_model(self, model, default_name='trained_model_' + str(method) + '.pkl'))

        
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
                                                  metrics['MSE'],
                                                  metrics['R2'])
            if len(output_columns) == 1:
                self.data_save=single_plot_and_evaluate(self,y_test, y_pred, method, data_test,
                                           output_columns, N_start_test, N_end_test,
                                           metrics['MSE'],metrics['R2'])
            self.lineEdit_DEVICE.setText("CPU")
            #这个是为了防止弹出保存模型的窗口而设置的延迟2秒功能  
            QTimer.singleShot(2000, lambda: ask_and_save_model(self, model, default_name='trained_model_' + str(method) + '.pkl'))

        
        if method == 'GL':
            self.new_model = 1
            model, X_test, y_test, y_pred, metrics = group_lasso_predictor(
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
                    R2=metrics['R2']
                )
            self.lineEdit_DEVICE.setText("CPU")
    
        if method == 'MTW':
            self.new_model = 1
            model, X_test, y_test, y_pred, metrics = MTW_Lasso(
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
                    R2=metrics['R2']
                )
            self.lineEdit_DEVICE.setText("GPU")
        
        if method == 'REMTW':
            self.new_model = 1
            model, X_test, y_test, y_pred, metrics = REMTW_Lasso(
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
                    R2=metrics['R2']
                )
            self.lineEdit_DEVICE.setText("GPU")
           
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
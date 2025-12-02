# -*- coding: utf-8 -*-
"""
其他参数模块逻辑实现
功能：将多个其他参数文件与功率谱分析结果文件进行时间对齐
"""

import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QMessageBox, 
                             QProgressDialog, QApplication)
from PyQt5.QtCore import Qt
from .Other import Ui_Other

class OtherParameterModule(QMainWindow, Ui_Other):
    """其他参数模块主类"""
    
    def __init__(self, parent=None):
        super(OtherParameterModule, self).__init__()
        self.setupUi(self)
        self.parent_window = parent
        
        # 初始化数据存储
        self.other_parameter_files = []  # 存储其他参数文件路径
        self.target_file_path = None     # 目标对齐文件路径
        self.target_data = None          # 目标文件数据
        self.other_data_list = []        # 其他参数文件数据列表
        self.aligned_data = None         # 对齐后的数据
        self.product_name = ""           # 产品名称
        self.product_code = ""           # 产品代号
        # 连接信号槽
        self.pushButton_load_para.clicked.connect(self.load_other_parameter_files)
        self.pushButton_select_target_file.clicked.connect(self.select_target_alignment_file)
        self.pushButton_align_data.clicked.connect(self.perform_alignment)
        self.pushButton_save_data.clicked.connect(self.save_aligned_data)        
        # 连接产品信息输入框的信号
        self.product_name_input.textChanged.connect(self.on_product_info_changed)
        self.product_code_input.textChanged.connect(self.on_product_info_changed)
        # 更新初始状态
        self.update_status()



    def on_product_info_changed(self):
            """处理产品信息输入变化"""
            self.product_name = self.product_name_input.text().strip()
            self.product_code = self.product_code_input.text().strip()
            self.update_status()
    def load_other_parameter_files(self):
        """导入其他参数文件（支持多文件）"""
        file_filter = "数据文件 (*.csv *.txt *.xls *.xlsx);;所有文件 (*.*)"
        default_path = "./data/generated_files" if os.path.exists("./data/generated_files") else "./"
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择其他参数文件", default_path, file_filter)
        
        if files:
            self.other_parameter_files.extend(files)
            
            # 更新文件列表显示
            self.other_files_list.clear()
            for file_path in self.other_parameter_files:
                file_name = os.path.basename(file_path)
                self.other_files_list.addItem(file_name)
            
            self.update_status()
            QMessageBox.information(self, "导入成功", 
                                  f"成功导入 {len(files)} 个其他参数文件")
    
    def select_target_alignment_file(self):
        """选择需要对齐的目标文件（功率谱分析结果文件）"""
        file_filter = "CSV文件 (*.csv);;所有文件 (*.*)"
        default_path = "./data/data_va_analysis" if os.path.exists("./data/data_va_analysis") else "./"
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择目标对齐文件", default_path, file_filter)
        
        if file_path:
            self.target_file_path = file_path
            self.target_file_text.setText(os.path.basename(file_path))
            
            try:
                # 尝试加载目标文件数据
                self.target_data = pd.read_csv(file_path)
                self.update_status()
                QMessageBox.information(self, "选择成功", 
                                      "目标文件加载成功")
            except Exception as e:
                QMessageBox.warning(self, "加载失败", 
                                  f"无法加载目标文件: {str(e)}")
                self.target_file_path = None
                self.target_data = None
                self.target_file_text.clear()
    
    def load_other_parameter_data(self, file_path):
        """加载单个其他参数文件数据"""
        try:
            # 根据文件扩展名选择加载方式
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                data = pd.read_excel(file_path)
            elif file_path.endswith('.txt'):
                # 假设txt文件是空格或制表符分隔
                data = pd.read_csv(file_path, sep='\s+', header=None)
                # 如果只有两列，假设第一列是时间，第二列是参数值
                if data.shape[1] == 2:
                    data.columns = ['time', 'value']
            else:
                # 默认尝试CSV格式
                data = pd.read_csv(file_path)
            
            # 确保有时间列
            if 'time' not in data.columns and data.shape[1] >= 1:
                # 假设第一列是时间
                time_col = data.columns[0]
                data = data.rename(columns={time_col: 'time'})
            
            return data
        except Exception as e:
            raise Exception(f"加载文件 {os.path.basename(file_path)} 失败: {str(e)}")
    
    def perform_alignment(self):
        """执行数据对齐操作"""
        # 检查前置条件
        if not self.other_parameter_files:
            QMessageBox.warning(self, "警告", "请先导入其他参数文件")
            return
        
        if self.target_file_path is None:
            QMessageBox.warning(self, "警告", "请先选择目标对齐文件")
            return
        
        if self.target_data is None:
            QMessageBox.warning(self, "警告", "目标文件数据加载失败")
            return
        
        # 创建进度对话框
        progress = QProgressDialog("正在执行数据对齐...", "取消", 0, 
                                  len(self.other_parameter_files) + 3, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        try:
            progress.setValue(1)
            QApplication.processEvents()
            
            # 1. 加载所有其他参数文件数据
            self.other_data_list = []
            for i, file_path in enumerate(self.other_parameter_files):
                if progress.wasCanceled():
                    return
                
                progress.setValue(i + 1)
                progress.setLabelText(f"正在加载文件: {os.path.basename(file_path)}")
                QApplication.processEvents()
                
                try:
                    data = self.load_other_parameter_data(file_path)
                    self.other_data_list.append({
                        'file_path': file_path,
                        'file_name': os.path.basename(file_path),
                        'data': data
                    })
                except Exception as e:
                    QMessageBox.warning(self, "加载失败", 
                                      f"文件 {os.path.basename(file_path)} 加载失败: {str(e)}")
                    continue
            
            progress.setValue(len(self.other_parameter_files) + 1)
            progress.setLabelText("正在执行时间对齐...")
            QApplication.processEvents()
            
            # 2. 执行时间对齐
            self.aligned_data = self.time_align_data()
            
            progress.setValue(len(self.other_parameter_files) + 2)
            progress.setLabelText("对齐完成")
            QApplication.processEvents()
            
            # 3. 更新状态
            self.update_status()
            
            QMessageBox.information(self, "对齐成功", 
                                  f"数据对齐完成！\n生成特征数据集: {self.aligned_data.shape}")
            
        except Exception as e:
            QMessageBox.critical(self, "对齐失败", f"数据对齐过程中出错: {str(e)}")
        finally:
            progress.setValue(len(self.other_parameter_files) + 3)
    '''  
    老的对齐算法，是把所有的都放在一起的目标文件列和其他参数文件都列放在一起了 
    def time_align_data(self):
        """时间对齐核心算法"""
        # 检查目标文件是否包含必要的时间信息
        required_columns = ['start_time', 'end_time']
        missing_cols = [col for col in required_columns if col not in self.target_data.columns]
        if missing_cols:
            raise Exception(f"目标文件缺少必要的时间列: {missing_cols}")
        
        # 创建对齐后的数据副本
        aligned_df = self.target_data.copy()
        
        # 为每个其他参数文件创建对齐列
        for i, other_data_info in enumerate(self.other_data_list):
            other_data = other_data_info['data']
            file_name = other_data_info['file_name']
            
            # 检查其他参数文件是否包含时间列
            if 'time' not in other_data.columns:
                raise Exception(f"文件 {file_name} 缺少时间列")
            
            # 为每个时间段计算其他参数的平均值
            aligned_values = []
            
            for idx, row in self.target_data.iterrows():
                start_time = row['start_time']
                end_time = row['end_time']
                
                # 找到在当前时间段内的其他参数数据点
                time_mask = (other_data['time'] >= start_time) & (other_data['time'] <= end_time)
                time_data_in_range = other_data[time_mask]
                
                if len(time_data_in_range) > 0:
                    # 计算该时间段内参数值的平均值
                    if 'value' in time_data_in_range.columns:
                        avg_value = time_data_in_range['value'].mean()
                    else:
                        # 如果没有value列，使用第二列
                        value_col = time_data_in_range.columns[1]
                        avg_value = time_data_in_range[value_col].mean()
                else:
                    # 如果没有数据点，使用线性插值
                    try:
                        # 找到前后最近的数据点进行插值
                        before_data = other_data[other_data['time'] <= start_time]
                        after_data = other_data[other_data['time'] >= end_time]
                        
                        if len(before_data) > 0 and len(after_data) > 0:
                            before_time = before_data['time'].iloc[-1]
                            after_time = after_data['time'].iloc[0]
                            
                            if 'value' in other_data.columns:
                                before_value = before_data['value'].iloc[-1]
                                after_value = after_data['value'].iloc[0]
                            else:
                                value_col = other_data.columns[1]
                                before_value = before_data[value_col].iloc[-1]
                                after_value = after_data[value_col].iloc[0]
                            
                            # 线性插值
                            time_ratio = (start_time - before_time) / (after_time - before_time)
                            avg_value = before_value + time_ratio * (after_value - before_value)
                        else:
                            avg_value = np.nan
                    except:
                        avg_value = np.nan
                
                aligned_values.append(avg_value)
            
            # 添加对齐后的列到结果数据框
            column_name = f"other_param_{i+1}_{os.path.splitext(file_name)[0]}"
            aligned_df[column_name] = aligned_values
        
        return aligned_df
    
    def save_aligned_data(self):
        """保存对齐后的数据"""
        if self.aligned_data is None:
            QMessageBox.warning(self, "警告", "请先执行数据对齐操作")
            return
        
        file_filter = "CSV文件 (*.csv);;Excel文件 (*.xlsx);;所有文件 (*.*)"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存对齐数据", "aligned_dataset.csv", file_filter)
        
        if file_path:
            try:
                if file_path.endswith('.xlsx'):
                    self.aligned_data.to_excel(file_path, index=False)
                else:
                    self.aligned_data.to_csv(file_path, index=False)
                
                QMessageBox.information(self, "保存成功", 
                                      f"对齐数据已保存到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存文件时出错: {str(e)}")
    '''
    def time_align_data(self):
        """时间对齐核心算法"""
        # 检查目标文件是否包含必要的时间信息
        required_columns = ['start_time', 'end_time']
        missing_cols = [col for col in required_columns if col not in self.target_data.columns]
        if missing_cols:
            raise Exception(f"目标文件缺少必要的时间列: {missing_cols}")
        
        # 创建新的数据框，只包含时间信息和其他参数
        aligned_df = pd.DataFrame()
        
        # 添加时间列（从目标文件复制）
        aligned_df['start_time'] = self.target_data['start_time']
        aligned_df['end_time'] = self.target_data['end_time']
        
        
        # 添加产品信息作为特征列
        if self.product_name:
            aligned_df['product_name'] = self.product_name
        else:
            aligned_df['product_name'] = "未命名产品"
            
        if self.product_code:
            aligned_df['product_code'] = self.product_code
        else:
            aligned_df['product_code'] = "未指定代号"
            
            
        # 为每个其他参数文件创建对齐列
        for i, other_data_info in enumerate(self.other_data_list):
            other_data = other_data_info['data']
            file_name = other_data_info['file_name']
            
            # 检查其他参数文件是否包含时间列
            if 'time' not in other_data.columns:
                raise Exception(f"文件 {file_name} 缺少时间列")
            
            # 为每个时间段计算其他参数的平均值
            aligned_values = []
            
            for idx, row in self.target_data.iterrows():
                start_time = row['start_time']
                end_time = row['end_time']
                
                # 找到在当前时间段内的其他参数数据点
                time_mask = (other_data['time'] >= start_time) & (other_data['time'] <= end_time)
                time_data_in_range = other_data[time_mask]
                
                if len(time_data_in_range) > 0:
                    # 计算该时间段内参数值的平均值
                    if 'value' in time_data_in_range.columns:
                        avg_value = time_data_in_range['value'].mean()
                    else:
                        # 如果没有value列，使用第二列
                        value_col = time_data_in_range.columns[1]
                        avg_value = time_data_in_range[value_col].mean()
                else:
                    # 如果没有数据点，使用线性插值
                    try:
                        # 找到前后最近的数据点进行插值
                        before_data = other_data[other_data['time'] <= start_time]
                        after_data = other_data[other_data['time'] >= end_time]
                        
                        if len(before_data) > 0 and len(after_data) > 0:
                            before_time = before_data['time'].iloc[-1]
                            after_time = after_data['time'].iloc[0]
                            
                            if 'value' in other_data.columns:
                                before_value = before_data['value'].iloc[-1]
                                after_value = after_data['value'].iloc[0]
                            else:
                                value_col = other_data.columns[1]
                                before_value = before_data[value_col].iloc[-1]
                                after_value = after_data[value_col].iloc[0]
                            
                            # 线性插值
                            time_ratio = (start_time - before_time) / (after_time - before_time)
                            avg_value = before_value + time_ratio * (after_value - before_value)
                        else:
                            avg_value = np.nan
                    except:
                        avg_value = np.nan
                
                aligned_values.append(avg_value)
            
            # 添加对齐后的列到结果数据框
            column_name = f"other_param_{i+1}_{os.path.splitext(file_name)[0]}"
            aligned_df[column_name] = aligned_values
        
        return aligned_df

    def save_aligned_data(self):
        """保存对齐后的数据"""
        if self.aligned_data is None:
            QMessageBox.warning(self, "警告", "请先执行数据对齐操作")
            return
        
        file_filter = "CSV文件 (*.csv);;Excel文件 (*.xlsx);;所有文件 (*.*)"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存其他参数整合数据", "other_parameters_dataset.csv", file_filter)
        
        if file_path:
            try:
                if file_path.endswith('.xlsx'):
                    self.aligned_data.to_excel(file_path, index=False)
                else:
                    self.aligned_data.to_csv(file_path, index=False)
                
                QMessageBox.information(self, "保存成功", 
                                      f"其他参数整合数据已保存到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存文件时出错: {str(e)}")
 
    def update_status(self):
        """更新状态显示"""
        status_parts = []
        
        if self.other_parameter_files:
            status_parts.append(f"已导入 {len(self.other_parameter_files)} 个其他参数文件")
        
        if self.target_file_path:
            status_parts.append("已选择目标对齐文件")
        
        if self.aligned_data is not None:
            status_parts.append(f"已对齐完成 ({self.aligned_data.shape[0]}行×{self.aligned_data.shape[1]}列)")
        
        
        
        # 添加产品信息状态
        if self.product_name or self.product_code:
            product_info = []
            if self.product_name:
                product_info.append(f"产品: {self.product_name}")
            if self.product_code:
                product_info.append(f"代号: {self.product_code}")
            status_parts.append(" | ".join(product_info))
        
        if status_parts:
            status_text = " | ".join(status_parts)
        else:
            status_text = "请先导入其他参数文件和选择目标对齐文件"
        
        self.status_label.setText(status_text)
        
        # 更新文件信息
        file_info = []
        if self.target_data is not None:
            file_info.append(f"目标文件: {self.target_data.shape[0]}行×{self.target_data.shape[1]}列")
        
        if self.other_data_list:
            file_info.append(f"其他参数文件: {len(self.other_data_list)}个")
        
        self.file_info_label.setText("\n".join(file_info))

def POP_Other_para(parent=None):
    """创建其他参数模块窗口实例"""
    return OtherParameterModule(parent)
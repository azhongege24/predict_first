import os
import numpy as np
import scipy.io as sio
import pandas as pd
from datetime import datetime

class ResultSaver:
    """功率谱分析结果保存工具类"""
    
    def __init__(self):
        """初始化保存器"""
        self.supported_formats = ['mat', 'txt', 'xlsx', 'csv']
    
    def save_single_result(self, result, product_code, serial_number, 
                          channel, direction, file_path, format='mat', 
                          additional_info=None):
        """
        保存单个分析结果
        
        参数:
            result: 分析结果字典，包含'time_range', 'frequency', 'power_spectrum'
            product_code: 产品代号
            serial_number: 产品序号
            channel: 通道名称
            direction: 方向
            file_path: 保存路径(不含扩展名)
            format: 保存格式
            additional_info: 额外信息字典
        """
        if format not in self.supported_formats:
            raise ValueError(f"不支持的保存格式: {format}，支持的有: {self.supported_formats}")
            
        # 构建完整的元数据
        metadata = {
            'product_code': product_code,
            'serial_number': serial_number,
            'channel': channel,
            'direction': direction,
            'time_range': result['time_range'],
            'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **(additional_info or {})
        }
        
        # 构建完整文件路径
        full_path = f"{file_path}.{format}"
        
        # 根据格式保存
        if format == 'mat':
            self._save_mat(full_path, result, metadata)
        elif format == 'txt':
            self._save_txt(full_path, result, metadata)
        elif format in ['xlsx', 'csv']:
            self._save_table(full_path, result, metadata, format)
            
        return full_path
    
    def save_multiple_results(self, results, product_code, serial_number, 
                             channel, direction, base_path, format='mat',
                             additional_info=None):
        """保存多个分析结果"""
        saved_files = []
        
        # 创建目录(如果不存在)
        dir_name = os.path.dirname(base_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            
        for i, result in enumerate(results):
            # 为每个结果创建唯一的文件名
            time_str = f"{result['time_range'][0]:.2f}-{result['time_range'][1]:.2f}"
            file_path = f"{base_path}_segment_{i+1}_time_{time_str}"
            
            # 保存单个结果
            saved_file = self.save_single_result(
                result, product_code, serial_number, 
                channel, direction, file_path, format,
                additional_info
            )
            saved_files.append(saved_file)
            
        return saved_files
    
    def _save_mat(self, file_path, result, metadata):
        """保存为mat格式"""
        data = {
            'frequency': result['frequency'],
            'power_spectrum': result['power_spectrum'],
            'metadata': metadata
        }
        sio.savemat(file_path, data)
    
    def _save_txt(self, file_path, result, metadata):
        """保存为txt格式"""
        # 先写入元数据
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("=== 分析元数据 ===\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n=== 功率谱数据 ===\n")
            f.write("频率(Hz)\t功率谱\n")
            
            # 写入数据
            for freq, psd in zip(result['frequency'], result['power_spectrum']):
                f.write(f"{freq:.6f}\t{psd:.10f}\n")
    
    def _save_table(self, file_path, result, metadata, format='xlsx'):
        """保存为表格格式(xlsx或csv)"""
        # 创建数据框
        df = pd.DataFrame({
            '频率(Hz)': result['frequency'],
            '功率谱': result['power_spectrum']
        })
        
        # 对于Excel格式，创建一个包含元数据的工作表
        if format == 'xlsx':
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # 元数据工作表
                meta_df = pd.DataFrame(list(metadata.items()), columns=['属性', '值'])
                meta_df.to_excel(writer, sheet_name='元数据', index=False)
                
                # 数据工作表
                df.to_excel(writer, sheet_name='功率谱数据', index=False)
        else:  # csv格式
            # 先保存元数据
            meta_lines = ["=== 分析元数据 ==="]
            meta_lines.extend([f"{key}: {value}" for key, value in metadata.items()])
            meta_lines.append("=== 功率谱数据 ===")
            meta_lines.append("频率(Hz),功率谱")
            
            # 再保存数据
            data_lines = [f"{freq:.6f},{psd:.10f}" for freq, psd in 
                         zip(result['frequency'], result['power_spectrum'])]
            
            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(meta_lines + data_lines))
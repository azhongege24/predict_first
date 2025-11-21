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
    
    def save_structured_dataset(self, results, product_code, serial_number, 
                               channel, direction, base_path, format='mat',
                               additional_info=None):
            """
            保存结构化数据集
            
            参数:
                results: 分析结果列表，每个结果包含'frequency', 'power_spectrum'
                product_code: 产品代号
                serial_number: 产品序号
                channel: 通道名称
                direction: 方向
                base_path: 保存路径(不含扩展名)
                format: 保存格式 ('mat', 'csv')
                additional_info: 额外信息字典
                
            返回:
                保存的文件路径
            """
            if format not in ['mat', 'csv']:
                raise ValueError(f"结构化数据集不支持格式: {format}，支持的有: mat, csv")
            
            # 创建目录(如果不存在)
            dir_name = os.path.dirname(base_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            
            # 检查结果是否包含扩展信息（多文件合并）
            has_extended_info = any('source_file' in result for result in results)
            
            # 构建完整的元数据
            metadata = {
                'product_code': product_code,
                'serial_number': serial_number,
                'channel': channel,
                'direction': direction,
                'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'num_segments': len(results),
                'points_per_spectrum': len(results[0]['power_spectrum']) if results else 0,
                'frequency_range': {
                    'min': float(results[0]['frequency'][0]) if results else 0,
                    'max': float(results[0]['frequency'][-1]) if results else 0
                },
                'has_extended_info': has_extended_info,  # 标记是否包含多文件信息
                **(additional_info or {})
            }
            
            # 构建完整文件路径
            full_path = f"{base_path}_structured_dataset.{format}"
            
            # 提取所有功率谱数据
            power_spectra = []
            time_ranges = []
            source_files = []
            source_channels = []
            source_directions = []
            file_indices = []
            segment_indices = []
            
            for i, result in enumerate(results):
                power_spectra.append(result['power_spectrum'])
                time_ranges.append(result['time_range'])
                
                # 提取扩展信息（如果存在）
                if has_extended_info:
                    source_files.append(result.get('source_file', 'Unknown'))
                    source_channels.append(result.get('source_channel', 'Unknown'))
                    source_directions.append(result.get('source_direction', 'Unknown'))
                    file_indices.append(result.get('file_index', -1))
                    segment_indices.append(result.get('segment_index', -1))
            
            # 转换为numpy数组
            power_spectra_array = np.array(power_spectra)
            time_ranges_array = np.array(time_ranges)
            
            # 根据格式保存
            if format == 'mat':
                # 保存为MATLAB格式
                data = {
                    'power_spectra': power_spectra_array,  # 行数=分段数，列数=功率谱点数
                    'time_ranges': time_ranges_array,      # 每行对应一个时间段 [开始时间, 结束时间]
                    'frequency': results[0]['frequency'] if results else np.array([]),  # 频率轴
                    'metadata': metadata
                }
                
                # 添加扩展信息（如果存在）
                if has_extended_info:
                    data['source_files'] = np.array(source_files, dtype=object)
                    data['source_channels'] = np.array(source_channels, dtype=object)
                    data['source_directions'] = np.array(source_directions, dtype=object)
                    data['file_indices'] = np.array(file_indices)
                    data['segment_indices'] = np.array(segment_indices)
                
                sio.savemat(full_path, data)
                
            elif format == 'csv':
                # 保存为CSV格式
                # 创建列名：P1, P2, ..., Pn (n=功率谱点数)
                num_points = len(results[0]['power_spectrum']) if results else 0
                column_names = [f'P{i+1}' for i in range(num_points)]
                
                # 创建DataFrame
                df = pd.DataFrame(power_spectra_array, columns=column_names)
                
                # 添加时间范围信息
                df['start_time'] = time_ranges_array[:, 0]
                df['end_time'] = time_ranges_array[:, 1]
                df['segment_id'] = range(1, len(results) + 1)
                
                # 添加扩展信息（如果存在）
                if has_extended_info:
                    df['source_file'] = source_files
                    df['source_channel'] = source_channels
                    df['source_direction'] = source_directions
                    df['file_index'] = file_indices
                    df['segment_index'] = segment_indices
                
                # 保存CSV文件
                df.to_csv(full_path, index=False)
                
                # 保存元数据到单独的文件
                meta_path = f"{base_path}_metadata.txt"
                with open(meta_path, 'w', encoding='utf-8') as f:
                    f.write("=== 结构化数据集元数据 ===\n")
                    for key, value in metadata.items():
                        f.write(f"{key}: {value}\n")
                    f.write(f"\n数据结构说明:\n")
                    f.write(f"- 数据文件: {os.path.basename(full_path)}\n")
                    f.write(f"- 行数: {len(results)} (功率谱分段数量)\n")
                    f.write(f"- 列数: {num_points} (每个功率谱的点数)\n")
                    f.write(f"- 列名格式: P1, P2, ..., P{num_points} (对应功率谱密度值)\n")
                    f.write(f"- 基础列: start_time, end_time, segment_id\n")
                    if has_extended_info:
                        f.write(f"- 扩展列: source_file, source_channel, source_direction, file_index, segment_index\n")
            
            return full_path
    
    
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
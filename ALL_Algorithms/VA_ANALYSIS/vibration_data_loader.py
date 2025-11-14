import os
import numpy as np
import scipy.io as sio
import pandas as pd
from pathlib import Path
import h5py
class VibrationDataLoader:
    """振动数据读取工具类，支持txt和mat格式"""
    
    @staticmethod
    def parse_channel_info(filename):
        """从文件名解析通道信息和方向"""
        # 示例文件名: "A通道X方向振动.txt"
        name = os.path.splitext(filename)[0]
        channel = None
        direction = None
        
        if "通道" in name and "方向" in name:
            channel_part = name.split("通道")[0]
            dir_part = name.split("通道")[1].split("方向")[0]
            channel = f"{channel_part}通道"
            direction = f"{dir_part}方向"
        # 尝试格式2: "通道名称C1位置X向振动.txt"
        elif "通道名称" in name and "位置" in name and "向" in name:
            try:
                # 提取通道信息（通道名称C1）
                channel_start = name.find("通道名称")
                pos_start = name.find("位置")
                if channel_start != -1 and pos_start != -1:
                    channel = name[channel_start:pos_start]  # "通道名称C1"
                
                # 提取方向信息（X向）
                xiang_start = name.find("向")
                if pos_start != -1 and xiang_start != -1:
                    direction_part = name[pos_start+2:xiang_start]  # "X"
                    direction = f"{direction_part}向"
            except:
                pass
        
        # 如果还是无法解析，使用默认值
        if not channel and not direction:
            # 尝试从文件名中提取一些基本信息
            if "通道" in name:
                parts = name.split("通道")
                if len(parts) > 1:
                    channel = f"通道{parts[1][:2]}"  # 尝试获取通道号
            
            # 寻找方向相关的字符
            directions = ['X', 'Y', 'Z', 'x', 'y', 'z']
            for dir_char in directions:
                if dir_char in name:
                    direction = f"{dir_char.upper()}向"
                    break
            
        return channel, direction
    
    @staticmethod
    def read_txt_file(file_path):
        """读取txt格式的振动数据"""
        try:
            # 尝试用空格分隔
            data = np.loadtxt(file_path)
            if data.ndim == 2 and data.shape[1] == 2:
                return data[:, 0], data[:, 1]  # 时间，振动量值
            
            # 尝试用其他分隔符
            df = pd.read_csv(file_path, sep=None, engine='python', header=None)
            if len(df.columns) >= 2:
                return df[0].values, df[1].values
                
            raise ValueError("txt文件格式不正确，需要两列数据")
        except Exception as e:
            raise Exception(f"读取txt文件失败: {str(e)}")
    
    @staticmethod
    def read_mat_file_v7_3(file_path):
        """读取MATLAB v7.3格式的振动数据（使用HDF5）"""

        
        try:
            with h5py.File(file_path, 'r') as f:
                # 检查文件是否包含's'变量
                if 's' in f:
                    s_data = f['s'][:]
                    # 转置数据，因为HDF5存储方式与MATLAB不同
                    if s_data.ndim == 2 and s_data.shape[0] == 2:
                        return s_data[0, :], s_data[1, :]  # 时间，振动量值
                    elif s_data.ndim == 2 and s_data.shape[1] == 2:
                        return s_data[:, 0], s_data[:, 1]  # 时间，振动量值
                    else:
                        raise ValueError("mat文件中的变量s格式不正确")
                else:
                    # 尝试查找其他可能的变量名
                    possible_vars = ['data', 'signal', 'vibration', 'time', 't', 'x', 'y']
                    for var_name in possible_vars:
                        if var_name in f:
                            data = f[var_name][:]
                            if data.ndim == 2 and data.shape[1] == 2:
                                return data[:, 0], data[:, 1]
                            elif data.ndim == 1:
                                # 如果是单列数据，创建时间序列
                                time_data = np.arange(len(data))
                                return time_data, data
                    
                    raise ValueError("mat文件中未找到合适的变量")
        except Exception as e:
            raise Exception(f"读取MATLAB v7.3格式文件失败: {str(e)}")
    
    
    @staticmethod
    def read_mat_file(file_path):
        """读取mat格式的振动数据"""
        try:
            mat_data = sio.loadmat(file_path)
            if 's' in mat_data:
                s = mat_data['s']
                if s.ndim == 2 and s.shape[1] == 2:
                    return s[:, 0].flatten(), s[:, 1].flatten()  # 时间，振动量值
                else:
                    raise ValueError("mat文件中的变量s格式不正确")
            else:
                raise ValueError("mat文件中未找到变量s")
        except Exception as e:
            # 检查是否是v7.3格式错误
            error_msg = str(e)
            if "HDF reader for matlab v7.3" in error_msg or "v7.3" in error_msg:
                # 如果是v7.3格式，使用HDF5读取器
                return VibrationDataLoader.read_mat_file_v7_3(file_path)
            else:
                # 其他错误，抛出原始异常
                raise Exception(f"读取mat文件失败: {str(e)}")
    
    @staticmethod
    def load_data(file_path):
        """根据文件扩展名自动选择读取方式"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.txt':
            return VibrationDataLoader.read_txt_file(file_path)
        elif file_ext == '.mat':
            return VibrationDataLoader.read_mat_file(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
    
    @staticmethod
    def get_file_structure(root_dir):
        """获取数据文件的目录结构：产品代号 -> 产品序号 -> 通道文件"""
        structure = {}
        root_path = Path(root_dir)
        
        if not root_path.is_dir():
            return structure
            
        # 一级目录：产品代号
        for product_dir in root_path.iterdir():
            if product_dir.is_dir():
                product_code = product_dir.name
                structure[product_code] = {}
                
                # 二级目录：产品序号
                for serial_dir in product_dir.iterdir():
                    if serial_dir.is_dir():
                        serial_number = serial_dir.name
                        structure[product_code][serial_number] = []
                        
                        # 收集通道文件
                        for file in serial_dir.iterdir():
                            if file.is_file() and file.suffix.lower() in ['.txt', '.mat']:
                                structure[product_code][serial_number].append(str(file))
        
        return structure
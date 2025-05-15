import numpy as np
import pandas as pd
from pathlib import Path
import time

class WideFormatVibrationGenerator:
    def __init__(self, sampling_rate=1000):
        """
        宽格式振动数据生成器
        :param sampling_rate: 采样频率（Hz）
        """
        self.sampling_rate = sampling_rate
    
    def generate_wide_format_data(
        self,
        duration: float = 10.0,
        groups: int = 3,
        base_freq_range: tuple = (30, 300),
        noise_level: float = 0.2
    ) -> pd.DataFrame:
        """
        生成宽格式振动数据（每列一个组）
        :param duration: 信号时长（秒）
        :param groups: 数据组数（列数）
        :param base_freq_range: 基础频率范围（Hz）
        :param noise_level: 噪声强度系数（0-1）
        """
        np.random.seed(int(time.time() % 1000))  # 动态随机种子
        
        # 生成时间轴
        n_points = int(duration * self.sampling_rate)
        time_axis = np.linspace(0, duration, n_points)
        
        # 初始化DataFrame
        df = pd.DataFrame({'Time (s)': time_axis})
        
        # 为每组生成特征信号
        for group_num in range(1, groups+1):
            # 生成唯一的特征频率
            base_freq = np.random.uniform(*base_freq_range)
            
            # 生成主振动信号
            main_signal = np.sin(2 * np.pi * base_freq * time_axis)
            
            # 添加谐波成分
            harmonics = sum(
                np.sin(2 * np.pi * n * base_freq * time_axis) * 0.3**n
                for n in range(2, 5)
            )
            
            # 添加随机噪声
            noise = noise_level * np.random.randn(n_points)
            
            # 组合信号
            vibration = np.round(main_signal + harmonics + noise, 6)
            
            # 添加到DataFrame
            df[f'Group_{group_num} (g)'] = vibration
        
        return df
    
    def save_wide_format_csv(
        self,
        df: pd.DataFrame,
        filename: str,
        metadata: dict = None
    ):
        """
        保存宽格式数据到CSV（带可选元数据）
        :param df: 生成的DataFrame
        :param filename: 输出文件名
        :param metadata: 元数据字典（如设备信息）
        """
  
        

        df.to_csv(filename, index=False)
        
        print(f"成功保存到：{Path(filename).resolve()}")

# 使用示例--------------------------------------------------
if __name__ == "__main__":
    # 初始化生成器（采样率2kHz）
    gen = WideFormatVibrationGenerator(sampling_rate=1000)
    
    # 生成3组10秒数据
    df_data = gen.generate_wide_format_data(
        duration=10.0,
        groups=1,
        base_freq_range=(50, 500),
        noise_level=0.3
    )
    
    
    # 保存文件
    gen.save_wide_format_csv(
        df_data,
        "vibration_groups_single.csv",
        metadata=None
    )
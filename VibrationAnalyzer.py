import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd
import json
import os

class VibrationAnalyzer:
    def __init__(self, sampling_rate=1000):
        """
        初始化振动分析器
        :param sampling_rate: 采样率(Hz)
        """
        self.sampling_rate = sampling_rate
        
    def calculate_psd(self, vibration_data, nperseg=256):
        """
        计算加速度功率谱密度(PSD)
        :param vibration_data: 振动数据(单组或多组)
        :param nperseg: 每个段的长度(点数)
        :return: (frequencies, psd) 或 多组数据的PSD字典
        """
        if isinstance(vibration_data, dict):
            # 多组数据处理
            results = {}
            for name, data in vibration_data.items():
                f, Pxx = signal.welch(data, fs=self.sampling_rate, nperseg=nperseg)
                results[name] = {'frequencies': f.tolist(), 'psd': Pxx.tolist()}
            return results
        else:
            # 单组数据处理
            f, Pxx = signal.welch(vibration_data, fs=self.sampling_rate, nperseg=nperseg)
            return f, Pxx
            
    def analyze_and_save(self, vibration_data, output_path, format='json'):
        """
        分析并保存结果
        :param vibration_data: 振动数据(单组或多组)
        :param output_path: 输出路径
        :param format: 保存格式('json'或'csv')
        """
        result = self.calculate_psd(vibration_data)
        
        if isinstance(vibration_data, dict):
            # 多组数据保存
            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=4)#indent=4为缩进级别
            elif format == 'csv':
                # 将所有数据合并到一个DataFrame中
                dfs = []
                for name, data in result.items():
                    df = pd.DataFrame({
                        'frequency': data['frequencies'],
                        'psd': data['psd'],
                        'group': name
                    })
                    dfs.append(df)
                combined_df = pd.concat(dfs)
                combined_df.to_csv(output_path, index=False)
        else:
            # 单组数据保存
            f, Pxx = result
            if format == 'json':
                data = {
                    'frequencies': f.tolist(),
                    'psd': Pxx.tolist(),
                    'sampling_rate': self.sampling_rate
                }
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=4)
            elif format == 'csv':
                df = pd.DataFrame({'frequency': f, 'psd': Pxx})
                df.to_csv(output_path, index=False)
    
    @staticmethod
    def generate_test_data(num_samples=1000, num_groups=3):
        """
        生成测试数据
        :param num_samples: 每组样本数
        :param num_groups: 组数
        :return: 单组或多组测试数据
        """
        if num_groups == 1:
            # 单组数据
            t = np.linspace(0, 1, num_samples)
            return 0.5 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.normal(size=num_samples)
        else:
            # 多组数据
            data = {}
            for i in range(num_groups):
                t = np.linspace(0, 1, num_samples)
                freq = 30 + i * 20
                data[f'group_{i+1}'] = 0.5 * np.sin(2 * np.pi * freq * t) + 0.1 * np.random.normal(size=num_samples)
            return data
        
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = VibrationAnalyzer(sampling_rate=1000)
            # 测试单组数据
    single_data = analyzer.generate_test_data(num_groups=1)
    f, psd = analyzer.calculate_psd(single_data)
    analyzer.analyze_and_save(single_data, 'Vibration_Analyzer_data/single_result.csv', format='csv')

    # 测试多组数据
    multi_data = analyzer.generate_test_data(num_groups=3)
    results = analyzer.calculate_psd(multi_data)
    analyzer.analyze_and_save(multi_data, 'Vibration_Analyzer_data/multi_result.csv', format='csv')
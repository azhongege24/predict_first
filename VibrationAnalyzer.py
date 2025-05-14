from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd
import json
import os
import scipy.io as sio
from PyQt5.QtWidgets import QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView

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
            
    def analyze_data(self, vibration_data):
        """
        分析振动数据，返回分析结果
        :param vibration_data: 振动数据(单组或多组)
        :return: 分析结果 (单组: (frequencies, psd)，多组: 字典)
        """
        return self.calculate_psd(vibration_data)
    
    def save_analysis_result(self, result):
        """
        保存分析结果到文件，支持 JSON、CSV 和 MATLAB 格式
        :param result: 分析结果 (单组: (frequencies, psd)，多组: 字典)
        """
        # 打开文件保存对话框
        file_save, _ = QFileDialog.getSaveFileName(
            None, "保存文件", "",
            "JSON Files (*.json);;CSV Files (*.csv);;MATLAB Files (*.mat)"
        )
        if file_save:
            try:
                # 根据文件扩展名选择保存方式
                if file_save.endswith('.mat'):
                    # 将结果转换为 MATLAB 兼容格式
                    if isinstance(result, dict):
                        mat_data = {name: {'frequencies': data['frequencies'], 'psd': data['psd']} for name, data in result.items()}
                    else:
                        f, Pxx = result
                        mat_data = {'frequencies': f, 'psd': Pxx}
                    sio.savemat(file_save, mat_data)
                elif file_save.endswith('.json'):
                    # 保存为 JSON 文件
                    with open(file_save, 'w') as f:
                        json.dump(result, f, indent=4)
                elif file_save.endswith('.csv'):
                    # 保存为 CSV 文件
                    if isinstance(result, dict):
                        # 多组数据保存
                        dfs = []
                        for name, data in result.items():
                            df = pd.DataFrame({
                                'frequency': data['frequencies'],
                                'psd': data['psd'],
                                'group': name
                            })
                            dfs.append(df)
                        combined_df = pd.concat(dfs)
                        combined_df.to_csv(file_save, index=False)
                    else:
                        # 单组数据保存
                        f, Pxx = result
                        df = pd.DataFrame({'frequency': f, 'psd': Pxx})
                        df.to_csv(file_save, index=False)

                # 显示保存成功信息
                print("Data saved successfully")
            except Exception as e:
                # 捕获异常并显示错误信息
                print(f"Save failed: {str(e)}")
    @staticmethod
    def generate_test_data(file_path=None, num_samples=1000, num_groups=3):
        """
        生成测试数据或从外部文件加载数据
        :param file_path: 外部数据文件路径 (可选)
        :param num_samples: 每组样本数 (仅在生成测试数据时使用)
        :param num_groups: 组数 (仅在生成测试数据时使用)
        :return: 单组或多组测试数据
        """
        if file_path:
            # 从外部文件加载数据
            try:
                if file_path.endswith(('.xls', '.xlsx')):
                    # 读取 Excel 文件
                    data = pd.read_excel(file_path)
                elif file_path.endswith('.mat'):
                    # 读取 MATLAB 文件
                    mat_data = sio.loadmat(file_path)
                    # 假设 MAT 文件中包含 'data' 键
                    data = pd.DataFrame(mat_data['data'])
                elif file_path.endswith('.csv'):
                    # 读取 CSV 文件
                    data = pd.read_csv(file_path)
                else:
                    raise ValueError("Unsupported file format. Please use .csv, .xls, .xlsx, or .mat")

                print(f"成功加载数据文件: {file_path}")
                return data
            except Exception as e:
                print(f"加载数据文件失败: {str(e)}")
                return None
        else:
            # 生成测试数据
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
            
    def visualize_data(self, vibration_data, psd_result=None):
        """
        可视化振动数据和 PSD 分析结果
        :param vibration_data: 振动数据(单组或多组)
        :param psd_result: PSD 分析结果(可选)
        """
        if isinstance(vibration_data, dict):
            # 多组数据可视化
            for name, data in vibration_data.items():
                plt.figure(figsize=(10, 4))
                
                # 绘制振动数据
                plt.subplot(1, 2, 1)
                t = np.linspace(0, len(data) / self.sampling_rate, len(data))
                plt.plot(t, data, label=f'{name} - Raw Data')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                plt.title(f'{name} - Vibration Data')
                plt.legend()
                plt.grid()

                # 绘制 PSD 分析结果
                if psd_result and name in psd_result:
                    plt.subplot(1, 2, 2)
                    frequencies = psd_result[name]['frequencies']
                    psd = psd_result[name]['psd']
                    plt.semilogy(frequencies, psd, label=f'{name} - PSD')
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('PSD')
                    plt.title(f'{name} - Power Spectral Density')
                    plt.legend()
                    plt.grid()

                plt.tight_layout()
                plt.show()
        else:
            # 单组数据可视化
            plt.figure(figsize=(10, 4))
            
            # 绘制振动数据
            plt.subplot(1, 2, 1)
            t = np.linspace(0, len(vibration_data) / self.sampling_rate, len(vibration_data))
            plt.plot(t, vibration_data, label='Raw Data')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title('Vibration Data')
            plt.legend()
            plt.grid()

            # 绘制 PSD 分析结果
            if psd_result:
                frequencies, psd = psd_result
                plt.subplot(1, 2, 2)
                plt.semilogy(frequencies, psd, label='PSD')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('PSD')
                plt.title('Power Spectral Density')
                plt.legend()
                plt.grid()

            plt.tight_layout()
            plt.show()    
    def preview_data(self, file_path):
        """
        预览从外部文件导入的振动数据（单组或多组），并在 UI 的 QGraphicsView 上显示
        :param file_path: 数据文件路径
        """
        try:
            # 读取文件
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                data = pd.read_excel(file_path)
            else:
                raise ValueError("仅支持 CSV 或 Excel 文件格式")

            # 创建 Matplotlib 图形
            figure = Figure()
            canvas = FigureCanvas(figure)
            ax = figure.add_subplot(111)

            # 判断数据结构
            if 'group' in data.columns:
                # 多组数据处理
                groups = data['group'].unique()
                for group in groups:
                    group_data = data[data['group'] == group]
                    ax.plot(group_data['frequency'], group_data['psd'], label=f'{group}')
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('PSD')
                ax.set_title('Group Data')
                ax.legend()
            elif 'frequency' in data.columns and 'psd' in data.columns:
                # 单组数据处理
                ax.plot(data['frequency'], data['psd'], label='Single Group')
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('PSD')
                ax.set_title('Single Group Data')
                ax.legend()
            else:
                raise ValueError("文件格式不正确，缺少必要的列（frequency, psd 或 group）")

            # 绘制图形
            canvas.draw()

            # 更新图形到界面
            self.graphicscene = QGraphicsScene()
            self.graphicscene.addWidget(canvas)
            self.graphicsView.setScene(self.graphicscene)
            self.graphicsView.setDragMode(QGraphicsView.ScrollHandDrag)
            self.graphicsView.show()

        except Exception as e:
            print(f"数据预览失败: {str(e)}")
        
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = VibrationAnalyzer(sampling_rate=1000)
            # 测试单组数据
    single_data = analyzer.generate_test_data(num_groups=1)
    f, psd = analyzer.calculate_psd(single_data)
    analyzer.visualize_data(single_data, psd_result=(f, psd))
   

    # 测试多组数据
    multi_data = analyzer.generate_test_data(num_groups=3)
    results = analyzer.calculate_psd(multi_data)
    analyzer.visualize_data(multi_data, psd_result=results)

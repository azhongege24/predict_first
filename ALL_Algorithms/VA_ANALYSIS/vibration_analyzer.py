import os
from pathlib import Path
from .vibration_data_loader import VibrationDataLoader
from .data_annotation import DataAnnotation
from .power_spectrum_analyzer import PowerSpectrumAnalyzer
from .result_saver import ResultSaver
import numpy as np
class VibrationAnalysisController:
    """振动分析主控制器，协调各个模块工作"""
    
    def __init__(self, annotation_file=None):
        """初始化控制器"""
        self.data_loader = VibrationDataLoader()
        self.annotation = DataAnnotation(annotation_file)
        self.analyzer = PowerSpectrumAnalyzer()
        self.result_saver = ResultSaver()
        
        # 当前分析参数
        self.current_params = {
            'method': 'welch',
            'window': 'hann',
            'overlap_ratio': 0.5,
            'fs': 5000,
            'nperseg': 1024,
            'start_time': None,
            'end_time': None,
            'num_segments': None,
            'segment_duration': None
        }
    
    def set_analysis_params(self, method='welch', window='hann', overlap_ratio=0.5, 
                           fs=2000, nperseg=1024, start_time=None, end_time=None, num_segments=None,
                           segment_duration=None,number_psd=None   ):
        self.current_params = {
            'method': method,
            'window': window,
            'overlap_ratio': overlap_ratio,
            'fs': fs,
            'nperseg': nperseg,
            'start_time': start_time,
            'end_time': end_time,
            'num_segments': num_segments,
            'segment_duration': segment_duration,
            'number_psd': number_psd
        }
    
    def get_file_structure(self, root_dir):
        """获取数据文件结构"""
        return self.data_loader.get_file_structure(root_dir)
    
    def analyze_file(self, file_path, product_code, serial_number):
        """
        分析单个文件
        
        参数:
            file_path: 文件路径
            product_code: 产品代号
            serial_number: 产品序号
            
        返回:
            分析结果和通道信息
        """
        # 解析通道信息
        file_name = os.path.basename(file_path)
        channel, direction = self.data_loader.parse_channel_info(file_name)
        
        # 读取数据
        time_data, signal_data = self.data_loader.load_data(file_path)
        
        # 分析功率谱
        results = self.analyzer.analyze_multiple_segments(
            time_data, signal_data,
            fs=self.current_params['fs'],
            method=self.current_params['method'],
            window=self.current_params['window'],
            nperseg=self.current_params['nperseg'],
            start_time=self.current_params['start_time'],
            end_time=self.current_params['end_time'],
            num_segments=self.current_params['num_segments'],
            segment_duration=self.current_params['segment_duration'],
            overlap_ratio=self.current_params['overlap_ratio'],
            number_psd=self.current_params['number_psd']
            
        )
        
        return {
            'channel': channel,
            'direction': direction,
            'results': results,
            'product_code': product_code,
            'serial_number': serial_number,
            'file_path': file_path
        }
    
    def analyze_multiple_files(self, file_paths, product_code, serial_number):
        """分析多个文件"""
        all_results = []
        for file_path in file_paths:
            try:
                result = self.analyze_file(file_path, product_code, serial_number)
                all_results.append(result)
            except Exception as e:
                print(f"分析文件 {file_path} 失败: {str(e)}")
        return all_results
    
    def save_analysis_results(self, analysis_result, save_dir):
        """保存分析结果"""
        # 创建保存目录
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 构建基础文件名
        base_filename = f"{analysis_result['product_code']}_{analysis_result['serial_number']}"
        if analysis_result['channel']:
            base_filename += f"_{analysis_result['channel']}"
        if analysis_result['direction']:
            base_filename += f"_{analysis_result['direction']}"
        
        full_base_path = str(save_path / base_filename)
        
        # 保存结果
        saved_files = self.result_saver.save_multiple_results(
            analysis_result['results'],
            product_code=analysis_result['product_code'],
            serial_number=analysis_result['serial_number'],
            channel=analysis_result['channel'],
            direction=analysis_result['direction'],
            base_path=full_base_path,
            format='mat',  # 默认保存为mat格式
            additional_info={
                'analysis_method': self.current_params['method'],
                'window_function': self.current_params['window'],
                'overlap_ratio': self.current_params['overlap_ratio'],
                'source_file': analysis_result['file_path']
            }
        )
        
        return saved_files
    def generate_feature_dataset(self, analysis_result, save_path=None):
        """
        生成结构化特征数据集
        
        参数:
            analysis_result: 分析结果
            save_path: 保存路径，None则不保存
            
        返回:
            结构化数据集 (段数 x 160的二维数组)
        """
        # 提取所有功率谱数据
        number_psd = self.current_params.get('number_psd', 160)
        features = []
        for result in analysis_result['results']:
            # 每个功率谱应该已经是160个点
            if len(result['power_spectrum']) == number_psd:
                features.append(result['power_spectrum'])
            else:
                # 如果不是160个点，进行插值
                freq = result['frequency']
                psd = result['power_spectrum']
                target_freq = np.linspace(freq.min(), freq.max(), 160)
                target_psd = np.interp(target_freq, freq, psd)
                features.append(target_psd)
        
        # 转换为numpy数组
        feature_array = np.array(features)
        
        # 如果指定了保存路径，则保存
        if save_path:
            np.savetxt(save_path, feature_array, delimiter=',')
            
        return feature_array
    # 备注相关方法
    def add_annotation(self, annotation_type, **kwargs):
        """添加备注"""
        if annotation_type == 'product':
            return self.annotation.add_product_annotation(
                kwargs.get('product_code'), kwargs.get('note')
            )
        elif annotation_type == 'serial':
            return self.annotation.add_serial_annotation(
                kwargs.get('product_code'), kwargs.get('serial_number'), kwargs.get('note')
            )
        elif annotation_type == 'channel':
            return self.annotation.add_channel_annotation(
                kwargs.get('channel'), kwargs.get('note')
            )
        elif annotation_type == 'direction':
            return self.annotation.add_direction_annotation(
                kwargs.get('direction'), kwargs.get('note')
            )
        return False
    
    def get_annotation(self, annotation_type,** kwargs):
        """获取备注"""
        if annotation_type == 'product':
            return self.annotation.get_product_annotation(kwargs.get('product_code'))
        elif annotation_type == 'serial':
            return self.annotation.get_serial_annotation(
                kwargs.get('product_code'), kwargs.get('serial_number')
            )
        elif annotation_type == 'channel':
            return self.annotation.get_channel_annotation(kwargs.get('channel'))
        elif annotation_type == 'direction':
            return self.annotation.get_direction_annotation(kwargs.get('direction'))
        return ""
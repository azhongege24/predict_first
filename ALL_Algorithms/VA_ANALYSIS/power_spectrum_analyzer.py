import numpy as np
from scipy import signal
from scipy.signal.windows import get_window

class PowerSpectrumAnalyzer:
    """功率谱分析工具类"""
    
    def __init__(self):
        """初始化分析器"""
        # 支持的分析方法
        self.supported_methods = {
            'welch': self._welch_method,
            'fft': self._fft_method
        }
        
        # 支持的窗口函数
        self.supported_windows = [
            'hann', 'hamming', 'blackman', 'bartlett', 
            'boxcar', 'triang', 'parzen', 'bohman',
            'blackmanharris', 'nuttall', 'barthann'
        ]
        
        # 默认参数
        self.default_method = 'welch'
        self.default_window = 'hann'
        self.default_overlap_ratio = 0.5
        self.default_freq_range = (20, 2000)  # Hz
    def resample_power_spectrum(self, freq, psd, target_points=160):
        """将功率谱重采样到固定点数"""
        # 创建目标频率轴
        min_freq = freq.min()
        max_freq = freq.max()
        target_freq = np.linspace(min_freq, max_freq, target_points)
        
        # 插值得到目标点数的功率谱
        target_psd = np.interp(target_freq, freq, psd)
        
        return target_freq, target_psd
    
    def _calculate_sampling_freq(self, time_data):
        """从时间数据计算采样频率"""
        if len(time_data) < 2:
            raise ValueError("时间数据点不足，无法计算采样频率")
            
        dt = np.mean(np.diff(time_data))
        return 1.0 / dt if dt != 0 else 0
    
    def _welch_method(self, signal_data, fs, window, nperseg, noverlap):
        """使用Welch方法计算功率谱"""
        f, Pxx = signal.welch(
            signal_data, 
            fs=fs, 
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=True
        )
        return f, Pxx
    
    def _fft_method(self, signal_data, fs, window, nperseg, noverlap):
        """使用FFT方法计算功率谱"""
        # 应用窗口函数
        window = get_window(window, nperseg)
        signal_windowed = signal_data * window
        
        # 计算FFT
        n = len(signal_windowed)
        fft_result = np.fft.fft(signal_windowed)
        fft_freq = np.fft.fftfreq(n, 1/fs)
        
        # 取单边谱
        positive_freq_mask = fft_freq >= 0
        f = fft_freq[positive_freq_mask]
        Pxx = np.abs(fft_result[positive_freq_mask])**2 / (fs * n)
        
        return f, Pxx
    
    def _get_window_function(self, window_name ,nperseg=None):
        """获取窗口函数"""
        # 首先标准化窗口名称（转换为小写）
        if window_name is None:
            window_name = self.default_window
        else:
            window_name = str(window_name).lower()
            
        # 检查窗口函数是否支持
        if window_name not in self.supported_windows:
            # 提供一个更友好的错误信息，并显示支持的窗口列表
            supported_windows_str = ', '.join(self.supported_windows)
            raise ValueError(f"不支持的窗口函数: '{window_name}'。\n支持的窗口函数有: {supported_windows_str}")
       

        return window_name
    
    def segment_data(self, time_data, signal_data, start_time=None, end_time=None, 
                    num_segments=None, segment_duration=None,nperseg=None, overlap_ratio=0.5):
        """
        将数据分段
        
        参数:
            time_data: 时间数据
            signal_data: 振动信号数据
            start_time: 起始时间
            end_time: 结束时间
            num_segments: 分段数量
            segment_duration: 每段时长(秒)
            overlap_ratio: 重叠比例(0-1)
            
        返回:
            分段后的时间和信号数据列表
        """
        # 应用时间范围筛选
        if start_time is None:
            start_time = time_data[0]
        if end_time is None:
            end_time = time_data[-1]
            
        time_mask = (time_data >= start_time) & (time_data <= end_time)
        filtered_time = time_data[time_mask]
        filtered_signal = signal_data[time_mask]
        
        if len(filtered_time) < 2:
            return []
            
        # 计算采样频率
        fs = self._calculate_sampling_freq(filtered_time)
        
        # 确定分段长度(点数)
        if nperseg is not None:
            # 点数模式：使用nperseg作为每段点数
            segment_length = nperseg
            # 计算实际分段数量
            total_points = len(filtered_time)
            step = int(segment_length * (1 - overlap_ratio))
            num_segments = max(1, (total_points - segment_length) // step + 1)
        elif segment_duration is not None:
            # 时间模式：使用segment_duration计算每段点数
            segment_length = int(segment_duration * fs)
            num_segments = int(np.ceil((end_time - start_time) / segment_duration))
        elif num_segments is not None:
            # 分段数量模式：计算每段点数
            segment_length = int(len(filtered_time) / num_segments)
        else:
            # 如果没有指定分段参数，返回整个时间段
            return [(filtered_time, filtered_signal)]
        
        # 确保分段长度合理
        segment_length = max(1024, segment_length)  # 最小1024点
        if segment_length > len(filtered_time):
            return [(filtered_time, filtered_signal)]
        
        # 计算重叠点数
        overlap_points = int(segment_length * overlap_ratio)
        step = segment_length - overlap_points
        
        # 分段
        segments = []
        for i in range(num_segments):
            start_idx = i * step
            end_idx = start_idx + segment_length
            
            if start_idx >= len(filtered_time):
                break
                
            # 处理最后一段可能不足的情况
            if end_idx > len(filtered_time):
                end_idx = len(filtered_time)
                start_idx = max(0, end_idx - segment_length)
                
            segment_time = filtered_time[start_idx:end_idx]
            segment_signal = filtered_signal[start_idx:end_idx]
            segments.append((segment_time, segment_signal))
            
        return segments
    
    def analyze_single_segment(self, time_data, signal_data, fs=None, method='welch', 
                              window='hann', nperseg=None, overlap_ratio=0.5,number_psd=None):
        """
        分析单个数据段的功率谱
        
        参数:
            time_data: 时间数据
            signal_data: 振动信号数据
            fs: 采样频率，None则自动计算
            method: 分析方法
            window: 窗口函数
            nperseg: 每段点数，None则自动计算
            overlap_ratio: 重叠比例
            
        返回:
            频率数组和功率谱数组
        """
        if len(time_data) < 2 or len(signal_data) < 2:
            raise ValueError("数据点不足，无法进行分析")
            
        # 计算采样频率
        if fs is None:
            fs = self._calculate_sampling_freq(time_data)
            
        if fs <= 0:
            raise ValueError("无效的采样频率")
            
        # 确定每段点数
        if nperseg is None:
            nperseg = min(8192, len(signal_data))  # 默认最大8192点
        nperseg = max(128, nperseg)  # 确保有足够的点数
        
        # 计算重叠点数
        noverlap = int(nperseg * overlap_ratio)
        
        # 获取窗口函数
        window_func = self._get_window_function(window,nperseg)
        
        # 检查分析方法
        if method not in self.supported_methods:
            raise ValueError(f"不支持的分析方法: {method}，支持的有: {list(self.supported_methods.keys())}")
            
        # 调用相应的分析方法
        f, Pxx = self.supported_methods[method](
            signal_data, 
            fs=fs, 
            window=window_func,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        # 应用频率范围过滤
        nyquist = fs / 2
        lower_freq = min(self.default_freq_range[0], nyquist)
        upper_freq = min(self.default_freq_range[1], nyquist)
        
        freq_mask = (f >= lower_freq) & (f <= upper_freq)
        f_filtered = f[freq_mask]
        psd_filtered = Pxx[freq_mask]

        # 重采样到160个点
        f_resampled, psd_resampled = self.resample_power_spectrum(f_filtered, psd_filtered, number_psd)
        return f_resampled, psd_resampled
    
    def analyze_multiple_segments(self, time_data, signal_data, fs=None, method='welch', 
                                 window='hann', nperseg=None, start_time=None, 
                                 end_time=None, num_segments=None, segment_duration=None,
                                 overlap_ratio=0.5,number_psd=None):
        """
        分析多个数据段的功率谱
        
        返回:
            分段信息和对应的功率谱结果列表
        """
        # 数据分段
        segments = self.segment_data(
            time_data, signal_data, 
            start_time=start_time, 
            end_time=end_time,
            num_segments=num_segments,
            segment_duration=segment_duration,
            nperseg=nperseg,  # 添加这行，传递nperseg参数
            overlap_ratio=overlap_ratio
        )
        
        # 分析每个分段
        results = []
        for seg_time, seg_signal in segments:
            if len(seg_time) < 2:
                continue
                
            # 计算该段的时间范围
            seg_start = seg_time[0]
            seg_end = seg_time[-1]
            
            # 分析功率谱
            f, Pxx = self.analyze_single_segment(
                seg_time, seg_signal, 
                fs=fs, 
                method=method,
                window=window,
                nperseg=nperseg,
                overlap_ratio=overlap_ratio,
                number_psd=number_psd
            )
            
            results.append({
                'time_range': (seg_start, seg_end),
                'frequency': f,
                'power_spectrum': Pxx
            })
            
        return results
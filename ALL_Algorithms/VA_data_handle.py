import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView,QFileDialog
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

def analyze_VA_psd(self, file_path=None, fs=1000.0, nperseg=1024, scaling='density'):
    """
    振动数据功率谱分析（集成到主窗口显示）
    :param file_path: 输入文件路径
    :param fs: 采样频率(Hz)
    :param nperseg: 分段长度
    :param scaling: 'density'或'spectrum'
    """
    try:
        if not file_path:
            file_path = self.VA_inpath
            
        # 1. 数据加载
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file_path)
        else:
            raise ValueError("仅支持CSV/Excel文件")

        # 2. 检查数据列
        if 'Time (s)' not in data.columns:
            raise ValueError("缺少Time (s)列")
            
        vibration_cols = [col for col in data.columns 
                         if col.startswith('Group')]
        if not vibration_cols:
            raise ValueError("未找到Group开头的振动数据列")

        # 3. 创建Matplotlib图形（集成到UI）
        self.figure = plt.figure(figsize=(7, 4), dpi=120)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.navi = NavigationToolbar(self.canvas, self.graphicsView)
        ax = self.figure.add_subplot(111)

        # 4. 计算并绘制PSD
        psd_results = {}
        colors = plt.cm.viridis(np.linspace(0, 1, len(vibration_cols)))
        
        for idx, col in enumerate(vibration_cols):
            # Welch方法计算PSD
            f, Pxx = signal.welch(
                data[col],
                fs=fs,
                window='hann',
                nperseg=nperseg,
                scaling=scaling
            )
            
            # 保存计算结果
            psd_results[col] = {
                'frequency': f,
                'psd': Pxx,
                'rms': np.sqrt(np.trapz(Pxx, f))
            }
            
            # 在UI上绘制曲线
            ax.semilogy(
                f, Pxx,
                color=colors[idx],
                alpha=0.7,
                linewidth=1.5,
                label=f'{col.split(" ")[0]} (RMS={psd_results[col]["rms"]:.3f}g)'
            )

        # 5. 图形美化
        ax.set_title(f'功率谱密度分析 ({scaling})')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('PSD [g²/Hz]' if scaling == 'density' else 'Power [g²]')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()
        
        # 6. 显示到UI
        self.canvas.draw()
        self.graphicscene = QGraphicsScene()
        self.graphicscene.addWidget(self.canvas)
        self.graphicsView.setScene(self.graphicscene)
        self.graphicsView.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphicsView.show()

        # 7. 返回计算结果（可选）
        return {
            'psd_data': psd_results,
            'parameters': {
                'fs': fs,
                'nperseg': nperseg,
                'scaling': scaling
            }
        }

    except Exception as e:
        print(f"PSD分析错误: {str(e)}")
        return None
def save_psd_result_util(parent, psd_result):
    """
    通用PSD保存工具函数
    :param parent: 主窗口self（用于弹窗和控件）
    :param psd_result: 分析结果字典
    """
    if not psd_result:
        parent.lineEdit_state.setText("请先分析后再保存！")
        return

    file_save, _ = QFileDialog.getSaveFileName(
        parent, "保存PSD结果", "",
        "CSV Files (*.csv);;Excel Files (*.xlsx)"
    )
    if file_save:
        try:
            import pandas as pd
            all_rows = []
            for group, result in psd_result.items():
                freq = result['frequency']
                psd = result['psd']
                rms = result['rms']
                for f, p in zip(freq, psd):
                    all_rows.append({
                        'Group': group,
                        'Frequency': f,
                        'PSD': p,
                        'RMS': rms
                    })
            df = pd.DataFrame(all_rows)
            if file_save.endswith('.xlsx'):
                df.to_excel(file_save, index=False)
            else:
                df.to_csv(file_save, index=False)
            parent.lineEdit_state.setText("PSD结果保存成功")
        except Exception as e:
            parent.lineEdit_state.setText(f"保存失败: {str(e)}")
     
def preview_VAdata(self, file_path=None):
    """
    预览从外部文件导入的结构振动数据（单组或多组），并在 UI 的 QGraphicsView 上显示
    :param file_path: 数据文件路径
    """
    try:
        if not file_path:
            file_path = self.VA_inpath
            
        # 读取文件
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file_path)
        else:
            raise ValueError("仅支持 CSV 或 Excel 文件格式")

        # 创建 Matplotlib 图形
        self.figure = plt.figure(figsize=(7, 4), dpi=120)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.navi = NavigationToolbar(self.canvas, self.graphicsView)
        ax = self.figure.add_subplot(111)

        # 检查数据列并绘制
        if 'Time (s)' not in data.columns:
            raise ValueError("文件格式不正确，缺少 Time (s) 列")
            
        # 获取所有振动数据列（排除时间列）
        vibration_columns = [col for col in data.columns if col != 'Time (s)' and col.startswith('Group')]
        
        if not vibration_columns:
            raise ValueError("未找到有效振动数据列（应以Group开头）")

        # 绘制各振动组数据
        cmap = plt.get_cmap('tab10')#10种常用颜色
        for i,group_col in enumerate(vibration_columns) :
            ax.plot(data['Time (s)'][:100], 
                    data[group_col][:100], 
                    color=cmap(i%10), #循环颜色 
                    alpha=0.5,
                    label=f'{group_col.split(" ")[0]}')  # 提取组名忽略单位

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (g)')
        ax.set_title('预览前一百个数据点')
        ax.legend()
        ax.grid(True)

        # 绘制图形
        self.canvas.draw()

        # 更新图形到界面
        self.graphicscene = QGraphicsScene()
        self.graphicscene.addWidget(self.canvas)
        self.graphicsView.setScene(self.graphicscene)
        self.graphicsView.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphicsView.show()

    except Exception as e:
        # 这里添加您的异常处理逻辑（例如显示错误弹窗）
        print(f"可视化错误: {str(e)}")

if __name__ == "__main__":
    # 假设有一个测试数据文件 test.csv，内容包含 Time (s), Group1, Group2 ...
    test_file = "va_data_test/test.csv"  # 你可以换成自己的数据文件路径

    # 1. 直接用pandas读取数据
    data = pd.read_csv(test_file)
    # 2. 选出振动数据列
    vibration_cols = [col for col in data.columns if col.startswith('Group')]
    # 3. 计算PSD
    from scipy import signal
    fs = 1000.0
    nperseg = 1024
    scaling = 'density'
    psd_results = {}
    for col in vibration_cols:
        f, Pxx = signal.welch(
            data[col],
            fs=fs,
            window='hann',
            nperseg=nperseg,
            scaling=scaling
        )
        psd_results[col] = {
            'frequency': f,
            'psd': Pxx,
            'rms': np.sqrt(np.trapz(Pxx, f))
        }
    # 4. 保存结果到文件
    # 组装DataFrame
    all_rows = []
    for group, result in psd_results.items():
        freq = result['frequency']
        psd = result['psd']
        rms = result['rms']
        for f_val, p_val in zip(freq, psd):
            all_rows.append({
                'Group': group,
                'Frequency': f_val,
                'PSD': p_val,
                'RMS': rms
            })
    df = pd.DataFrame(all_rows)
    df.to_csv("va_data_test/psd_result_test.csv", index=False)
    print("PSD分析和保存完成，结果已写入 psd_result_test.csv")

    

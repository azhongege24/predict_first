import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView
from matplotlib import pyplot as plt

    
def preview_VAdata(self, file_path=None):
    """
    预览从外部文件导入的振动数据（单组或多组），并在 UI 的 QGraphicsView 上显示
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
        self.canvas.draw()

        # 更新图形到界面
        self.graphicscene = QGraphicsScene()
        self.graphicscene.addWidget(self.canvas)
        self.graphicsView.setScene(self.graphicscene)
        self.graphicsView.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphicsView.show()

    except Exception as e:
        print(f"数据预览失败: {str(e)}")

    # def preview_VA_data(self, file_path=None):
    #     """
    #     预览从外部文件导入的振动数据（单组或多组），并在 UI 的 QGraphicsView 上显示
    #     :param file_path: 数据文件路径
    #     """
    #     try:
    #         if not file_path:
    #             file_path = self.VA_inpath
    #         # 读取文件
    #         if file_path.endswith('.csv'):
    #             data = pd.read_csv(file_path)
    #         elif file_path.endswith(('.xls', '.xlsx')):
    #             data = pd.read_excel(file_path)
    #         else:
    #             raise ValueError("仅支持 CSV 或 Excel 文件格式")

    #         # 创建 Matplotlib 图形
    #         figure = Figure(figsize=(8, 4))
    #         canvas = FigureCanvas(figure)
    #         ax = figure.add_subplot(111)

    #         # 判断数据结构
    #         if 'group' in data.columns:
    #             # 多组数据处理
    #             groups = data['group'].unique()
    #             for group in groups:
    #                 group_data = data[data['group'] == group]
    #                 ax.plot(group_data['frequency'], group_data['psd'], label=f'{group}')
    #             ax.set_xlabel('Frequency (Hz)')
    #             ax.set_ylabel('PSD')
    #             ax.set_title('Group Data')
    #             ax.legend()
    #         elif 'frequency' in data.columns and 'psd' in data.columns:
    #             # 单组数据处理
    #             ax.plot(data['frequency'], data['psd'], label='Single Group')
    #             ax.set_xlabel('Frequency (Hz)')
    #             ax.set_ylabel('PSD')
    #             ax.set_title('Single Group Data')
    #             ax.legend()
    #         else:
    #             raise ValueError("文件格式不正确，缺少必要的列（frequency, psd 或 group）")

    #         # 绘制图形
    #         canvas.draw()

    #         # 更新图形到界面
    #         self.graphicscene = QGraphicsScene()
    #         self.graphicscene.addWidget(canvas)
    #         self.graphicsView.setScene(self.graphicscene)
    #         self.graphicsView.setDragMode(QGraphicsView.ScrollHandDrag)
    #         self.graphicsView.show()

    #     except Exception as e:
    #         print(f"数据预览失败: {str(e)}")
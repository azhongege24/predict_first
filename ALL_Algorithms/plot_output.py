"""为算法图表选择保存目录。"""

import os


def plot_output_dir(default_path):
    """有独立输出根目录时将图表放在其中，否则沿用原目录。"""
    # 未设置独立目录时，保留已有图表的保存位置。
    root = os.environ.get("VIBRATION_PLOT_OUTPUT_ROOT")
    if not root:
        return default_path

    # 使用绝对路径，避免工作目录变化后把图片写到意外位置。
    if not os.path.isabs(root):
        raise ValueError("图表输出根目录必须是绝对路径")

    # 每种图表保留原有子目录名，清理旧图时只触及该子目录。
    return os.path.join(os.path.abspath(root), default_path)

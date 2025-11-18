import joblib
from PyQt5.QtWidgets import QFileDialog


def select_pretrained_model_path(parent):
    """
    弹出文件选择框，选择预训练模型文件，仅返回路径，不加载模型。
    :param parent: 主窗口self（用于弹窗）
    :return: 选择的模型文件路径或None
    """
    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        "选择预训练模型文件",
        "./trained_models",  # 默认打开的文件夹
        "模型文件 (*.pkl *.joblib);;所有文件 (*)"
    )
    if file_path:
        parent.selected_model_path = file_path
        if hasattr(parent, 'lineEdit_pretrained_model_path'):
            parent.lineEdit_pretrained_model_path.setText(file_path)
        return file_path
    else:
        parent.lineEdit_pretrained_model_path.setText("未选择模型文件")
        return None
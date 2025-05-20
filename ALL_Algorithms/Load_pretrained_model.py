import joblib
from PyQt5.QtWidgets import QFileDialog

'''
def load_pretrained_model_util(parent):
    """
    弹出文件选择框，加载预训练模型（如.pkl文件）
    :param parent: 主窗口self（用于弹窗）
    :return: 加载的模型对象或None
    """
    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        "选择预训练模型文件",
        "./trained_models",  # 默认打开的文件夹，可根据实际情况修改
        "模型文件 (*.pkl *.joblib);;所有文件 (*)"
    )
    if file_path:
        try:
            model = joblib.load(file_path)
            parent.lineEdit_pretrained_model_path.setText("模型加载成功！")
            parent.selected_model_path = file_path  # 新增：保存路径
            if hasattr(parent, 'lineEdit_pretrained_model_path'):
                parent.lineEdit_pretrained_model_path.setText(file_path)
            return model
        except Exception as e:
            import traceback
            print(traceback.format_exc())  # 打印详细错误
            parent.lineEdit_pretrained_model_path.setText(f"模型加载失败: {str(e)}")
            return None
    else:
        parent.lineEdit_pretrained_model_path.setText("未选择模型文件")
        return None
'''


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
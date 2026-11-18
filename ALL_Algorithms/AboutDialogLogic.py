"""显示软件基本信息的独立窗口。"""

from PyQt5.QtWidgets import QDialog

from ALL_Algorithms.AboutDialog import Ui_AboutDialog


class AboutDialog(QDialog, Ui_AboutDialog):
    """展示界面设计文件中的信息行头。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # 创建信息标签和关闭按钮，并让按钮关闭当前窗口。
        self.setupUi(self)
        self.close_button.clicked.connect(self.accept)

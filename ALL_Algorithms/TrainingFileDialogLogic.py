"""选择、排序并合并训练数据文件的窗口。"""

import os

from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QMessageBox,
)

from ALL_Algorithms.TrainingFileDialog import Ui_TrainingFileDialog
from utils.training_dataset_merge import read_training_file, merge_training_frames


class TrainingFileDialog(QDialog, Ui_TrainingFileDialog):
    """在确认之前暂存文件和数据，不改变主窗口当前训练数据。"""

    def __init__(self, parent=None, initial_dir="data/"):
        super().__init__(parent)
        # 使用 Designer 生成的界面类创建窗口布局和控件。
        self.setupUi(self)
        self.initial_dir = initial_dir
        self.entries = []
        self.merged_data = None
        self.selected_paths = []

        # 把生成界面的按钮连接到文件列表与合并处理逻辑。
        self.add_button.clicked.connect(self.add_files)
        self.remove_button.clicked.connect(self.remove_selected)
        self.move_up_button.clicked.connect(lambda: self.move_selected(-1))
        self.move_down_button.clicked.connect(lambda: self.move_selected(1))
        self.confirm_button.clicked.connect(self.confirm_merge)
        self.cancel_button.clicked.connect(self.reject)

    def add_files(self):
        """读取新文件，并在列表中显示各自的行列数。"""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "添加训练文件", self.initial_dir,
            "数据文件 (*.csv *.xls *.xlsx *.mat)",
        )
        existing = {os.path.normcase(os.path.abspath(path)) for path, _ in self.entries}
        problems = []
        for path in paths:
            identity = os.path.normcase(os.path.abspath(path))
            if identity in existing:
                problems.append(f"{os.path.basename(path)}：已添加")
                continue
            try:
                data = read_training_file(path)
            except Exception as exc:
                problems.append(f"{os.path.basename(path)}：{exc}")
                continue
            self.entries.append((path, data))
            existing.add(identity)
            self.initial_dir = os.path.dirname(path)
        self.refresh_list()
        if problems:
            QMessageBox.warning(self, "部分文件未添加", "\n".join(problems))

    def remove_selected(self):
        """移除选中的文件，保留其他文件原有顺序。"""
        row = self.file_list.currentRow()
        if row >= 0:
            self.entries.pop(row)
            self.refresh_list()
            if self.entries:
                self.file_list.setCurrentRow(min(row, len(self.entries) - 1))

    def move_selected(self, offset):
        """交换相邻文件的位置，使用户能够确定拼接行序。"""
        row = self.file_list.currentRow()
        target = row + offset
        if 0 <= row < len(self.entries) and 0 <= target < len(self.entries):
            self.entries[row], self.entries[target] = self.entries[target], self.entries[row]
            self.refresh_list()
            self.file_list.setCurrentRow(target)

    def refresh_list(self):
        """把文件顺序、行数和字段数更新到窗口。"""
        self.file_list.clear()
        for index, (path, data) in enumerate(self.entries, 1):
            self.file_list.addItem(
                f"{index}. {os.path.basename(path)}  —  {len(data)} 行，{len(data.columns)} 列"
            )
            self.file_list.item(index - 1).setToolTip(path)
        self.summary.setText(
            f"已添加 {len(self.entries)} 个文件，共 {sum(len(data) for _, data in self.entries)} 行"
            if self.entries else "尚未添加文件"
        )

    def confirm_merge(self):
        """合并成功后才提交给调用窗口。"""
        try:
            merged, mixed = merge_training_frames([data for _, data in self.entries])
        except Exception as exc:
            QMessageBox.warning(self, "无法拼接", str(exc))
            return
        if mixed:
            QMessageBox.warning(
                self, "字段类型不一致",
                "以下同名字段在不同文件中包含数值和文字，已保留原值；"
                "合并后会按字符串列识别：" + "、".join(mixed),
            )
        self.merged_data = merged
        self.selected_paths = [path for path, _ in self.entries]
        self.accept()

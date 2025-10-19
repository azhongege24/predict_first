import json
import os
from datetime import datetime

class DataAnnotation:
    """数据备注管理类"""
    
    def __init__(self, annotation_file=None):
        """初始化，可指定备注文件路径"""
        self.annotations = {
            "product_codes": {},    # 产品代号备注
            "serial_numbers": {},   # 产品序号备注
            "channels": {},         # 通道备注
            "directions": {}        # 方向备注
        }
        self.annotation_file = annotation_file or "vibration_annotations.json"
        self._load_annotations()
    
    def _load_annotations(self):
        """从文件加载备注"""
        if os.path.exists(self.annotation_file):
            try:
                with open(self.annotation_file, 'r', encoding='utf-8') as f:
                    self.annotations = json.load(f)
            except Exception as e:
                print(f"加载备注文件失败: {str(e)}")
    
    def save_annotations(self):
        """保存备注到文件"""
        try:
            with open(self.annotation_file, 'w', encoding='utf-8') as f:
                json.dump(self.annotations, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存备注文件失败: {str(e)}")
            return False
    
    def add_product_annotation(self, product_code, note):
        """添加产品代号备注"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.annotations["product_codes"][product_code] = {
            "note": note,
            "timestamp": timestamp
        }
        return self.save_annotations()
    
    def add_serial_annotation(self, product_code, serial_number, note):
        """添加产品序号备注"""
        key = f"{product_code}::{serial_number}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.annotations["serial_numbers"][key] = {
            "note": note,
            "timestamp": timestamp
        }
        return self.save_annotations()
    
    def add_channel_annotation(self, channel, note):
        """添加通道备注"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.annotations["channels"][channel] = {
            "note": note,
            "timestamp": timestamp
        }
        return self.save_annotations()
    
    def add_direction_annotation(self, direction, note):
        """添加方向备注"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.annotations["directions"][direction] = {
            "note": note,
            "timestamp": timestamp
        }
        return self.save_annotations()
    
    def get_product_annotation(self, product_code):
        """获取产品代号备注"""
        return self.annotations["product_codes"].get(product_code, {}).get("note", "")
    
    def get_serial_annotation(self, product_code, serial_number):
        """获取产品序号备注"""
        key = f"{product_code}::{serial_number}"
        return self.annotations["serial_numbers"].get(key, {}).get("note", "")
    
    def get_channel_annotation(self, channel):
        """获取通道备注"""
        return self.annotations["channels"].get(channel, {}).get("note", "")
    
    def get_direction_annotation(self, direction):
        """获取方向备注"""
        return self.annotations["directions"].get(direction, {}).get("note", "")
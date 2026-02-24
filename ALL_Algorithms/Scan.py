# -*- coding: utf-8 -*-
"""
特征列扫描功能模块
用于检查模型特征列与预测数据特征列的匹配情况
"""

def scan_model_features(model_input_columns, model_output_columns, predict_input_columns, predict_output_columns):
    """
    扫描模型特征列与预测数据特征列的匹配情况
    
    Args:
        model_input_columns: 模型输入特征列列表
        model_output_columns: 模型输出特征列列表
        predict_input_columns: 预测数据输入特征列列表
        predict_output_columns: 预测数据输出特征列列表
    
    Returns:
        dict: 包含扫描结果的字典
    """
    result = {
        'model_features_info': "",
        'predict_data_match': "否",
        'state_info': "未开始扫描"
    }
    
    try:
        # 构建模型特征信息字符串
        if model_input_columns and model_output_columns:
            features_info = "模型特征信息:\n"
            features_info += f"输入特征列 ({len(model_input_columns)}个): {', '.join(model_input_columns)}\n"
            features_info += f"输出特征列 ({len(model_output_columns)}个): {', '.join(model_output_columns)}"
            result['model_features_info'] = features_info
        else:
            result['model_features_info'] = "模型未包含特征列信息"
        
        # 检查预测数据是否包含模型特征列
        if model_input_columns and predict_input_columns:
            # 检查输入特征列匹配
            missing_inputs = [col for col in model_input_columns if col not in predict_input_columns]
            if not missing_inputs:
                result['predict_data_match'] = "是"
                result['state_info'] = "预测数据包含所有模型特征列"
            else:
                result['predict_data_match'] = "否"
                result['state_info'] = f"预测数据缺少输入特征列: {', '.join(missing_inputs)}"
        else:
            result['state_info'] = "模型或预测数据特征列信息不完整"
            
    except Exception as e:
        result['state_info'] = f"扫描过程中出错: {str(e)}"
    
    return result

def check_feature_compatibility(model_columns, predict_columns):
    """
    检查特征列的兼容性
    
    Args:
        model_columns: 模型特征列列表
        predict_columns: 预测数据特征列列表
    
    Returns:
        tuple: (是否兼容, 缺失的特征列列表)
    """
    if not model_columns:
        return True, []
    
    missing_columns = [col for col in model_columns if col not in predict_columns]
    return len(missing_columns) == 0, missing_columns
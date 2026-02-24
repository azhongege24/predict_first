import numpy as np
import random
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer
from PyQt5.QtWidgets import QProgressDialog, QMessageBox, QApplication
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from PyQt5.QtCore import Qt
from sklearn.svm import SVR
from mutar import GroupLasso,MTW,ReMTW
from ALL_Algorithms.Gassu_process import MultitaskGPRegressor
from ALL_Algorithms.Algorithms import MMoERegressor
import torch
def random_search_random_forest(parent_window, data_train, input_columns, output_columns, 
                               n_iter=20, cv_folds=5):
    """
    随机森林随机搜索函数
    """
    try:
        # 参数搜索空间
        param_distributions = {
            'n_estimators': list(range(50, 501, 50)),  # 50-500，步长50
            'max_depth': list(range(5, 31)) + [None],  # 包含5-30所有整数和无限制
            'random_state': list(range(1, 101)),       # 包含1-100
            'scale_features': [True, False]
        }
        
        # 准备数据
        X = data_train[input_columns].values
        y = data_train[output_columns].values
        
        # 如果多输出，取第一个输出进行评估
        if y.ndim > 1 and y.shape[1] > 1:
            y = y[:, 0]  # 取第一个输出进行参数调优
        
        best_score = float('inf')  # 最小化MSE
        best_params = None
        
        # 进度对话框
        progress_dialog = QProgressDialog("正在进行随机森林参数寻优...", "取消", 0, n_iter, parent_window)
        progress_dialog.setWindowTitle("随机搜索进行中")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)  # 立即显示
        progress_dialog.show()
        
        # 执行随机搜索
        for i in range(n_iter):
            # 检查是否取消了操作
            if progress_dialog.wasCanceled():
                break
            
            # 更新进度
            progress_dialog.setValue(i)
            progress_dialog.setLabelText(f"正在进行随机搜索 ({i+1}/{n_iter})...")
            QApplication.processEvents()  # 处理UI事件
            
            # 随机选择参数
            n_estimators = random.choice(param_distributions['n_estimators'])
            max_depth = random.choice(param_distributions['max_depth'])
            random_state = random.choice(param_distributions['random_state'])
            scale_features = random.choice(param_distributions['scale_features'])
            
            try:
                # 数据预处理
                X_processed = X.copy()
                if scale_features:
                    scaler = StandardScaler()
                    X_processed = scaler.fit_transform(X_processed)
                
                # 创建模型
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1  # 使用所有核心
                )
                
                # 交叉验证评估
                mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
                scores = cross_val_score(model, X_processed, y, 
                                       cv=cv_folds, scoring=mse_scorer)
                avg_score = -np.mean(scores)  # 转换为正数
                
                # 更新最佳参数
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'random_state': random_state,
                        'scale_features': scale_features,
                        'score': avg_score
                    }
                    
            except Exception as e:
                print(f"参数组合评估失败: n_estimators={n_estimators}, max_depth={max_depth}, random_state={random_state}, error={e}")
                continue
        
        # 关闭进度对话框
        progress_dialog.close()
        
        if best_params is None:
            QMessageBox.warning(parent_window, "搜索失败", "所有参数组合评估都失败了！")
            return None
        
        return best_params
        
    except Exception as e:
        QMessageBox.warning(parent_window, "随机搜索失败", f"随机搜索过程中出现错误：{str(e)}")
        return None



from sklearn.tree import DecisionTreeRegressor

def random_search_decision_tree(parent_window, data_train, input_columns, output_columns, 
                               n_iter=20, cv_folds=5):
    """
    决策树随机搜索函数
    """
    try:
        # 参数搜索空间
        param_distributions = {
            'max_depth': list(range(1, 31)) + [None],  # 包含1-30所有整数和无限制
            'random_state': list(range(1, 101)),       # 包含1-100
            'scale_features': [True, False]
        }
        
        # 准备数据
        X = data_train[input_columns].values
        y = data_train[output_columns].values
        
        # 如果多输出，取第一个输出进行评估
        if y.ndim > 1 and y.shape[1] > 1:
            y = y[:, 0]  # 取第一个输出进行参数调优
        
        best_score = float('inf')  # 最小化MSE
        best_params = None
        
        # 进度对话框
        progress_dialog = QProgressDialog("正在进行决策树参数寻优...", "取消", 0, n_iter, parent_window)
        progress_dialog.setWindowTitle("随机搜索进行中")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)  # 立即显示
        progress_dialog.show()
        
        # 执行随机搜索
        for i in range(n_iter):
            # 检查是否取消了操作
            if progress_dialog.wasCanceled():
                break
            
            # 更新进度
            progress_dialog.setValue(i)
            progress_dialog.setLabelText(f"正在进行随机搜索 ({i+1}/{n_iter})...")
            QApplication.processEvents()  # 处理UI事件
            
            # 随机选择参数
            max_depth = random.choice(param_distributions['max_depth'])
            random_state = random.choice(param_distributions['random_state'])
            scale_features = random.choice(param_distributions['scale_features'])
            
            try:
                # 数据预处理
                X_processed = X.copy()
                if scale_features:
                    scaler = StandardScaler()
                    X_processed = scaler.fit_transform(X_processed)
                
                # 创建模型
                model = DecisionTreeRegressor(
                    max_depth=max_depth,
                    random_state=random_state
                )
                
                # 交叉验证评估
                mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
                scores = cross_val_score(model, X_processed, y, 
                                       cv=cv_folds, scoring=mse_scorer)
                avg_score = -np.mean(scores)  # 转换为正数
                
                # 更新最佳参数
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {
                        'max_depth': max_depth,
                        'random_state': random_state,
                        'scale_features': scale_features,
                        'score': avg_score
                    }
                    
            except Exception as e:
                print(f"参数组合评估失败: max_depth={max_depth}, random_state={random_state}, error={e}")
                continue
        
        # 关闭进度对话框
        progress_dialog.close()
        
        if best_params is None:
            QMessageBox.warning(parent_window, "搜索失败", "所有参数组合评估都失败了！")
            return None
        
        return best_params
        
    except Exception as e:
        QMessageBox.warning(parent_window, "随机搜索失败", f"随机搜索过程中出现错误：{str(e)}")
        return None

def random_search_svm(parent_window, data_train, input_columns, output_columns, 
                     n_iter=10, cv_folds=3):
    """
    支持向量机随机搜索函数
    """
    try:
        # 参数搜索空间
        param_distributions = {
            # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # 核函数类型
            # 'C': [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2000],  # 正则化参数
            # 'epsilon': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],  # epsilon参数
            # # 'random_state': list(range(1, 101)),  # 包含1-100
            # 'scale_features': [True, False]
            'kernel': [ 'rbf'],  # 优化：只使用最常见的两种核函数
            'C': [0.1, 1, 10, 100, 1000],  # 优化：减少C参数数量
            'epsilon': [0.01, 0.1, 0.3],  # 优化：减少epsilon参数数量
            'scale_features': [True, False]
        }
        
        # 准备数据
        X = data_train[input_columns].values
        y = data_train[output_columns].values
        
        # 如果多输出，取第一个输出进行评估
        if y.ndim > 1 and y.shape[1] > 1:
            y = y[:, 0]  # 取第一个输出进行参数调优
        
        best_score = float('inf')  # 最小化MSE
        best_params = None
        
        # 进度对话框
        progress_dialog = QProgressDialog("正在进行SVM参数寻优...", "取消", 0, n_iter, parent_window)
        progress_dialog.setWindowTitle("随机搜索进行中")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)  # 立即显示
        progress_dialog.show()
        
        # 执行随机搜索
        for i in range(n_iter):
            # 检查是否取消了操作
            if progress_dialog.wasCanceled():
                break
            
            # 更新进度
            progress_dialog.setValue(i)
            progress_dialog.setLabelText(f"正在进行随机搜索 ({i+1}/{n_iter})...")
            QApplication.processEvents()  # 处理UI事件
            
            # 随机选择参数
            kernel = random.choice(param_distributions['kernel'])
            C = random.choice(param_distributions['C'])
            epsilon = random.choice(param_distributions['epsilon'])
            # random_state = random.choice(param_distributions['random_state'])
            scale_features = random.choice(param_distributions['scale_features'])
            
            try:
                # 数据预处理
                X_processed = X.copy()
                if scale_features:
                    scaler = StandardScaler()
                    X_processed = scaler.fit_transform(X_processed)
                
                # 创建模型
                model = SVR(
                    kernel=kernel,
                    C=C,
                    epsilon=epsilon,
                    # random_state=random_state
                )
                
                # 交叉验证评估
                mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
                scores = cross_val_score(model, X_processed, y, 
                                       cv=cv_folds, scoring=mse_scorer)
                avg_score = -np.mean(scores)  # 转换为正数
                
                # 更新最佳参数
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {
                        'kernel': kernel,
                        'C': C,
                        'epsilon': epsilon,
                        # 'random_state': random_state,
                        'scale_features': scale_features,
                        'score': avg_score
                    }
                    
            except Exception as e:
                print(f"参数组合评估失败: kernel={kernel}, C={C}, epsilon={epsilon},  error={e}")
                continue
        
        # 关闭进度对话框
        progress_dialog.close()
        
        if best_params is None:
            QMessageBox.warning(parent_window, "搜索失败", "所有参数组合评估都失败了！")
            return None
        
        return best_params
        
    except Exception as e:
        QMessageBox.warning(parent_window, "随机搜索失败", f"随机搜索过程中出现错误：{str(e)}")
        return None
    


from sklearn.neural_network import MLPRegressor
 
def random_search_mlp(parent_window, data_train, input_columns, output_columns, 
                     n_iter=15, cv_folds=3):
    """
    多层感知机(MLP)随机搜索函数
    """
    try:
        # 参数搜索空间
        param_distributions = {
            'hidden_layer_sizes': [
                (50,), (100,), (200,), (50, 25), (100, 50), (200, 100), 
                (100, 50, 25), (200, 100, 50), (50, 50), (100, 100)
            ],  # 隐藏层结构
            'max_iter': [200, 500, 1000, 1500],  # 最大迭代次数
            'alpha': [0.0001, 0.001, 0.01, 0.1],  # L2正则化参数
            'random_state': list(range(1, 101)),  # 随机种子
            'scale_features': [True, False]
        }
        
        # 准备数据
        X = data_train[input_columns].values
        y = data_train[output_columns].values
        
        # 如果多输出，取第一个输出进行评估
        if y.ndim > 1 and y.shape[1] > 1:
            y = y[:, 0]  # 取第一个输出进行参数调优
        
        best_score = float('inf')  # 最小化MSE
        best_params = None
        
        # 进度对话框
        progress_dialog = QProgressDialog("正在进行MLP参数寻优...", "取消", 0, n_iter, parent_window)
        progress_dialog.setWindowTitle("随机搜索进行中")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)  # 立即显示
        progress_dialog.show()
        
        # 执行随机搜索
        for i in range(n_iter):
            # 检查是否取消了操作
            if progress_dialog.wasCanceled():
                break
            
            # 更新进度
            progress_dialog.setValue(i)
            progress_dialog.setLabelText(f"正在进行随机搜索 ({i+1}/{n_iter})...")
            QApplication.processEvents()  # 处理UI事件
            
            # 随机选择参数
            hidden_layer_sizes = random.choice(param_distributions['hidden_layer_sizes'])
            max_iter = random.choice(param_distributions['max_iter'])
            alpha = random.choice(param_distributions['alpha'])
            random_state = random.choice(param_distributions['random_state'])
            scale_features = random.choice(param_distributions['scale_features'])
            
            try:
                # 数据预处理
                X_processed = X.copy()
                if scale_features:
                    scaler = StandardScaler()
                    X_processed = scaler.fit_transform(X_processed)
                
                # 创建模型
                model = MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    max_iter=max_iter,
                    alpha=alpha,
                    random_state=random_state,
                    learning_rate_init=5e-4,
                    verbose=False,
                    early_stopping=True  # 添加早停防止过拟合
                )
                
                # 交叉验证评估
                mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
                scores = cross_val_score(model, X_processed, y, 
                                       cv=cv_folds, scoring=mse_scorer)
                avg_score = -np.mean(scores)  # 转换为正数
                
                # 更新最佳参数
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {
                        'hidden_layer_sizes': hidden_layer_sizes,
                        'max_iter': max_iter,
                        'alpha': alpha,
                        'random_state': random_state,
                        'scale_features': scale_features,
                        'score': avg_score
                    }
                    
            except Exception as e:
                print(f"参数组合评估失败: hidden_layer_sizes={hidden_layer_sizes}, max_iter={max_iter}, alpha={alpha}, error={e}")
                continue
        
        # 关闭进度对话框
        progress_dialog.close()
        
        if best_params is None:
            QMessageBox.warning(parent_window, "搜索失败", "所有参数组合评估都失败了！")
            return None
        
        return best_params
        
    except Exception as e:
        QMessageBox.warning(parent_window, "随机搜索失败", f"随机搜索过程中出现错误：{str(e)}")
        return None
        
def random_search_extra_trees(parent_window, data_train, input_columns, output_columns, 
                             n_iter=20, cv_folds=5):
    """
    极端树（Extra Trees）随机搜索函数
    """
    try:
        # 参数搜索空间
        param_distributions = {
            'n_estimators': list(range(50, 501, 50)),  # 50-500，步长50
            'max_depth': list(range(5, 31)) + [None],  # 包含5-30所有整数和无限制
            'random_state': list(range(1, 101)),       # 包含1-100
            'scale_features': [True, False]
        }
        
        # 准备数据
        X = data_train[input_columns].values
        y = data_train[output_columns].values
        
        # 如果多输出，取第一个输出进行评估
        if y.ndim > 1 and y.shape[1] > 1:
            y = y[:, 0]  # 取第一个输出进行参数调优
        
        best_score = float('inf')  # 最小化MSE
        best_params = None
        
        # 进度对话框
        progress_dialog = QProgressDialog("正在进行极端树参数寻优...", "取消", 0, n_iter, parent_window)
        progress_dialog.setWindowTitle("随机搜索进行中")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)  # 立即显示
        progress_dialog.show()
        
        # 执行随机搜索
        for i in range(n_iter):
            # 检查是否取消了操作
            if progress_dialog.wasCanceled():
                break
            
            # 更新进度
            progress_dialog.setValue(i)
            progress_dialog.setLabelText(f"正在进行随机搜索 ({i+1}/{n_iter})...")
            QApplication.processEvents()  # 处理UI事件
            
            # 随机选择参数
            n_estimators = random.choice(param_distributions['n_estimators'])
            max_depth = random.choice(param_distributions['max_depth'])
            random_state = random.choice(param_distributions['random_state'])
            scale_features = random.choice(param_distributions['scale_features'])
            
            try:
                # 数据预处理
                X_processed = X.copy()
                if scale_features:
                    scaler = StandardScaler()
                    X_processed = scaler.fit_transform(X_processed)
                
                # 创建模型
                model = ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1  # 使用所有核心
                )
                
                # 交叉验证评估
                mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
                scores = cross_val_score(model, X_processed, y, 
                                       cv=cv_folds, scoring=mse_scorer)
                avg_score = -np.mean(scores)  # 转换为正数
                
                # 更新最佳参数
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'random_state': random_state,
                        'scale_features': scale_features,
                        'score': avg_score
                    }
                    
            except Exception as e:
                print(f"参数组合评估失败: n_estimators={n_estimators}, max_depth={max_depth}, random_state={random_state}, error={e}")
                continue
        
        # 关闭进度对话框
        progress_dialog.close()
        
        if best_params is None:
            QMessageBox.warning(parent_window, "搜索失败", "所有参数组合评估都失败了！")
            return None
        
        return best_params
        
    except Exception as e:
        QMessageBox.warning(parent_window, "随机搜索失败", f"随机搜索过程中出现错误：{str(e)}")


def random_search_linear_regression(parent_window, data_train, input_columns, output_columns, 
                                  n_iter=20, cv_folds=5):
    """
    线性回归（Linear Regression）随机搜索函数
    """
    try:
        # 参数搜索空间
        param_distributions = {
            'fit_intercept': [True, False],  # 是否拟合截距
            'scale_features': [True, False]  # 是否进行特征缩放
        }
        
        # 准备数据
        X = data_train[input_columns].values
        y = data_train[output_columns].values
        
        # 如果多输出，取第一个输出进行评估
        if y.ndim > 1 and y.shape[1] > 1:
            y = y[:, 0]  # 取第一个输出进行参数调优
        
        best_score = float('inf')  # 最小化MSE
        best_params = None
        
        # 进度对话框
        progress_dialog = QProgressDialog("正在进行线性回归参数寻优...", "取消", 0, n_iter, parent_window)
        progress_dialog.setWindowTitle("随机搜索进行中")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)  # 立即显示
        progress_dialog.show()
        
        # 执行随机搜索
        for i in range(n_iter):
            # 检查是否取消了操作
            if progress_dialog.wasCanceled():
                break
            
            # 更新进度
            progress_dialog.setValue(i)
            progress_dialog.setLabelText(f"正在进行随机搜索 ({i+1}/{n_iter})...")
            QApplication.processEvents()  # 处理UI事件
            
            # 随机选择参数
            fit_intercept = random.choice(param_distributions['fit_intercept'])
            scale_features = random.choice(param_distributions['scale_features'])
            
            try:
                # 数据预处理
                X_processed = X.copy()
                if scale_features:
                    scaler = StandardScaler()
                    X_processed = scaler.fit_transform(X_processed)
                
                # 创建模型
                model = LinearRegression(
                    fit_intercept=fit_intercept
                )
                
                # 交叉验证评估
                mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
                scores = cross_val_score(model, X_processed, y, 
                                       cv=cv_folds, scoring=mse_scorer)
                avg_score = -np.mean(scores)  # 转换为正数
                
                # 更新最佳参数
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {
                        'fit_intercept': fit_intercept,
                        'scale_features': scale_features,
                        'score': avg_score
                    }
                    
            except Exception as e:
                print(f"参数组合评估失败: fit_intercept={fit_intercept}, scale_features={scale_features}, error={e}")
                continue
        
        # 关闭进度对话框
        progress_dialog.close()
        
        if best_params is None:
            QMessageBox.warning(parent_window, "搜索失败", "所有参数组合评估都失败了！")
            return None
        
        return best_params
        
    except Exception as e:
        QMessageBox.warning(parent_window, "随机搜索失败", f"随机搜索过程中出现错误：{str(e)}")


def random_search_group_lasso(parent_window, data_train, input_columns, output_columns, 
                             n_iter=15, cv_folds=3):
    """
    Group Lasso 随机搜索函数
    """
    try:
        # 参数搜索空间
        param_distributions = {
            'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],  # 正则化参数
            'max_iter': [500, 1000, 2000, 3000],  # 最大迭代次数
            'tol': [1e-3, 1e-4, 1e-5],  # 收敛容差
            'random_state': list(range(1, 101)),  # 随机种子
            'scale_features': [True, False]  # 是否标准化，与Group_Lasso.py保持一致
        }
        
        # 准备数据
        X = data_train[input_columns].values.astype(float)
        y = data_train[output_columns].values.astype(float)
        
        def clean_data(arr):
            """清理数据中的NaN和无穷大值"""
            arr = np.asarray(arr)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
            return arr
        
        X = clean_data(X)
        y = clean_data(y)
        
        # Group Lasso需要处理所有输出，不能只取第一个
        n_tasks = len(output_columns)
        
        best_score = float('inf')  # 最小化MSE
        best_params = None
        
        # 进度对话框
        progress_dialog = QProgressDialog("正在进行Group Lasso参数寻优...", "取消", 0, n_iter, parent_window)
        progress_dialog.setWindowTitle("随机搜索进行中")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)  # 立即显示
        progress_dialog.show()
        
        # 执行随机搜索
        for i in range(n_iter):
            # 检查是否取消了操作
            if progress_dialog.wasCanceled():
                break
            
            # 更新进度
            progress_dialog.setValue(i)
            progress_dialog.setLabelText(f"正在进行随机搜索 ({i+1}/{n_iter})...")
            QApplication.processEvents()  # 处理UI事件
            
            # 随机选择参数
            alpha = random.choice(param_distributions['alpha'])
            max_iter = random.choice(param_distributions['max_iter'])
            tol = random.choice(param_distributions['tol'])
            random_state = random.choice(param_distributions['random_state'])
            scale_features = random.choice(param_distributions['scale_features'])
            
            try:
                # 数据预处理
                X_processed = X.copy()
                if scale_features:
                    # 对特征进行标准化
                    scaler_X = StandardScaler()
                    X_processed = scaler_X.fit_transform(X_processed)
                
                # 转换为 3D 格式 (n_tasks, n_samples, n_features)
                X_processed_3d = np.repeat(X_processed[None, :, :], n_tasks, axis=0)
                y_processed_3d = y.T  # 转置为 (n_tasks, n_samples)
                
                # 创建模型
                model = GroupLasso(
                    alpha=alpha,
                    max_iter=max_iter,
                    tol=tol,
                    fit_intercept=True,
                    normalize=True
                )
                
                # 交叉验证评估
                # 自定义交叉验证评估，因为Group Lasso需要特殊的数据格式
                fold_scores = []
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

                for train_idx, val_idx in kf.split(X_processed):
                    # 分割数据
                    X_train_fold = X_processed[train_idx]
                    y_train_fold = y[train_idx]
                    X_val_fold = X_processed[val_idx]
                    y_val_fold = y[val_idx]
                    
                    # 转换为3D格式
                    X_train_3d = np.repeat(X_train_fold[None, :, :], n_tasks, axis=0)
                    y_train_3d = y_train_fold.T
                    X_val_3d = np.repeat(X_val_fold[None, :, :], n_tasks, axis=0)
                    
                    # 训练模型
                    model.fit(X_train_3d, y_train_3d)
                    
                    # 预测
                    y_pred = model.predict(X_val_3d).T  # 转置回 (n_samples, n_tasks)
                    
                    # 清理预测结果
                    y_pred = clean_data(y_pred)
                    
                    # 计算MSE
                    mse = mean_squared_error(y_val_fold, y_pred)
                    fold_scores.append(mse)
                
                avg_score = np.mean(fold_scores)

                # 更新最佳参数
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {
                        'alpha': alpha,
                        'max_iter': max_iter,
                        'tol': tol,
                        'random_state': random_state,
                        'scale_features': scale_features,
                        'score': avg_score
                    }
                    
            except Exception as e:
                print(f"参数组合评估失败: alpha={alpha}, max_iter={max_iter}, tol={tol}, error={e}")
                continue
        
        # 关闭进度对话框
        progress_dialog.close()
        
        if best_params is None:
            QMessageBox.warning(parent_window, "搜索失败", "所有参数组合评估都失败了！")
            return None
        
        return best_params
        
    except Exception as e:
        QMessageBox.warning(parent_window, "随机搜索失败", f"随机搜索过程中出现错误：{str(e)}")
        return None

def random_search_mtw(parent_window, data_train, input_columns, output_columns, 
                      n_iter=15, cv_folds=3):
    """
    MTW (Multitask Wasserstein) 随机搜索函数
    适配MTW.py中的数据格式要求
    """
    try:
        # 参数搜索空间
        param_distributions = {
            'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],  # Wasserstein距离正则化参数
            'beta': [0.1, 0.3, 0.5, 0.7, 0.8, 0.9],  # L1正则化参数
            'max_iter': [500, 1000, 2000, 3000],  # 最大迭代次数
            'tol': [1e-3, 1e-4, 1e-5],  # 收敛容差
            'random_state': list(range(1, 101)),  # 随机种子
            'gpu': [ False]  # 是否使用GPU加速
        }
        
        # 准备数据
        X = data_train[input_columns].values.astype(float)
        y = data_train[output_columns].values.astype(float)
        
        # 数据清理函数（与MTW.py中一致）
        def clean_data(arr):
            """清理数据中的NaN和无穷大值"""
            arr = np.asarray(arr)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
            return arr
        
        X = clean_data(X)
        y = clean_data(y)
        
        # MTW需要处理所有输出
        n_tasks = len(output_columns)
        
        best_score = float('inf')  # 最小化MSE
        best_params = None
        
        # 进度对话框
        progress_dialog = QProgressDialog("正在进行MTW参数寻优...", "取消", 0, n_iter, parent_window)
        progress_dialog.setWindowTitle("随机搜索进行中")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)  # 立即显示
        progress_dialog.show()
        
        # 执行随机搜索
        for i in range(n_iter):
            # 检查是否取消了操作
            if progress_dialog.wasCanceled():
                break
            
            # 更新进度
            progress_dialog.setValue(i)
            progress_dialog.setLabelText(f"正在进行随机搜索 ({i+1}/{n_iter})...")
            QApplication.processEvents()  # 处理UI事件
            
            # 随机选择参数
            alpha = random.choice(param_distributions['alpha'])
            beta = random.choice(param_distributions['beta'])
            max_iter = random.choice(param_distributions['max_iter'])
            tol = random.choice(param_distributions['tol'])
            random_state = random.choice(param_distributions['random_state'])
            gpu = random.choice(param_distributions['gpu'])
            
            try:
                # 数据预处理（与MTW.py中一致）
                X_processed = X.copy()
                
                # 转换为 3D 格式 (n_tasks, n_samples, n_features)
                X_processed_3d = np.repeat(X_processed[None, :, :], n_tasks, axis=0)
                y_processed_3d = y.T  # 转置为 (n_tasks, n_samples)
                
                # 创建模型（与MTW.py中一致）
                model = MTW(
                    alpha=alpha,
                    beta=beta,
                    gpu=gpu,
                    max_iter=max_iter,
                    tol=tol
                )
                
                # 自定义交叉验证评估，因为MTW需要特殊的数据格式
                fold_scores = []
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                
                for train_idx, val_idx in kf.split(X_processed):
                    # 分割数据
                    X_train_fold = X_processed[train_idx]
                    y_train_fold = y[train_idx]
                    X_val_fold = X_processed[val_idx]
                    y_val_fold = y[val_idx]
                    
                    # 转换为3D格式
                    X_train_3d = np.repeat(X_train_fold[None, :, :], n_tasks, axis=0)
                    y_train_3d = y_train_fold.T
                    X_val_3d = np.repeat(X_val_fold[None, :, :], n_tasks, axis=0)
                    
                    # 训练模型
                    model.fit(X_train_3d, y_train_3d)
                    
                    # 预测
                    y_pred = model.predict(X_val_3d).T  # 转置回 (n_samples, n_tasks)
                    
                    # 清理预测结果
                    y_pred = clean_data(y_pred)
                    
                    # 计算MSE
                    mse = mean_squared_error(y_val_fold, y_pred)
                    fold_scores.append(mse)
                
                avg_score = np.mean(fold_scores)
                
                # 更新最佳参数
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {
                        'alpha': alpha,
                        'beta': beta,
                        'max_iter': max_iter,
                        'tol': tol,
                        'random_state': random_state,
                        'gpu': gpu,
                        'score': avg_score
                    }
                    
            except Exception as e:
                print(f"参数组合评估失败: alpha={alpha}, beta={beta}, max_iter={max_iter}, error={e}")
                continue
        
        # 关闭进度对话框
        progress_dialog.close()
        
        if best_params is None:
            QMessageBox.warning(parent_window, "搜索失败", "所有参数组合评估都失败了！")
            return None
        
        return best_params
        
    except Exception as e:
        QMessageBox.warning(parent_window, "随机搜索失败", f"随机搜索过程中出现错误：{str(e)}")
        return None


def random_search_remtw(parent_window, data_train, input_columns, output_columns, 
                        n_iter=10, cv_folds=3):
    """
    ReMTW (Reweighted Multitask Wasserstein) 随机搜索函数
    适配ReMTW.py中的数据格式要求
    """
    try:
        # 参数搜索空间
        param_distributions = {
            'alpha': [0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0],  # Wasserstein距离正则化参数
            'beta': [0.05, 0.1, 0.2, 0.3, 0.5, 0.7],  # L1正则化参数
            'max_iter': [500, 1000, 2000, 3000],  # 最大迭代次数
            'tol': [1e-3, 1e-4, 1e-5],  # 收敛容差
            'random_state':[42],  # 随机种子
            'gpu': [False]  # 是否使用GPU加速
        }
        
        # 准备数据
        X = data_train[input_columns].values.astype(float)
        y = data_train[output_columns].values.astype(float)
        
        # 数据清理函数（与ReMTW.py中一致）
        def clean_data(arr):
            """清理数据中的NaN和无穷大值"""
            arr = np.asarray(arr)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
            return arr
        
        X = clean_data(X)
        y = clean_data(y)
        
        # ReMTW需要处理所有输出
        n_tasks = len(output_columns)
        
        best_score = float('inf')  # 最小化MSE
        best_params = None
        
        # 进度对话框
        progress_dialog = QProgressDialog("正在进行ReMTW参数寻优...", "取消", 0, n_iter, parent_window)
        progress_dialog.setWindowTitle("随机搜索进行中")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)  # 立即显示
        progress_dialog.show()
        
        # 执行随机搜索
        for i in range(n_iter):
            # 检查是否取消了操作
            if progress_dialog.wasCanceled():
                break
            
            # 更新进度
            progress_dialog.setValue(i)
            progress_dialog.setLabelText(f"正在进行随机搜索 ({i+1}/{n_iter})...")
            QApplication.processEvents()  # 处理UI事件
            
            # 随机选择参数
            alpha = random.choice(param_distributions['alpha'])
            beta = random.choice(param_distributions['beta'])
            max_iter = random.choice(param_distributions['max_iter'])
            tol = random.choice(param_distributions['tol'])
            random_state = random.choice(param_distributions['random_state'])
            gpu = random.choice(param_distributions['gpu'])
            
            try:
                # 数据预处理（与ReMTW.py中一致）
                X_processed = X.copy()
                
                # 转换为 3D 格式 (n_tasks, n_samples, n_features)
                X_processed_3d = np.repeat(X_processed[None, :, :], n_tasks, axis=0)
                y_processed_3d = y.T  # 转置为 (n_tasks, n_samples)
                
                # 创建模型（与ReMTW.py中一致）
                model = ReMTW(
                    alpha=alpha,
                    beta=beta,
                    gpu=gpu,
                    max_iter=max_iter,
                    tol=tol
                )
                
                # 自定义交叉验证评估，因为ReMTW需要特殊的数据格式
                fold_scores = []
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                
                for train_idx, val_idx in kf.split(X_processed):
                    # 分割数据
                    X_train_fold = X_processed[train_idx]
                    y_train_fold = y[train_idx]
                    X_val_fold = X_processed[val_idx]
                    y_val_fold = y[val_idx]
                    
                    # 转换为3D格式
                    X_train_3d = np.repeat(X_train_fold[None, :, :], n_tasks, axis=0)
                    y_train_3d = y_train_fold.T
                    X_val_3d = np.repeat(X_val_fold[None, :, :], n_tasks, axis=0)
                    
                    # 训练模型
                    model.fit(X_train_3d, y_train_3d)
                    
                    # 预测
                    y_pred = model.predict(X_val_3d).T  # 转置回 (n_samples, n_tasks)
                    
                    # 清理预测结果
                    y_pred = clean_data(y_pred)
                    
                    # 计算MSE
                    mse = mean_squared_error(y_val_fold, y_pred)
                    fold_scores.append(mse)
                
                avg_score = np.mean(fold_scores)
                
                # 更新最佳参数
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {
                        'alpha': alpha,
                        'beta': beta,
                        'max_iter': max_iter,
                        'tol': tol,
                        'random_state': random_state,
                        'gpu': gpu,
                        'score': avg_score
                    }
                    
            except Exception as e:
                print(f"参数组合评估失败: alpha={alpha}, beta={beta}, max_iter={max_iter}, error={e}")
                continue
        
        # 关闭进度对话框
        progress_dialog.close()
        
        if best_params is None:
            QMessageBox.warning(parent_window, "搜索失败", "所有参数组合评估都失败了！")
            return None
        
        return best_params
        
    except Exception as e:
        QMessageBox.warning(parent_window, "随机搜索失败", f"随机搜索过程中出现错误：{str(e)}")
        return None


def random_search_gaussian_process(parent_window, data_train, input_columns, output_columns, 
                                  n_iter=8, cv_folds=2):
    """
    高斯过程随机搜索函数
    适配Gassu_process.py中的MultitaskGPRegressor
    """
    try:
        # 参数搜索空间
        param_distributions = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],  # 学习率
            'training_iterations': [100, 200, 500, 1000],  # 训练迭代次数
            # 'device': ['cpu', 'cuda'],  # 设备选择
            'device': [ 'cuda']  # 设备选择
        }
        
        # 准备数据
        X = data_train[input_columns].values.astype(float)
        y = data_train[output_columns].values.astype(float)
        
        # 数据清理函数
        def clean_data(arr):
            """清理数据中的NaN和无穷大值"""
            arr = np.asarray(arr)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
            return arr
        
        X = clean_data(X)
        y = clean_data(y)
        
        # 优化：如果数据量过大，进行采样
        if len(X) > 500:
            # 随机采样500个样本以加速搜索
            sample_indices = np.random.choice(len(X), size=500, replace=False)
            X = X[sample_indices]
            y = y[sample_indices]
            print(f"数据量过大，已采样500个样本进行参数搜索")
        
        # 获取输入输出维度
        input_dim = X.shape[1]
        output_dim = y.shape[1]
        
        best_score = float('inf')  # 最小化MSE
        best_params = None
        
        # 进度对话框
        progress_dialog = QProgressDialog("正在进行高斯过程参数寻优...", "取消", 0, n_iter, parent_window)
        progress_dialog.setWindowTitle("随机搜索进行中")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)  # 立即显示
        progress_dialog.show()
        
        # 执行随机搜索
        for i in range(n_iter):
            # 检查是否取消了操作
            if progress_dialog.wasCanceled():
                break
            
            # 更新进度
            progress_dialog.setValue(i)
            progress_dialog.setLabelText(f"正在进行随机搜索 ({i+1}/{n_iter})...")
            QApplication.processEvents()  # 处理UI事件
            
            # 随机选择参数
            learning_rate = random.choice(param_distributions['learning_rate'])
            training_iterations = random.choice(param_distributions['training_iterations'])
            device = random.choice(param_distributions['device'])
            
            # 检查CUDA是否可用
            if device == 'cuda' and not torch.cuda.is_available():
                device = 'cpu'
            
            try:
                # 自定义交叉验证评估
                fold_scores = []
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                
                for train_idx, val_idx in kf.split(X):
                    # 分割数据
                    X_train_fold = X[train_idx]
                    y_train_fold = y[train_idx]
                    X_val_fold = X[val_idx]
                    y_val_fold = y[val_idx]
                    
                    # 创建模型
                    model = MultitaskGPRegressor(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        learning_rate=learning_rate,
                        training_iterations=training_iterations,
                        device=device
                    )
                    
                    # 训练模型（设置verbose=False避免输出干扰）
                    model.fit(X_train_fold, y_train_fold, verbose=False)
                    
                    # 预测
                    y_pred = model.predict(X_val_fold)
                    
                    # 清理预测结果
                    y_pred = clean_data(y_pred)
                    
                    # 计算MSE
                    mse = mean_squared_error(y_val_fold, y_pred)
                    fold_scores.append(mse)
                
                avg_score = np.mean(fold_scores)
                
                # 更新最佳参数
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {
                        'learning_rate': learning_rate,
                        'training_iterations': training_iterations,
                        'device': device,
                        'score': avg_score
                    }
                    
            except Exception as e:
                print(f"参数组合评估失败: learning_rate={learning_rate}, training_iterations={training_iterations}, error={e}")
                continue
        
        # 关闭进度对话框
        progress_dialog.close()
        
        if best_params is None:
            QMessageBox.warning(parent_window, "搜索失败", "所有参数组合评估都失败了！")
            return None
        
        return best_params
        
    except Exception as e:
        QMessageBox.warning(parent_window, "随机搜索失败", f"随机搜索过程中出现错误：{str(e)}")
        return None


def random_search_mmoe(parent_window, data_train, input_columns, output_columns, 
                       n_iter=6, cv_folds=2):
    """
    MMoE (Multi-gate Mixture-of-Experts) 随机搜索函数
    适配Algorithms.py中的MMoERegressor
    """
    try:
        # 参数搜索空间
        param_distributions = {
            'mmoe_num_experts': [3, 5, 8, 10],  # 专家数量
            'mmoe_expert_hidden': [32, 64, 128],  # 专家网络隐藏层大小
            'mmoe_learning_rate': [0.001, 0.005, 0.01],  # 学习率
            'mmoe_dropout_rate': [0.1, 0.2, 0.3],  # Dropout率
            'mmoe_epochs': [50, 100, 200],  # 训练轮数
            'mmoe_batch_size': [16, 32, 64],  # 批处理大小
            'mmoe_lambda_balance': [0.1, 0.2, 0.3],  # 平衡系数
            'mmoe_scale_features': [True, False]  # 是否标准化特征
        }
        
        # 准备数据
        X = data_train[input_columns].values.astype(float)
        y = data_train[output_columns].values.astype(float)
        
        # 数据清理函数
        def clean_data(arr):
            """清理数据中的NaN和无穷大值"""
            arr = np.asarray(arr)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
            return arr
        
        X = clean_data(X)
        y = clean_data(y)
        
        # 优化：如果数据量过大，进行采样
        if len(X) > 1000:
            # 随机采样1000个样本以加速搜索
            sample_indices = np.random.choice(len(X), size=1000, replace=False)
            X = X[sample_indices]
            y = y[sample_indices]
            print(f"数据量过大，已采样1000个样本进行参数搜索")
        
        # 获取输入输出维度
        input_dim = X.shape[1]
        output_dim = y.shape[1]
        
        best_score = float('inf')  # 最小化MSE
        best_params = None
        
        # 进度对话框
        progress_dialog = QProgressDialog("正在进行MMoE参数寻优...", "取消", 0, n_iter, parent_window)
        progress_dialog.setWindowTitle("随机搜索进行中")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)  # 立即显示
        progress_dialog.show()
        
        # 执行随机搜索
        for i in range(n_iter):
            # 检查是否取消了操作
            if progress_dialog.wasCanceled():
                break
            
            # 更新进度
            progress_dialog.setValue(i)
            progress_dialog.setLabelText(f"正在进行随机搜索 ({i+1}/{n_iter})...")
            QApplication.processEvents()  # 处理UI事件
            
            # 随机选择参数
            mmoe_num_experts = random.choice(param_distributions['mmoe_num_experts'])
            mmoe_expert_hidden = random.choice(param_distributions['mmoe_expert_hidden'])
            mmoe_learning_rate = random.choice(param_distributions['mmoe_learning_rate'])
            mmoe_dropout_rate = random.choice(param_distributions['mmoe_dropout_rate'])
            mmoe_epochs = random.choice(param_distributions['mmoe_epochs'])
            mmoe_batch_size = random.choice(param_distributions['mmoe_batch_size'])
            mmoe_lambda_balance = random.choice(param_distributions['mmoe_lambda_balance'])
            mmoe_scale_features = random.choice(param_distributions['mmoe_scale_features'])
            
            try:
                # 自定义交叉验证评估
                fold_scores = []
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                
                for train_idx, val_idx in kf.split(X):
                    # 分割数据
                    X_train_fold = X[train_idx]
                    y_train_fold = y[train_idx]
                    X_val_fold = X[val_idx]
                    y_val_fold = y[val_idx]
                    
                    # 创建模型
                    model = MMoERegressor(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        num_experts=mmoe_num_experts,
                        expert_hidden=mmoe_expert_hidden,
                        learning_rate=mmoe_learning_rate,
                        dropout_rate=mmoe_dropout_rate,
                        num_epochs=mmoe_epochs,
                        batch_size=mmoe_batch_size,
                        lambda_balance=mmoe_lambda_balance
                    )
                    
                    # 训练模型（设置verbose=False避免输出干扰）
                    model.fit(X_train_fold, y_train_fold, verbose=False)
                    
                    # 预测
                    y_pred = model.predict(X_val_fold)
                    
                    # 清理预测结果
                    y_pred = clean_data(y_pred)
                    
                    # 计算MSE
                    mse = mean_squared_error(y_val_fold, y_pred)
                    fold_scores.append(mse)
                
                avg_score = np.mean(fold_scores)
                
                # 更新最佳参数
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {
                        'mmoe_num_experts': mmoe_num_experts,
                        'mmoe_expert_hidden': mmoe_expert_hidden,
                        'mmoe_learning_rate': mmoe_learning_rate,
                        'mmoe_dropout_rate': mmoe_dropout_rate,
                        'mmoe_epochs': mmoe_epochs,
                        'mmoe_batch_size': mmoe_batch_size,
                        'mmoe_lambda_balance': mmoe_lambda_balance,
                        'mmoe_scale_features': mmoe_scale_features,
                        'score': avg_score
                    }
                    
            except Exception as e:
                print(f"参数组合评估失败: mmoe_num_experts={mmoe_num_experts}, mmoe_expert_hidden={mmoe_expert_hidden}, error={e}")
                continue
        
        # 关闭进度对话框
        progress_dialog.close()
        
        if best_params is None:
            QMessageBox.warning(parent_window, "搜索失败", "所有参数组合评估都失败了！")
            return None
        
        return best_params
        
    except Exception as e:
        QMessageBox.warning(parent_window, "随机搜索失败", f"随机搜索过程中出现错误：{str(e)}")
        return None

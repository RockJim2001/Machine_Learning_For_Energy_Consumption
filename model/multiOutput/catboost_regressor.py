#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：catboost_regressor.py
@Author ：RockJim
@Date ：2023/12/13 11:30
@Description ：catboostRegressor来实现多输出回归预测
@Version ：1.0
"""
from model.multiOutput.base import Base
from catboost import CatBoostRegressor as CatBoostRegression


# import torch
# import numpy as np
# from catboost import CatBoostRegressor
# from sklearn.metrics import mean_squared_error
#
# # 生成一些示例数据
# X_np = np.random.rand(100, 2)
# y_np = np.random.rand(100, 3)
#
# # 将 NumPy 数组转换为张量
# X_tensor = torch.tensor(X_np, dtype=torch.float32)
# y_tensor = torch.tensor(y_np, dtype=torch.float32)
#
# # 将张量转换为 NumPy 数组
# X_np = X_tensor.numpy()
# y_np = y_tensor.numpy()
#
# # 创建 CatBoost 回归模型
# catboost_regressor = CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, loss_function='MultiRMSE')
#
# # 训练模型
# catboost_regressor.fit(X_np, y_np, verbose=50)
#
# # 进行预测
# y_pred_np = catboost_regressor.predict(X_np)
#
# # 评估模型性能
# mse = mean_squared_error(y_np, y_pred_np)
# print(f'Mean Squared Error: {mse}')


class CATBoostRegressor(Base):
    def __init__(self, x_train, y_train, x_test, y_test):
        super(CATBoostRegressor, self).__init__(x_train, y_train, x_test, y_test)
        model = CatBoostRegression(iterations=100, depth=12, learning_rate=0.1, loss_function='MultiRMSE')
        self.model = model

    def train(self):
        # 将张量转换为 NumPy 数组
        x_train = self.x_train.numpy()
        y_train = self.y_train.numpy()
        self.model.fit(x_train, y_train, verbose=50)

    def test(self):
        # 将张量转换为 NumPy 数组
        x_test = self.x_test.numpy()
        y_pred = self.model.predict(x_test)
        return y_pred

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：multilayer_perceptron_regressor.py
@Author ：RockJim
@Date ：2023/12/12 19:41
@Description ：基于多层感知机的多输出回归预测
@Version ：1.0
"""
from sklearn.neural_network import MLPRegressor

from model.multiOutput.base import Base


# from sklearn.neural_network import MLPRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
#
# # 生成一些示例数据
# from sklearn.datasets import make_regression
# X, y = make_regression(n_samples=100, n_features=2, n_targets=3, random_state=42)
#
# # 将数据集分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 创建多层感知机回归模型
# mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
#
# # 训练模型
# mlp_regressor.fit(X_train, y_train)
#
# # 进行预测
# y_pred = mlp_regressor.predict(X_test)
#
# # 评估模型性能
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')

class MultilayerPerceptronRegressor(Base):
    def __init__(self, x_train, y_train, x_test, y_test):
        super(MultilayerPerceptronRegressor, self).__init__(x_train, y_train, x_test, y_test)
        # 创建支持向量机回归模型
        model = MLPRegressor(hidden_layer_sizes=(20, 20), max_iter=100, random_state=42)

        self.model = model

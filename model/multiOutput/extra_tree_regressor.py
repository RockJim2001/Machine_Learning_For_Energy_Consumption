#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：extra_tree_regressor.py
@Author ：RockJim
@Date ：2023/12/12 17:09
@Description ：多输出的极限树回归
@Version ：1.0
"""
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import ExtraTreeRegressor as ExtraTreeRegression

from model.multiOutput.base import Base


# from sklearn.datasets import make_regression
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.metrics import mean_squared_error
#
# # 生成一些示例数据
# X, y = make_regression(n_samples=100, n_features=2, n_targets=3, random_state=42)
#
# # 将数据集分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 创建逻辑回归回归模型
# logistic_regression = LogisticRegression()
#
# # 使用MultiOutputRegressor包装逻辑回归回归模型
# model = MultiOutputRegressor(logistic_regression)
#
# # 训练模型
# model.fit(X_train, y_train)
#
# # 进行预测
# y_pred = model.predict(X_test)
#
# # 评估模型性能
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')


class ExtraTreeRegressor(Base):
    def __init__(self, x_train, y_train, x_test, y_test):
        super(ExtraTreeRegressor, self).__init__(x_train, y_train, x_test, y_test)
        # 创建支持向量机回归模型
        # svm_regressor = LogisticRegression()
        svm_regressor = ExtraTreeRegression()

        # 使用MultiOutputRegressor包装支持向量机模型
        model = MultiOutputRegressor(svm_regressor)
        self.model = model


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：gradient_boosted_decision_trees_regressor.py
@Author ：RockJim
@Date ：2023/12/13 14:51
@Description ：Gradient Boosted Decision Trees（GBDT）实现多输出回归预测
@Version ：1.0
"""
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from model.multiOutput.base import Base


# from sklearn.datasets import make_regression
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.metrics import mean_squared_error
#
# # 生成一些示例数据
# X, y = make_regression(n_samples=100, n_features=2, n_targets=3, random_state=42)
#
# # 将数据集分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 创建 GBDT 回归模型
# gbdt_regressor = GradientBoostingRegressor()
#
# # 使用 MultiOutputRegressor 包装 GBDT 模型
# model = MultiOutputRegressor(gbdt_regressor)
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


class GBDTRegressor(Base):
    def __init__(self, x_train, y_train, x_test, y_test):
        super(GBDTRegressor, self).__init__(x_train, y_train, x_test, y_test)
        # 创建 GBDT 回归模型
        params = {
            'learning_rate': 0.05,  # 设置学习率
            'n_estimators': 500,  # 设置树的数量
            'max_depth': 10,  # 设置树的最大深度
            'min_samples_split': 2,  # 设置节点划分的最小样本数
            'min_samples_leaf': 1,  # 设置叶子节点的最小样本数
            'subsample': 0.8  # 设置子采样率
        }
        gbdt_regressor = GradientBoostingRegressor()

        # 使用MultiOutputRegressor包装支持向量机模型
        model = MultiOutputRegressor(gbdt_regressor)
        self.model = model

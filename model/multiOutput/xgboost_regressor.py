#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：xgboost_regressor.py
@Author ：RockJim
@Date ：2023/12/13 11:09
@Description ：XGBoost实现多输出回归
@Version ：1.0
"""
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from model.multiOutput.base import Base


# from sklearn.datasets import make_regression
# from sklearn.model_selection import train_test_split
# from sklearn.multioutput import MultiOutputRegressor
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error
#
# # 生成一些示例数据
# X, y = make_regression(n_samples=100, n_features=2, n_targets=3, random_state=42)
#
# # 将数据集分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 创建XGBoost回归模型
# xgb_regressor = XGBRegressor()
#
# # 使用MultiOutputRegressor包装XGBoost模型
# model = MultiOutputRegressor(xgb_regressor)
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


class XGBoostRegressor(Base):
    def __init__(self, x_train, y_train, x_test, y_test):
        super(XGBoostRegressor, self).__init__(x_train, y_train, x_test, y_test)
        # 创建支持向量机回归模型
        # 自定义参数设置
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'reg:squarederror'
        }
        xgboost_regressor = XGBRegressor(**params)
        # xgboost_regressor = XGBRegressor()

        # 使用MultiOutputRegressor包装支持向量机模型
        model = MultiOutputRegressor(xgboost_regressor)
        self.model = model

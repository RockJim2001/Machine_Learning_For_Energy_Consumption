#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：linear_regression.py
@Author ：RockJim
@Date ：2023/12/2 17:06
@Description ：线性回归
@Version ：1.0
"""
# linear regression for multioutput regression
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from torch import nn

from model.multiOutput.base import Base


# # create datasets
# X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1)
# # define model
# model = LinearRegression()
# # fit model
# model.fit(X, y)
# # make a prediction
# data_in = [
#     [-2.02220122, 0.31563495, 0.82797464, -0.30620401, 0.16003707, -1.44411381, 0.87616892, -0.50446586, 0.23009474,
#      0.76201118]]
# yhat = model.predict(data_in)
# # summarize prediction
# print(yhat[0])


class LinearRegressor(Base):
    def __init__(self, x_train, y_train, x_test, y_test):
        super(LinearRegressor, self).__init__(x_train, y_train, x_test, y_test)

        model = LinearRegression()
        self.model = model
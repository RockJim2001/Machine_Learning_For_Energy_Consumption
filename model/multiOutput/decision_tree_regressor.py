#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：decision_tree_regressor.py
@Author ：RockJim
@Date ：2023/12/2 17:09
@Description ：K-折交叉验证
@Version ：1.0
"""
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor as DecisionTreeRegression

from model.multiOutput.base import Base


# # evaluate multioutput regression model with k-fold cross-validation
# from numpy import absolute
# from numpy import mean
# from numpy import std
# from sklearn.datasets import make_regression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedKFold
#
# # create datasets
# X, Y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1)
# # define model
# model = DecisionTreeRegressor()
# # evaluate model
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(model, X, Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
# # summarize performance
# n_scores = absolute(n_scores)
# print('Result: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# 当数据规模很大时，不需要进行划分train、Vail、test三部分，参考https://zhuanlan.zhihu.com/p/83841282

class DecisionTreeRegressor(Base):
    def __init__(self, x_train, y_train, x_test, y_test):
        super(DecisionTreeRegressor, self).__init__(x_train, y_train, x_test, y_test)
        model = DecisionTreeRegression()
        self.model = model
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        self.cv = cv

    def train(self):
        # n_scores = cross_val_score(self.model, self.x_train, self.y_train, scoring='neg_mean_squared_error',
        #                            cv=self.cv, n_jobs=-1, error_score='raise')
        # print(f'n_scores: {-n_scores.mean()}')
        self.model.fit(self.x_train, self.y_train)

    def test(self):
        # self.training = False
        # n_scores = cross_val_score(self.model, self.x_test, self.y_test, scoring='neg_mean_squared_error',
        #                            cv=2, n_jobs=-1, error_score='raise')
        # print(f'n_scores: {-n_scores.mean()}')
        pred_test = self.model.predict(self.x_test)
        return pred_test


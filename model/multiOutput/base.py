#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：base.py
@Author ：RockJim
@Date ：2023/12/2 21:44
@Description ：base模型
@Version ：1.0
"""
from torch import nn


class Base(nn.Module):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = None

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def test(self):
        pred_y = self.model.predict(self.x_test)
        return pred_y

    def predict(self, predict_data):
        pred_y = self.model.predict(predict_data)
        return pred_y


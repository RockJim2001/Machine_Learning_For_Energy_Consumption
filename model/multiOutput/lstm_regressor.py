#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：lstm_regressor.py
@Author ：RockJim
@Date ：2023/12/13 14:38
@Description ：基于LSTM实现多输出回归
@Version ：1.0
"""
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from model.multiOutput.base import Base


# 创建 LSTM 模型
class LSTMRegression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMRegression, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class LSTMRegressor(Base):
    def __init__(self, x_train, y_train, x_test, y_test, batch_size=32, shuffle=True):
        super(LSTMRegressor, self).__init__(x_train, y_train, x_test, y_test)
        model = LSTMRegression(input_size=31, hidden_size=5, output_size=6)
        self.model = model
        # 将数据转为 DataLoader
        train_dataset = TensorDataset(self.x_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.train_dataset = train_dataset
        self.train_loader = train_loader
        # test_dataset = TensorDataset(self.x_test, self.y_test)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        # self.test_dataset = test_dataset
        # self.test_loader = test_loader

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

    def train(self, num_epochs=50):
        for epoch in range(num_epochs):
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs.unsqueeze(1))  # LSTM 需要输入维度为 (batch_size, seq_len, input_size)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

    def test(self):
        # 进行预测
        with torch.no_grad():
            pred_test = self.model(self.x_test.unsqueeze(1))
        return pred_test

    def predict(self, predict_data):
        with torch.no_grad():
            predict_result = self.model(predict_data.unsqueeze(1))
        return predict_result
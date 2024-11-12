#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：load_model_demo.py
@Author ：RockJim
@Date ：2024/2/19 21:33
@Description ：None
@Version ：1.0
"""
import os

import joblib
import numpy as np
import torch
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config.config import log_dir, ROOT_PATH
from dataset.data_load import data_processing, load_data


def load_XGBoost_model(path: str = ''):
    # 加载模型
    model = torch.load(path)
    # 加载数据集 ---- 通过文件加载, test_data.csv文件也是放在
    file_path = os.path.join(ROOT_PATH, 'resource', 'test_data.csv')
    # file_data = genfromtxt(file_path, delimiter=',', ndmin=2)

    # 加载数据集 ---- 通过数组
    file_data = np.array([[14.03640, 0.80227, 5.74372, 0.74985, 13.45913, 0.92773, 25.07251, 26.90743, 19.00000]])
    x = file_data
    # 归一化处理
    scaler_x = joblib.load(os.path.join(log_dir, 'scaler_x.pkl'))
    x_test = scaler_x.transform(x)

    # 转为tensor格式
    X_test = torch.tensor(x_test, dtype=torch.float32)
    model.x_test = X_test
    pred_test = model.test()
    # 对预测结果进行反归一化处理
    scaler_y = joblib.load(os.path.join(log_dir, 'scaler_y.pkl'))
    restored_data_pred = scaler_y.inverse_transform(pred_test)
    print(f'变量值：')
    print(f'预测数据：{pred_test}')
    print(f'反归一化后的数据：{restored_data_pred}')
    print(restored_data_pred[0][0])


if __name__ == '__main__':
    model_weight_path = '../result/XGBoostRegressor/XGBoostRegressor.pth'
    load_XGBoost_model(model_weight_path)

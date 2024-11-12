#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：utils.py
@Author ：RockJim
@Date ：2023/12/2 16:57
@Description ：数据加载的工具类
@Version ：1.0
"""
import os

import torch

from config.config import ROOT_PATH, DATASET_NAME, parent_dir
from dataset.data_load import load_data, data_processing


def data_load():
    # 加载数据集
    x_data, y_data = load_data(os.path.join(ROOT_PATH, 'resource', parent_dir, DATASET_NAME))
    # 数据归一化处理
    x_train, x_test, y_train, y_test = data_processing(x_data, y_data)

    # 将这些值转为tensor格式
    X_train = torch.tensor(x_train, dtype=torch.float32)
    Y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(x_test, dtype=torch.float32)
    Y_test = torch.tensor(y_test, dtype=torch.float32)
    return X_train, Y_train, X_test, Y_test
    # return x_train, y_train, x_test, y_test

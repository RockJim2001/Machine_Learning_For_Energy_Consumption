#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：ML_project 
@Product_name ：PyCharm
@File ：data_load.py
@Author ：RockJim
@Date ：2023/7/24 23:46
@Description ：进行数据加载
@Version ：1.0 
'''
import os
import time

import joblib
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config.config import log_dir
from config.log_config import log

logger = log().getLogger(__name__)


def load_data(file_path: str):
    """
        从文件中读取数据，以list的方式进行返回
    :param file_path: 数据文件存储路径
    :return:
    """
    assert os.path.exists(file_path), logger.error("数据集路径{}不存在".format(file_path))
    logger.info("从{}加载数据集".format(file_path))
    start_time = time.time()
    root_path, full_file_name = os.path.split(file_path)
    file_name, file_format = full_file_name.split('.')
    if file_format == 'csv':
        file_data = genfromtxt(file_path, delimiter=',')

    # col_index = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    col_index = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]    # 30输入
    x_data = file_data[:, col_index]
    # y_data = file_data[:, [14, 15, 16, 17, 18, 19]]
    # y_data = file_data[:, [33, 34, 35, 36, 38, 39]]     # 6输出
    y_data = file_data[:, [32, 33, 34, 35]]     # 4输出
    # y_data = file_data[:, [14]]
    end_time = time.time()
    run_time = end_time - start_time
    logger.info("数据加载完成，耗时{}".format(run_time))
    return x_data, y_data


def data_processing(x, y):
    """
        对数据进行预处理操作
    :return:
    """
    logger.info("数据处理")

    # 数据标准化预处理
    logger.info("数据标准化预处理")
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    joblib.dump(scaler, log_dir + '/scaler_x.pkl')
    y = scaler.fit_transform(y)
    joblib.dump(scaler, log_dir + '/scaler_y.pkl')
    # 拆分数据集
    logger.info("拆分数据集")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

    if y_train.shape[1] == 1:
        y_train = np.array(y_train).reshape(-1, 1)
    if y_test.shape[1] == 1:
        y_test = np.array(y_test).reshape(-1, 1)
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.fit_transform(x_test)
    # joblib.dump(scaler, log_dir + '/scaler_x.pkl')
    # y_train = scaler.fit_transform(y_train)
    # y_test = scaler.fit_transform(y_test)
    if y_train.shape[1] == 1:
        y_train = scaler.fit_transform(y_train).ravel()
    if y_test.shape[1] == 1:
        y_test = scaler.fit_transform(y_test).ravel()
    # joblib.dump(scaler, log_dir + "/scaler_y.pkl")
    return x_train, x_test, y_train, y_test

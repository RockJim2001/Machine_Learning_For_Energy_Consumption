import numpy as np
import pandas as pd
import pickle
import geatpy as ea

import os

import joblib
import torch
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config.config import log_dir, ROOT_PATH, DATASET_NAME

"""
决策变量的最小化目标4目标优化问题
min f1
min f2
min f3
min f4

s.t.
x1 x2 x3 x4 x5 x6 x7 x8 x9
"""


# 读取训练好的机器学习模型
def load_XGBoost_model(path, x):
    # 加载模型
    model = torch.load(path)
    # 加载数据集 ---- 通过文件加载, test_data.csv文件也是放在
    file_path = os.path.join(ROOT_PATH, 'resource', 'test_data.csv')

    # 加载数据集 ---- 通过数组
    # file_data = np.array([[14.03640, 0.80227, 5.74372, 0.74985, 13.45913, 0.92773, 25.07251, 26.90743, 19.00000]])
    # x = file_data
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
    # print(f'变量值：')
    # print(f'预测数据：{pred_test}')
    # print(f'反归一化后的数据：{restored_data_pred}')
    # print("哈哈哈")
    return restored_data_pred[0][0], restored_data_pred[0][1], restored_data_pred[0][2], restored_data_pred[0][3]


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=4):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 30  # 初始化Dim（决策变量维数）
        maxormins = [1, 1, 1, -1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.4, 0.2, 0.5, 0.1, 0.2, 1.4, 0.4,
              5.0, 0.7, 5.0, 0.7, 10.0, 0.7, 24.0, 15.0]  # 决策变量下界
        ub = [360, 0.22, 0.25, 0.9, 0.9, 0.9, 0.9, 0.7, 0.7, 0.7, 0.7, 1.0, 1.0, 1.0, 1.0, 1.5, 0.8, 1.5, 0.35, 0.4, 2.4, 0.8,
              15.0, 1.0, 13.0, 1.0, 20.0, 1.0, 28.0, 45.0]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=float)
        popsize = Vars.shape[0]
        F1 = np.array([float("Inf")] * popsize).reshape(popsize, 1)
        F2 = np.array([float("Inf")] * popsize).reshape(popsize, 1)
        F3 = np.array([float("Inf")] * popsize).reshape(popsize, 1)
        F4 = np.array([float("Inf")] * popsize).reshape(popsize, 1)
        for i in range(popsize):
            x1 = Vars[i, [0]][0]
            x2 = Vars[i, [1]][0]
            x3 = Vars[i, [2]][0]
            x4 = Vars[i, [3]][0]
            x5 = Vars[i, [4]][0]
            x6 = Vars[i, [5]][0]
            x7 = Vars[i, [6]][0]
            x8 = Vars[i, [7]][0]
            x9 = Vars[i, [8]][0]
            x10 = Vars[i, [9]][0]
            x11 = Vars[i, [10]][0]
            x12 = Vars[i, [11]][0]
            x13 = Vars[i, [12]][0]
            x14 = Vars[i, [13]][0]

            x15 = Vars[i, [14]][0]
            x16 = Vars[i, [15]][0]
            x17 = Vars[i, [16]][0]
            x18 = Vars[i, [17]][0]
            x19 = Vars[i, [18]][0]

            x20 = Vars[i, [19]][0]
            x21 = Vars[i, [20]][0]
            x22 = Vars[i, [21]][0]
            x23 = Vars[i, [22]][0]
            x24 = Vars[i, [23]][0]

            x25 = Vars[i, [24]][0]
            x26 = Vars[i, [25]][0]
            x27 = Vars[i, [26]][0]
            x28 = Vars[i, [27]][0]
            x29 = Vars[i, [28]][0]

            x30 = Vars[i, [29]][0]

            # 对x进行归一化操作，使其在0-1之间
            X = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18,
                          x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30]).reshape(1, -1)
            # 计算目标函数值
            f1, f2, f3, f4 = load_XGBoost_model(os.path.join(ROOT_PATH, 'result_change', DATASET_NAME.split('.')[0],
                                                             'MultilayerPerceptronRegressor/MultilayerPerceptronRegressor.pth'),
                                                X)

            F1[i, 0] = f1
            F2[i, 0] = f2
            F3[i, 0] = f3
            F4[i, 0] = f4

        pop.ObjV = np.hstack([F1, F2, F3, F4])  # 把求得的目标函数值赋值给种群pop的ObjV

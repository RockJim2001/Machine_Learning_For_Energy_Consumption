#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：main.py
@Author ：RockJim
@Date ：2023/8/1 15:36
@Description ：ML项目的主目录
@Version ：1.0
"""
import json
import os
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from config.config import log_dir, ROOT_PATH, DATASET_NAME, parent_dir
from config.log_config import log
from dataset.utils import data_load
from evaluate.evaluate import relative_absolute_error, relative_squared_error
from model.multiOutput.ada_boost_regressor import AdaBoostRegressor
from model.multiOutput.bagging_regressor import BaggingRegressor
from model.multiOutput.catboost_regressor import CATBoostRegressor
from model.multiOutput.decision_tree_regressor import DecisionTreeRegressor
from model.multiOutput.gradient_boosted_decision_trees_regressor import GBDTRegressor
from model.multiOutput.kn_neighbors_regressor import KNeighborsRegressor
from model.multiOutput.lasso_regressor import LASSORegressor
from model.multiOutput.lgbm_regressor import LGBMRegressor
from model.multiOutput.linear_regression import LinearRegressor
from model.multiOutput.extra_tree_regressor import ExtraTreeRegressor
from model.multiOutput.lstm_regressor import LSTMRegressor
from model.multiOutput.multilayer_perceptron_regressor import MultilayerPerceptronRegressor
from model.multiOutput.random_forest_regressor import RandomForestRegressor
from model.multiOutput.support_vector_machine import SupportVectorMachine
from model.multiOutput.xgboost_regressor import XGBoostRegressor

logger = log().getLogger(__name__)


# 定义一个自定义解码器函数
def custom_decoder(obj):
    # 这里可以根据需要添加自定义的解码逻辑
    # 这个示例中直接返回原始对象
    return obj


# 自定义编码函数，将 float32 转为 float
def custom_encoder(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError("Object of type {} is not JSON serializable".format(type(obj)))


def train_and_eval(model_name: str, x_train, y_train, x_test, y_test):
    """
        进行训练和评估
    :param y_test:
    :param x_test:
    :param y_train:
    :param x_train:
    :param model_name:
    :return:
    """
    assert model_name in ['LinearRegressor', 'KNeighborsRegressor', 'RandomForestRegressor', 'DecisionTreeRegressor',
                          'SupportVectorMachine', 'MultilayerPerceptronRegressor', 'ExtraTreeRegressor',
                          'XGBoostRegressor', 'LGBMRegressor', 'GBDTRegressor', 'CATBoostRegressor',
                          'LSTMRegressor', 'AdaBoostRegressor', 'BaggingRegressor', 'LASSORegressor']
    if model_name == 'LinearRegressor':
        model = LinearRegressor(x_train, y_train, x_test, y_test)
    elif model_name == 'KNeighborsRegressor':
        model = KNeighborsRegressor(x_train, y_train, x_test, y_test)
    elif model_name == 'RandomForestRegressor':
        model = RandomForestRegressor(x_train, y_train, x_test, y_test)
    elif model_name == 'DecisionTreeRegressor':
        model = DecisionTreeRegressor(x_train, y_train, x_test, y_test)
    elif model_name == 'SupportVectorMachine':
        model = SupportVectorMachine(x_train, y_train, x_test, y_test)
    elif model_name == 'MultilayerPerceptronRegressor':
        model = MultilayerPerceptronRegressor(x_train, y_train, x_test, y_test)
    elif model_name == 'ExtraTreeRegressor':
        model = ExtraTreeRegressor(x_train, y_train, x_test, y_test)
    elif model_name == 'XGBoostRegressor':
        model = XGBoostRegressor(x_train, y_train, x_test, y_test)
    elif model_name == 'LGBMRegressor':
        model = LGBMRegressor(x_train, y_train, x_test, y_test)
    elif model_name == 'GBDTRegressor':
        model = GBDTRegressor(x_train, y_train, x_test, y_test)
    elif model_name == 'CATBoostRegressor':
        model = CATBoostRegressor(x_train, y_train, x_test, y_test)
    elif model_name == 'LSTMRegressor':
        model = LSTMRegressor(x_train, y_train, x_test, y_test)
    elif model_name == 'AdaBoostRegressor':
        model = AdaBoostRegressor(x_train, y_train, x_test, y_test)
    elif model_name == 'BaggingRegressor':
        model = BaggingRegressor(x_train, y_train, x_test, y_test)
    elif model_name == 'LASSORegressor':
        model = LASSORegressor(x_train, y_train, x_test, y_test)
    else:
        model = None
    return model


def draw_predict_picture(true, pred):
    """
        绘制回归预测得图形
    :return:
    """
    plt.plot(true, label='Actual Values', marker='o')
    plt.plot(pred, label='Predicted Values', marker='x')

    # 添加标签和标题
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.title('Regression Prediction Line Chart')
    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()


def evaluation(X_test, X_train, Y_test, pred_test, Y_train, train_time=0, test_time=0):
    # 将数据进行恢复
    scaler_x = joblib.load(os.path.join(log_dir, 'scaler_x.pkl'))

    restored_data_x_test = scaler_x.inverse_transform(X_test)
    restored_data_x_train = scaler_x.inverse_transform(X_train)

    scaler_y = joblib.load(os.path.join(log_dir, 'scaler_y.pkl'))

    restored_data_y_test = scaler_y.inverse_transform(Y_test)
    restored_data_pred = scaler_y.inverse_transform(pred_test)
    restored_data_y_train = scaler_y.inverse_transform(Y_train)

    # 计算评价指标
    # ------------------------- mse ----------------------------
    mse_linear = mean_squared_error(pred_test, Y_test)
    print(f"mse:{mse_linear}")

    mse_linear_original = mean_squared_error(restored_data_y_test, restored_data_pred)
    print(f"original_mse:{mse_linear_original}")

    # ------------------------- R^2 ----------------------------
    r2_linear = r2_score(pred_test, Y_test)
    print(f"r2:{r2_linear}")

    r2_linear_original = r2_score(restored_data_y_test, restored_data_pred)
    print(f"original_r2:{r2_linear_original}")

    # ------------------------- mae ----------------------------
    mae_linear = mean_absolute_error(pred_test, Y_test)
    print(f"mae:{mae_linear}")

    mae_linear_original = mean_absolute_error(restored_data_y_test, restored_data_pred)
    print(f"original_mae:{mae_linear_original}")

    # ------------------------- rae ----------------------------
    rae_linear = relative_absolute_error(pred_test, Y_test)
    print(f"rae:{rae_linear}")

    rae_linear_original = relative_absolute_error(restored_data_y_test, restored_data_pred)
    print(f"original_rae:{rae_linear_original}")

    # ------------------------- rse ----------------------------
    rse_linear = relative_squared_error(pred_test, Y_test)
    print(f"rse:{rse_linear}")

    rse_linear_original = relative_squared_error(restored_data_y_test, restored_data_pred)
    print(f"original_rse:{rse_linear_original}")

    # draw_predict_picture(restored_data[:, 0], restored_data_pred[:, 0])
    result = {}
    result['name'] = name
    result['total_data'] = {'train_time': train_time,
                            'test_time': test_time,
                            'mse': mse_linear,
                            'mse_original': mse_linear_original,
                            'r2': r2_linear,
                            'r2_original': r2_linear_original,
                            'mae': mae_linear,
                            'mae_original': mae_linear_original,
                            'rae': rae_linear,
                            'rae_original': rae_linear_original,
                            'rse': rse_linear,
                            'rse_original': rse_linear_original,
                            }
    for i in range(4):
        result['y' + str(i + 1) + '_data'] = {
            'train_time': train_time,
            'test_time': test_time,
            'mse': mean_squared_error(pred_test[:, i], Y_test[:, i]),
            'mse_original': mean_squared_error(restored_data_y_test[:, i], restored_data_pred[:, i]),
            'r2': r2_score(pred_test[:, i], Y_test[:, i]),
            'r2_original': r2_score(restored_data_y_test[:, i], restored_data_pred[:, i]),
            'mae': mean_absolute_error(pred_test[:, i], Y_test[:, i]),
            'mae_original': mean_absolute_error(restored_data_y_test[:, i], restored_data_pred[:, i]),
            'rae': relative_absolute_error(pred_test[:, i], Y_test[:, i]),
            'rae_original': relative_absolute_error(restored_data_y_test[:, i], restored_data_pred[:, i]),
            'rse': relative_squared_error(pred_test[:, i], Y_test[:, i]),
            'rse_original': relative_squared_error(restored_data_y_test[:, i], restored_data_pred[:, i]),
        }
    data_train = np.hstack((restored_data_x_train, restored_data_y_train))
    data_test = np.hstack((restored_data_x_test, restored_data_y_test, restored_data_pred))
    # data = np.hstack((data_x, data_y))
    return result, (data_train, data_test)


def main(X_train, Y_train, X_test, Y_test, name='LSTMRegressor'):
    #  -----------------------------线性回归----------------------------
    # name = 'LSTMRegressor'
    model = train_and_eval(name, X_train, Y_train, X_test, Y_test)
    # 进行训练
    start_time = int(datetime.now().timestamp())
    model.train()
    middle_time = int(datetime.now().timestamp())
    train_time = middle_time - start_time
    print(f"训练花费了{train_time}")
    # 进行预测
    pred_test = model.test()
    end_time = int(datetime.now().timestamp())

    test_time = end_time - middle_time
    print(f"测试花费了{test_time}")

    return evaluation(X_test, X_train, Y_test, pred_test, Y_train, train_time, test_time), model


# 根据计算出来的结果，进行集成
def boostResult(result, X_train, Y_train, X_test, Y_test):
    # 首先从result中选择最优的三个模型进行加权
    weight = [1 / 3, 1 / 3, 1 / 3]
    sorted_result = sorted(result, key=lambda x: x['data']['r2'], reverse=True)
    # 最佳的模型
    top_three_names = [item['name'] for item in sorted_result[:3]]
    # 进行这几个模型的结果汇总
    # X_train, Y_train, X_test, Y_test = data_load()
    # 存储结果
    # data_result = []
    # for name in top_three_names:
    #     model = train_and_eval(name, X_train, Y_train, X_test, Y_test)
    #     # 进行训练
    #     start_time = int(datetime.now().timestamp())
    #     model.train()
    #     middle_time = int(datetime.now().timestamp())
    #     train_time = middle_time - start_time
    #     print(f"训练花费了{train_time}")
    #     # 进行预测
    #     pred_test = model.test()
    #     end_time = int(datetime.now().timestamp())
    #
    #     test_time = end_time - middle_time
    #     print(f"测试花费了{test_time}")
    #     # result_temp, data_list = main(X_train, Y_train, X_test, Y_test, name)
    #     if isinstance(pred_test, np.ndarray):
    #         data_result.append(pred_test)
    #     elif isinstance(pred_test, torch.Tensor):
    #         data_result.append(pred_test.numpy())
    # mean_data_result = np.mean(data_result, axis=0)
    # print("使用第一个计算：")
    # # evaluation(X_test, X_train, Y_test, data_result[0][-len(Y_test):, -4:], Y_train)
    # evaluation(X_test, X_train, Y_test, data_result[0], Y_train)
    # print("使用第二个计算：")
    # evaluation(X_test, X_train, Y_test, data_result[1], Y_train)
    # print("使用第三个计算：")
    # evaluation(X_test, X_train, Y_test, data_result[2], Y_train)
    # print("使用三个平均值进行计算：")
    # evaluation(X_test, X_train, Y_test, mean_data_result, Y_train)
    # print("哈哈哈哈")


def get_best_model(file_path):
    # 读取数据
    with open(file_path, 'r') as f:
        data = json.load(f, object_hook=custom_decoder)
    sorted_result = sorted(data, key=lambda x: x['total_data']['r2'], reverse=True)
    # 最佳的模型
    top_three_names = [item['name'] for item in sorted_result[:1]]
    print(f"最佳的模型是：{top_three_names}")
    print(sorted_result[:1])
    return top_three_names


if __name__ == '__main__':

    X_train, Y_train, X_test, Y_test = data_load()
    print(f"训练数据{len(X_train)}条, 测试数据{len(X_test)}条")

    # 建立模型
    # models = ['LinearRegressor', 'KNeighborsRegressor', 'RandomForestRegressor', 'DecisionTreeRegressor',
    #           'SupportVectorMachine', 'MultilayerPerceptronRegressor', 'ExtraTreeRegressor',
    #           'XGBoostRegressor', 'LGBMRegressor', 'GBDTRegressor', 'CATBoostRegressor',
    #           'LSTMRegressor', 'AdaBoostRegressor', 'BaggingRegressor', 'LASSORegressor']
    models = ['LinearRegressor', 'KNeighborsRegressor', 'RandomForestRegressor', 'DecisionTreeRegressor',
              'SupportVectorMachine', 'MultilayerPerceptronRegressor', 'ExtraTreeRegressor',
              'XGBoostRegressor', 'LGBMRegressor', 'GBDTRegressor', 'CATBoostRegressor',
              'AdaBoostRegressor', 'BaggingRegressor', 'LASSORegressor']
    result = []
    # 数据名称
    # header = 'x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,y1,y2,y3,y4,y5,y6'
    # header = 'x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,' \
    #          'x30,x31,y1,y2,y3,y4,y5,y6'
    header = 'x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,' \
             'x30,y1,y2,y3,y4'
    for name in models:
        print(f"当前的模型是：{name}")
        # name = 'LogisticRegressor'
        print("哈哈哈哈")
        # 在root目录下为每个方法创建文件夹，对数据进行存储
        dir_path = os.path.join(ROOT_PATH, 'result_change', parent_dir, DATASET_NAME.split('.')[0], name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        (result_temp, (data_train, data_test)), model = main(X_train, Y_train, X_test, Y_test, name)
        result.append(result_temp)
        # 将模型结构保存为 pkl 文件
        model_path = os.path.join(dir_path, f'{name}.pth')
        torch.save(model, model_path)
        # 预测结果保存为 CSV 文件
        save_path_train = os.path.join(dir_path, f'{name}_data_train.csv')
        np.savetxt(save_path_train, data_train, delimiter=',', fmt='%.5f', header=header)
        save_path_test = os.path.join(dir_path, f'{name}_data_test.csv')
        np.savetxt(save_path_test, data_test, delimiter=',', fmt='%.5f', header=header)

    file_path = os.path.join(ROOT_PATH, 'result_change', parent_dir, DATASET_NAME.split('.')[0], 'result.json')
    if not os.path.exists(os.path.join(ROOT_PATH, 'result_change', parent_dir)):
        os.makedirs(os.path.join(ROOT_PATH, 'result_change', parent_dir))
    # 将所有的方法的结果保存到结果当中
    with open(file_path, 'w') as f:
        json.dump(result, f, indent=2, default=custom_encoder)

    best_model_name = get_best_model(file_path)[0]
    # 使用最佳的模型重新进行计算
    best_model = train_and_eval(best_model_name, X_train, Y_train, X_test, Y_test)
    best_model.train()

    temp_y = best_model.predict(X_test)
    print("哈哈哈哈")

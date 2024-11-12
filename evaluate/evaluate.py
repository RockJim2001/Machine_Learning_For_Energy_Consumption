#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：ML_project 
@Product_name ：PyCharm
@File ：evaluate.py
@Author ：RockJim
@Date ：2023/7/26 20:26
@Description ：对预测结果进行评估
@Version ：1.0 
'''
import numpy
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, explained_variance_score, r2_score, \
    median_absolute_error, mean_squared_log_error, mean_squared_error
from sklearn.metrics._regression import _check_reg_targets

from config.log_config import log

logger = log().getLogger("模型性能评估")


def evaluate_prediction(y_test, y_predict):
    """
        使用一定的指标来对预测结果进行计算
    :param y_test: 测试集的真值
    :param y_predict: 测试集上的预测值
    :return:
    """
    """
        均方误差（Mean Squared Error，MSE）：MSE是最常用的回归模型评估指标之一。
        它计算预测值与真实值之间的平方差的平均值，用来衡量预测值与真实值之间的差异。
        MSE越小越好，表示预测结果越接近真实值。
    """
    mse = mean_squared_error(y_test, y_predict)
    """
        平均绝对误差（Mean Absolute Error，MAE）：MAE计算预测值与真实值之间的绝对差的平均值，用来衡量预测值与真实值之间的平均误差。
        MAE越小越好。
    """
    mae = mean_absolute_error(y_test, y_predict)

    """
        可释方差（Explained Variance Score）：可释方差度量预测值与真实值之间的方差占比。
        可释方差的取值范围在0到1之间，越接近1表示模型预测效果越好，越接近0表示模型预测效果较差。
    """
    evs = explained_variance_score(y_test, y_predict)

    """
        MedAE计算预测值与真实值之间差异的绝对值的中值。与MAE类似，MedAE对异常值不敏感。
    """
    medAE = median_absolute_error(y_test, y_predict)
    """
        MSLE计算预测值和真实值的对数差的平方的平均值。它对于数据中较大的误差更敏感，适用于具有指数增长特性的数据。
    """
    msle = 0
    # msle = mean_squared_log_error(y_test, y_predict)
    """
        RMSE是MSE的平方根，它可以将误差值的单位转换回与原始数据相同的单位。RMSE在计算时对误差值进行了放大，使得较大误差对模型性能影响更为显著。
    """
    """
        RAE计算预测值与真实值之间差异的绝对值的平均值，然后再除以真实值的平均值，以便将误差归一化
    """
    rae = relative_absolute_error(y_test, y_predict)
    """
        RSE计算预测值与真实值之间差异的平方的平均值，然后再除以真实值的平均值，以便将误差归一化
    """
    rse = relative_squared_error(y_test, y_predict)
    """
        RMSE是MSE的平方根，它可以将误差值的单位转换回与原始数据相同的单位。RMSE在计算时对误差值进行了放大，使得较大误差对模型性能影响更为显著。
    """
    rmse = root_mean_squared_error(y_test, y_predict)
    """
            决定系数（Coefficient of Determination，R-squared）：决定系数用来评估回归模型的拟合程度，表示模型解释的数据方差所占比例。
            R-squared的取值范围在0到1之间，越接近1表示模型拟合效果越好，越接近0表示模型拟合效果较差。
    """
    r2 = r2_score(y_test, y_predict)

    logger.info("MAE：{}".format(mae))
    logger.info("MSE：{}".format(mse))
    logger.info("EVS：{}".format(evs))
    logger.info("MedAE:{}".format(medAE))
    logger.info("MSLE:{}".format(msle))
    logger.info("RAE:{}".format(rae))
    logger.info("RSE:{}".format(rse))
    logger.info("RMSE:{}".format(rmse))
    logger.info("R2：{}".format(r2))
    return mae, mse, evs, medAE, msle, rae, rse, rmse, r2


def root_mean_squared_error(y_true, y_pred):
    """
        RMSE是MSE的平方根，它可以将误差值的单位转换回与原始数据相同的单位。RMSE在计算时对误差值进行了放大，使得较大误差对模型性能影响更为显著。
    :param y_true:
    :param y_pred:
    :return:
    """
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def relative_absolute_error(y_true, y_pred, multioutput="uniform_average"):
    """
        RAE计算预测值与真实值之间差异的绝对值的平均值，然后再除以真实值的平均值，以便将误差归一化
    :param multioutput:
    :param y_true:
    :param y_pred:
    :return:
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput
    )
    absolute_errors = np.abs(y_true - y_pred)
    mean_true_value = np.mean(y_true)
    if np.sum(np.abs(y_true - mean_true_value)) == 0.0:
        return 1.00000
    rae = np.sum(absolute_errors) / np.sum(np.abs(y_true - mean_true_value))
    return rae


def relative_squared_error(y_true, y_pred, multioutput="uniform_average"):
    """
        计算相对平方误差（RSE）。
        RSE计算预测值与真实值之间差异的平方的平均值，然后再除以真实值的平均值，以便将误差归一化
    :param multioutput:
    :param y_true:  真实值的数组。
    :param y_pred:  预测值的数组。
    :return:    rse：float，相对平方误差的值。
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput
    )
    true_mean = np.mean(y_true)
    squared_error_num = np.sum(np.square(y_true - y_pred))
    squared_error_den = np.sum(np.square(y_true - true_mean))
    if squared_error_den == 0.0:
        return -1.0000
    rse = squared_error_num / squared_error_den
    return rse

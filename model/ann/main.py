#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：main.py
@Author ：RockJim
@Date ：2023/8/21 22:14
@Description ：测试的主函数
@Version ：1.0
"""
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from config.log_config import log
from config.config import DATASET_NAME, ROOT_PATH
from dataset.data_load import load_data, data_processing
from evaluate.evaluate import evaluate_prediction
from model.ann.ann import Net
from model.ann.genetic_algorithm import GeneticAlogrithm
from model.nsga.nsga_2 import MultiObjectiveProblem
from config.config import log_dir
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation

logger = log().getLogger(__name__)


def data_load():
    # 加载数据集
    x_data, y_data = load_data(os.path.join(ROOT_PATH, 'resource', DATASET_NAME))
    # 数据归一化处理
    x_train, x_test, y_train, y_test = data_processing(x_data, y_data)

    # 将这些值转为tensor格式
    X_train = torch.tensor(x_train, dtype=torch.float32)
    Y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(x_test, dtype=torch.float32)
    Y_test = torch.tensor(y_test, dtype=torch.float32)
    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    input_features = 12  # 单个输入样本的变量个数。
    output_features = 6  # 单个输出的变量个数。
    hidden_layer_range = [2, 128]
    learning_rate = 0.1
    epochs = 2
    population_size = 20
    num_generations = 1
    mutation_rate = 0.1
    # 加载数据
    X_train, Y_train, X_test, Y_test = data_load()



    # ga = GeneticAlogrithm(population_size, num_generations, mutation_rate, X_train, Y_train,
    #                       X_test, Y_test, input_features, hidden_layer_range, output_features)
    # # 选择最佳模型结构
    # best_hidden_dims = ga.forward()
    # best_hidden_dims = [2, 16]
    # logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # logger.info("最佳隐藏层参数为：{}".format(best_hidden_dims))
    # writer = SummaryWriter(log_dir=log_dir)
    # # 创建模型
    # best_model = Net(input_features, best_hidden_dims, output_features)
    # logger.info("最佳模型结构如下：\t\n{}".format(best_model))
    #
    # mseloss = nn.MSELoss()  # 均方误差对象作为损失函数
    # sgd = optim.SGD(best_model.parameters(), lr=learning_rate)  # 用随机梯度下降对象作为优化函数。
    # for epoch in range(epochs * 100):
    #     y_predict = best_model(X_train)  # 前向传播，自动调用Module的forward方法
    #     loss = mseloss(y_predict, Y_train)  # 计算损失函数
    #     loss.backward()  # 反向传播
    #     sgd.step()  # 更新参数(weight和bias)
    #     sgd.zero_grad()  # 清零梯度数据
    #     logger.info(
    #         f"epoch:{epoch}, loss:{loss:.6f}"
    #     )
    #     writer.add_scalar('Loss/train', loss.item(), epoch)
    #
    # writer.add_graph(best_model, X_train[0])
    # writer.close()
    # logger.info("在测试集上进行评估………………")
    # best_model.eval()
    # with torch.no_grad():
    #     predictions = best_model(X_test)
    # pred = predictions.detach().numpy()
    #
    # # 计算评估指标
    # evaluate_prediction(Y_test.detach().numpy(), pred)
    #
    # # 多目标优化参数的约束条件
    # constr_list = [np.array([0.5, 0.5, 0.5, 3.0, 0.20, 0.20,
    #                          0.20, 0.20, 0.2, 0.2, 0.2, 0.2]),
    #                np.array([1.0, 1.5, 1.5, 4.0, 0.40, 0.80,
    #                          0.40, 0.40, 0.3, 0.3, 0.4, 0.4])]
    #
    # # 初始化多目标优化问题
    # problem = MultiObjectiveProblem(best_model, input_features, output_features, constr_list)
    #
    # # 定义优化算法的参数和配置
    # algorithm = NSGA2(
    #     pop_size=100,  # 种群个数
    #     sampling=FloatRandomSampling(),
    #     crossover=SBX(prob=0.9, eta=15),
    #     mutation=PolynomialMutation(eta=20),
    #     eliminate_duplicates=True
    # )
    #
    # # 执行多目标优化
    # res = minimize(
    #     problem,
    #     algorithm,
    #     ('n_gen', 100),  # 迭代次数
    #     seed=1,
    #     verbose=True
    # )
    #
    # # # 输出优化结果
    # # print("Final population:")
    # # for solution in res.X:
    # #     print(solution)
    #
    # # 获得优化结果
    # optimal_solutions = res.X
    # optimal_objectives = res.F
    #
    # # print("Optimal Solutions:")
    # # print(optimal_solutions)
    # # print("Optimal Objectives:")
    # # print(optimal_objectives)
    # # 创建雷达图
    # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    #
    # # 创建角度
    # theta = np.linspace(0, 2 * np.pi, problem.n_obj, endpoint=False)
    #
    # # 绘制雷达图
    # for i in range(len(optimal_objectives)):
    #     ax.plot(theta, optimal_objectives[i])
    #
    # # 设置角度刻度标签
    # ax.set_xticks(theta)
    # ax.set_xticklabels(["F1", "F2", "F3", "F4", "F5", "F6"])
    #
    # # 添加网格线
    # ax.grid(True)
    #
    # # 添加标题和图例
    # plt.title("Pareto Front")
    # plt.legend(["Solution " + str(i + 1) for i in range(len(optimal_solutions))])
    #
    # # 设置图例显示的位置
    # plt.legend(loc='lower right')
    #
    # # 保存图片
    # plt.savefig('my_plot.png', dpi=300, bbox_inches='tight', format='jpg')

import os

import geatpy as ea
import numpy as np
import pandas as pd
from MOEAs.problems.problem_AdaBoost import MyProblem as MyProblem_AdaBoost
from MOEAs.problems.problem_Bagging import MyProblem as MyProblem_Bagging
from MOEAs.problems.problem_CATBoost import MyProblem as MyProblem_CATBoost
from MOEAs.problems.problem_DecisionTree import MyProblem as MyProblem_DecisionTree
from MOEAs.problems.problem_ExtraTree import MyProblem as MyProblem_ExtraTree
from MOEAs.problems.problem_GBDT import MyProblem as MyProblem_GBDT
from MOEAs.problems.problem_KNeighbors import MyProblem as MyProblem_KNeighbors
from MOEAs.problems.problem_LASSO import MyProblem as MyProblem_LASSO
from MOEAs.problems.problem_LGBM import MyProblem as MyProblem_LGBM
from MOEAs.problems.problem_Linear import MyProblem as MyProblem_Linear
from MOEAs.problems.problem_MultilayerPerceptron import MyProblem as MyProblem_MultilayerPerceptron
from MOEAs.problems.problem_RandomForest import MyProblem as MyProblem_RandomForest
from MOEAs.problems.problem_LSTM import MyProblem as MyProblem_LSTM
from MOEAs.problems.problem_SupportVectorMachine import MyProblem as MyProblem_SupportVectorMachine # ok
from MOEAs.problems.problem_XGBoost import MyProblem as MyProblem_XGBoost # ok


import matplotlib.pyplot as plt
import warnings

from config.config import ROOT_PATH, DATASET_NAME, parent_dir

warnings.filterwarnings('ignore')

"""
    主函数：通过调用机器学习模型，用于评估进化算法的适应度值
"""

# 这里选择需要执行的机器学习方法
'''
'LinearRegressor', 'KNeighborsRegressor', 'RandomForestRegressor', 'DecisionTreeRegressor',
'SupportVectorMachine', 'MultilayerPerceptronRegressor', 'ExtraTreeRegressor',
'XGBoostRegressor', 'LGBMRegressor', 'GBDTRegressor', 'CATBoostRegressor',
'LSTMRegressor', 'AdaBoostRegressor', 'BaggingRegressor', 'LASSORegressor'
'''

ML_method = 'XGBoostRegressor'

# 建立模型
if ML_method == 'LinearRegressor':
    current_problem = MyProblem_Linear() # OK 预计30s
if ML_method == 'KNeighborsRegressor':
    current_problem = MyProblem_KNeighbors()
if ML_method == 'RandomForestRegressor':
    current_problem = MyProblem_RandomForest() # No
if ML_method == 'DecisionTreeRegressor':
    current_problem = MyProblem_DecisionTree() # No
if ML_method == 'SupportVectorMachine':
    current_problem = MyProblem_SupportVectorMachine() # Ok
if ML_method == 'MultilayerPerceptronRegressor':
    current_problem = MyProblem_MultilayerPerceptron()
if ML_method == 'ExtraTreeRegressor':
    current_problem = MyProblem_ExtraTree()
if ML_method == 'XGBoostRegressor':
    current_problem = MyProblem_XGBoost()
if ML_method == 'LGBMRegressor':
    current_problem = MyProblem_LGBM()
if ML_method == 'GBDTRegressor':
    current_problem = MyProblem_GBDT()
if ML_method == 'CATBoostRegressor':
    current_problem = MyProblem_CATBoost() # No
if ML_method == 'LSTMRegressor':
    current_problem = MyProblem_LSTM()
if ML_method == 'AdaBoostRegressor':
    current_problem = MyProblem_AdaBoost() # NO
if ML_method == 'BaggingRegressor':
    current_problem = MyProblem_Bagging() # No
if ML_method == 'LASSORegressor':
    current_problem = MyProblem_LASSO() # Ok


if __name__ == '__main__':
    """===============================实例化问题对象============================"""
    problem = current_problem     # 生成问题对象
    """==================================种群设置==============================="""
    Encoding = 'RI'           # 编码方式
    NIND = 50               # 种群规模 # 50,100
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders,
                      [10] * len(problem.varTypes))    # 创建区域描述器

    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    myAlgorithm = ea.moea_NSGA2_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.mutOper.Pm = 0.1    # 修改变异算子的变异概率
    myAlgorithm.recOper.XOVR = 0.8  # 修改交叉算子的交叉概率
    myAlgorithm.MAXGEN = 100      # 最大进化代数 # 50-100
    myAlgorithm.logTras = 5         # 设置每多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = True     # 设置是否打印输出日志信息
    myAlgorithm.drawing = 1        # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """==========================调用算法模板进行种群进化=========================
    调用run执行算法模板，得到帕累托最优解集NDSet以及最后一代种群。NDSet是一个种群类Population的对象。
    NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。
    详见Population.py中关于种群类的定义。
    """
    [NDSet, population] = myAlgorithm.run()  # 执行算法模板，得到非支配种群以及最后一代种群
    # 对目标进行筛选
    Variable = []
    for i in range(len(NDSet.Phen.tolist())):
        v = NDSet.Phen.tolist()[i]
        Variable.append(v)
    NDSet.Phen = np.array(Variable)
    NDSet.sizes = len(Variable)

    problem.aimFunc(NDSet)
    df1 = pd.DataFrame(NDSet.Phen)
    df2 = pd.DataFrame(NDSet.ObjV)
    dir_path = os.path.join(ROOT_PATH, 'result_change', parent_dir, DATASET_NAME.split('.')[0], 'NSGA2')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    df1.to_csv(os.path.join(dir_path, f'Variable_NSGA2_{ML_method}.csv'),header=None, index=None)
    df2.to_csv(os.path.join(dir_path, f'Objective_NSGA2_{ML_method}.csv'),header=None, index=None)


    # print(dir(NDSet))
    print(population.getInfo)

    # NDSet.save()  # 把非支配种群的信息保存到文件中
    """==================================输出结果=============================="""
    print('用时：%s 秒' % myAlgorithm.passTime)
    print('非支配个体数：%d 个' % NDSet.sizes) if NDSet.sizes != 0 else print('没有找到可行解！')

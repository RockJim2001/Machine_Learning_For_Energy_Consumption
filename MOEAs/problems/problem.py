from math import pi, sin, sqrt
import os
import joblib
import torch
from config.config import log_dir, ROOT_PATH
import numpy as np

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

"""
.. module:: UF
   :platform: Unix, Windows
   :synopsis: Problems of the CEC2009 multi-objective competition

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""

# 读取训练好的机器学习模型
def load_model(path, x):
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

    return restored_data_pred[0][0],restored_data_pred[0][1],restored_data_pred[0][2],restored_data_pred[0][3]

class Current_problem(FloatProblem):
    """ Problem Current_problem.

    .. note:: Unconstrained problem. The default number of variables is 30.
    """

    def __init__(self, number_of_variables: int = 9, ML_method = ''):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        super(Current_problem, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 4
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(self.number_of_objectives)]

        self.lower_bound = [10.03825948, 0.700973746, 4.004031902, 0.700496774, 10.02248555, 0.700057589, 23.01352777, 20.09759722, 17.0]
        self.upper_bound = [15.95406531, 0.996256486, 8.977456514, 0.997588131, 14.96769559, 0.999917747, 25.97938143, 29.9062272, 19.0]

        self.problem_name = ''
        self.ML_method = ML_method

    # moea_MOEAD_archive_templet

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        x = solution.variables

        # 对x进行归一化操作，使其在0-1之间
        X = np.array(x).reshape(1, -1)
        # 计算目标函数值
        f1, f2, f3, f4 = 0, 0, 0, 0

        path = r'C:\Users\38952\Desktop\ML_project\result/XGBoostRegressor/'
        if self.ML_method == 'LinearRegressor':
            f1, f2, f3, f4 = load_model(path + 'LinearRegressor.pth', X)
        if self.ML_method == 'KNeighborsRegressor':
            self.current_problem = load_model(path + 'KNeighborsRegressor.pth', X)
            f1, f2, f3, f4 = load_model(path + 'KNeighborsRegressor.pth', X)
        if self.ML_method == 'RandomForestRegressor':
            self.current_problem = load_model(path + 'RandomForestRegressor.pth', X)
            f1, f2, f3, f4 = load_model(path + 'RandomForestRegressor.pth', X)
        if self.ML_method == 'DecisionTreeRegressor':
            self.current_problem = load_model(path + 'DecisionTreeRegressor.pth', X)
            f1, f2, f3, f4 = load_model(path + 'DecisionTreeRegressor.pth', X)
        if self.ML_method == 'SupportVectorMachine':
            self.current_problem = load_model(path + 'SupportVectorMachine.pth', X)
            f1, f2, f3, f4 = load_model(path + 'SupportVectorMachine.pth', X)
        if self.ML_method == 'MultilayerPerceptronRegressor':
            self.current_problem = load_model(path + 'MultilayerPerceptronRegressor.pth', X)
            f1, f2, f3, f4 = load_model(path + 'MultilayerPerceptronRegressor.pth', X)
        if self.ML_method == 'ExtraTreeRegressor':
            self.current_problem = load_model(path + 'ExtraTreeRegressor.pth', X)
            f1, f2, f3, f4 = load_model(path + 'ExtraTreeRegressor.pth', X)
        if self.ML_method == 'XGBoostRegressor':
            self.current_problem = load_model(path + 'XGBoostRegressor.pth', X)
            f1, f2, f3, f4 = load_model(path + 'XGBoostRegressor.pth', X)
        if self.ML_method == 'LGBMRegressor':
            self.current_problem = load_model(path + 'LGBMRegressor.pth', X)
            f1, f2, f3, f4 = load_model(path + 'LGBMRegressor.pth', X)
        if self.ML_method == 'GBDTRegressor':
            self.current_problem = load_model(path + 'GBDTRegressor.pth', X)
            f1, f2, f3, f4 = load_model(path + 'GBDTRegressor.pth', X)
        if self.ML_method == 'CATBoostRegressor':
            self.current_problem = load_model(path + 'CATBoostRegressor.pth', X)
            f1, f2, f3, f4 = load_model(path + 'CATBoostRegressor.pth', X)
        if self.ML_method == 'LSTMRegressor':
            self.current_problem = load_model(path + 'LSTMRegressor.pth', X)
            f1, f2, f3, f4 = load_model(path + 'LSTMRegressor.pth', X)
        if self.ML_method == 'AdaBoostRegressor':
            self.current_problem = load_model(path + 'AdaBoostRegressor.pth', X)
            f1, f2, f3, f4 = load_model(path + 'AdaBoostRegressor.pth', X)
        if self.ML_method == 'BaggingRegressor':
            self.current_problem = load_model(path + 'BaggingRegressor.pth', X)
            f1, f2, f3, f4 = load_model(path + 'BaggingRegressor.pth', X)
        if self.ML_method == 'LASSORegressor':
            self.current_problem = load_model(path + 'LASSORegressor.pth', X)
            f1, f2, f3, f4 = load_model(path + 'LASSORegressor.pth', X)


        solution.objectives[0] = f1
        solution.objectives[1] = f2
        solution.objectives[2] = f3
        solution.objectives[3] = f4

        return solution

    def get_name(self):
        return self.ML_method


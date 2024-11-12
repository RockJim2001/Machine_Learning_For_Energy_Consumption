from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
import copy
import pandas as pd
import numpy as np

class Profit(FloatProblem):
    """ 投资组合优化问题，目标函数
        fun1: 最大收益(加负号)
        fun2: 最小风险
    """

    def __init__(self, number_of_variables: int = 9, number_of_objectives=2):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        super(Profit, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # 总投资
        money = 1000000
        # 买入价格
        price = [None, 34.67, 45.16, 192.81, 42.27, 93.07, 49.41, 26.91, 70.17]
        # 收益率
        interest_rate = [0.0515, 0.0369, 0.1160, 0.0299, 0.0249, 0.0179, 0.0470, 0.0343, 0.0334]

        # 在评估个体之前要进行归一化操作
        X = copy.copy(solution.variables)
        sum = 0
        for i in range(self.number_of_variables):
            sum += X[i]
        for i in range(self.number_of_variables): # 归一化
            solution.variables[i] = X[i] / sum

        temp = 0
        for i in range(1,self.number_of_variables): # 股票投资手数应为整数
            solution.variables[i] = (((solution.variables[i]*money)//price[i])*price[i]) / money
            temp += solution.variables[i]
        solution.variables[0] = 1 - temp # 投资股票剩余的前用于投资国债(满足sum(x[k])==1, 钱全部由于投资)


        # 协方差矩阵
        data = pd.read_excel("00.xls", header=None)
        cov_matric = np.array(data)
        X = np.array(X[1:]).reshape((1, len(X[1:])))
        val = X.dot(cov_matric).dot(X.T)[0][0]

        # 收入
        income = 0
        for i in range(self.number_of_variables):
            income += money * solution.variables[i] * interest_rate[i]

        C0 = 0  # 股票交易成本
        for i in range(1,self.number_of_variables):
            C0 += 5  # 委托费
            C0 += max(5,100*0.003*((solution.variables[i]*money)//price[i])*price[i]) # 券商交易佣金
            C0 += max(1,100*0.002*((solution.variables[i]*money)//price[i])*price[i]) # 过户费
            C0 += 0.0012996*((solution.variables[i]*money)//price[i])*price[i]

        C1 = 0.01*solution.variables[i]*money


        solution.objectives[0] = -income/money  # 收益率
        solution.objectives[1] = val            # 投资风险

        return solution

    def get_name(self):
        return 'Profit'

# if __name__ == "__main__":
#     data = pd.read_excel("00.xls", header=None)
#     data = np.array(data)
#     interest_rate = [0.0515, 0.0369, 0.0160, 0.0299, 0.0249, 0.0179, 0.0470, 0.0343]
#     X = np.array(interest_rate).reshape((1,8))
#     val = X.dot(data).dot(X.T)[0][0]
#     print(val)

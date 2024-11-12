from abc import ABC, abstractmethod

from jmetal.util.point import IdealPoint, IdealPointMax, NadirPoint

"""
.. module:: aggregative_function
   :platform: Unix, Windows
   :synopsis: Implementation of aggregative (scalarizing) functions.
.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class AggregativeFunction(ABC):

    @abstractmethod
    def compute(self, vector: [], weight_vector: []) -> float:
        pass

    @abstractmethod
    def update(self, vector: []) -> None:
        pass

    @abstractmethod
    def get_ideal_point(self) -> list:
        pass

# 加权和
class WeightedSum(AggregativeFunction):

    def compute(self, vector: [], weight_vector: []) -> float:
        # 根据提供的函数对指定序列做映射， 返回目标值和权重向量乘积的和
        return sum(map(lambda x, y: x * y, vector, weight_vector))

    def update(self, vector: []) -> None:
        pass

    def get_ideal_point(self) -> list:
        pass


# 契比雪夫
class Tschebycheff(AggregativeFunction):

    def __init__(self, dimension: int):
        self.ideal_point = IdealPoint(dimension)

    def compute(self, vector: [], weight_vector: []) -> float:
        max_fun = -1.0e+30
        cup = []
        for i in range(len(vector)):  # 目标函数的个数
            diff = abs(vector[i] - self.ideal_point.point[i])

            if weight_vector[i] == 0:
                feval = 0.0001 * diff
            else:
                feval = diff * weight_vector[i]
            cup.append(feval)
            # 选取m个目标中 与权重向量乘积的最大值
            if feval > max_fun:
                max_fun = feval
        #print(cup)
        return max_fun

    def update(self, vector: []) -> None:
        self.ideal_point.update(vector)

    def update_z_nad(self, vector: []) -> None:
        '''记录每代种群在各个目标上的最大值'''
        pass

    def get_ideal_point(self) -> list:
        '''获取当前理想点'''
        return self.ideal_point.get_point()


# 契比雪夫
class ASF(AggregativeFunction):

    def __init__(self, dimension: int):
        self.ideal_point = IdealPoint(dimension)
        self.nadir_point = NadirPoint(dimension)

    def compute(self, vector: [], weight_vector: []) -> float:
        max_fun = -1.0e+30
        for i in range(len(vector)):  # 目标函数的个数
            diff = abs(vector[i] - self.ideal_point.point[i])/(self.nadir_point.point[i] - self.ideal_point.point[i])

            if weight_vector[i] == 0:
                feval = diff / 0.0001
            else:
                feval = diff / weight_vector[i]
            # 选取m个目标中 与权重向量乘积的最大值
            if feval > max_fun:
                max_fun = feval
        return max_fun

    def update(self, vector: []) -> None:
        self.ideal_point.update(vector)


    def get_ideal_point(self) -> list:
        '''获取当前理想点'''
        return self.ideal_point.get_point()

    def get_nadir_point(self) -> list:
        '''获取当前理想点'''
        return self.nadir_point.get_point()


# 带标准化的切比雪夫函数
class Tschebycheff_normal(AggregativeFunction):

    def __init__(self, dimension: int):
        self.ideal_point = IdealPoint(dimension)
        self.nadir_point = NadirPoint(dimension)

    def compute(self, vector: [], weight_vector: []) -> float:
        '''z_nad: 当前种群中每个目标的最大值（m个）'''
        max_fun = -1.0e+30
        for i in range(len(vector)):  # 目标函数的个数
            diff = abs(vector[i] - self.ideal_point.point[i])/(self.nadir_point.point[i] - self.ideal_point.point[i])
            if weight_vector[i] == 0:
                feval = 0.0001 * diff
            else:
                feval = weight_vector[i] * diff
            # 选取m个目标中 与权重向量乘积的最大值
            if feval > max_fun:
                max_fun = feval
        #print(self.ideal_point.point)
        return max_fun

    def update(self, vector: []) -> None:
        self.ideal_point.update(vector)

    def get_ideal_point(self) -> list:
        '''获取当前理想点'''
        return self.ideal_point.get_point()

    def get_nadir_point(self) -> list:
        '''获取当前理想点'''
        return self.nadir_point.get_point()

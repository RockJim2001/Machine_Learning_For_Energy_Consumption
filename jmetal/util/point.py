from abc import ABC, abstractmethod

"""
.. module:: point  理想点

"""


class Point(ABC):

    @abstractmethod
    def update(self, vector: []) -> None:
        pass

    @abstractmethod
    def get_point(self) -> list:
        pass


class IdealPoint(Point):

    def __init__(self, dimension: int):
        self.point = dimension * [float("inf")]

    # 更新理想点，每个目标的最小值
    def update(self, vector: []) -> None:
        # self.point, vector  原始点和新生成的点
        self.point = [y if x > y else x for x, y in zip(self.point, vector)]

    def get_point(self) -> list:
        return self.point


class IdealPointMax(Point):
    def __init__(self, dimension: int):
        self.point = dimension * [-float("inf")]

    # 更新理想点，每个目标的最大值
    def update(self, vector: []):
        self.point = [y if x < y else x for x, y in zip(self.point, vector)]

    def get_point(self) -> list:
        return self.point


class NadirPoint(Point):

    def __init__(self, dimension: int):
        self.point = dimension * [float("-inf")]

    # 更新天底点，每个目标的最小值
    def update(self, vector: []) -> None:
        # self.point, vector  原始点和新生成的点
        self.point = [y if y > x else x for x, y in zip(self.point, vector)]

    def get_point(self) -> list:
        return self.point

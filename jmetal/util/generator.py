import copy
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List

from jmetal.core.problem import Problem
from jmetal.core.solution import Solution

R = TypeVar('R')

"""
.. module:: generator
   :synopsis: Population generators implementation.
   相应的算法在生成个体解时，通过Generator来调用problem的create_solution()
"""


class Generator(Generic[R], ABC):

    @abstractmethod
    def new(self, problem: Problem) -> R:
        pass


class RandomGenerator(Generator):
    '''随机初始化方式'''
    def new(self, problem: Problem):
        return problem.create_solution()

class HeuristicGenerator(Generator):
    '''随机初始化方式'''
    def new(self, problem: Problem):
        return problem.create_solution()


class InjectorGenerator(Generator):

    def __init__(self, solutions: List[Solution]):
        super(InjectorGenerator, self).__init__()
        self.population = []

        for solution in solutions:
            self.population.append(copy.deepcopy(solution))

    def new(self, problem: Problem):
        if len(self.population) > 0:
            # If we have more solutions to inject, return one from the list
            return self.population.pop()
        else:
            # Otherwise generate a new solution
            solution = problem.create_solution()

        return solution

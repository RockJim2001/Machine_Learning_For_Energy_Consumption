from math import sqrt, pow, sin, pi, cos
import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class Problem1(FloatProblem):

    def __init__(self, number_of_variables: int=2):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super().__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['x', 'y']

        self.lower_bound = [500, 10]
        self.upper_bound = [1500, 35]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        V = 53470.0
        O = 0.2
        alpha = 71 * np.pi / 180
        a = 42.22
        H = 30.0
        T1 = 0.17
        x1 = solution.variables[0]
        x2 = solution.variables[1]
        b = (2.6725 + 0.001082 * x1 - 0.09283 * x2 + 0.001292 * x2 * x2 - 0.00001 * x1 * x2) ** 2
        h = (1.1942 + 0.000297 * x1 - 0.0055 * x2 + 0.000262 * x2 * x2 - 0.000008 * x1 * x2) ** 2

        solution.objectives[0] = V / h / b / (1 - O) / x2 + a * T1 * H / h / b / (1 - O)
        solution.objectives[1] = (h ** 2) / 2 / np.tan(alpha)

        return solution


    def get_name(self):
        return 'Problem1'
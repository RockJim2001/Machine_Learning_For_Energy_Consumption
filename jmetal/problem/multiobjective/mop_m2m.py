from math import pi, cos, sin, sqrt, exp

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

"""
.. module:: MOP01-MOP07 
   :platform: Unix, Windows
   :synopsis: MOP(M2M) problem family of multi-objective problems.
"""

class MOP1(FloatProblem):
    """ Problem MOP1. Continuous problem having a flat Pareto front

    .. note:: Unconstrained problem. The default number of variables and objectives are, respectively, 10 and 2.
    """

    def __init__(self, number_of_variables: int = 10, number_of_objectives=2):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        super().__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        t = self.eval_t(solution)
        g = self.eval_g(solution, t)

        solution.objectives[0] = (1+g)*solution.variables[0]
        solution.objectives[1] = (1+g)*(1-sqrt(solution.variables[0]))

        return solution

    def eval_t(self, solution: FloatSolution) -> list:
        t = [1 for _ in range(self.number_of_variables-1)]

        for i in range(self.number_of_variables-1):
            t[i] = solution.variables[i+1] - sin(0.5*pi*solution.variables[0])

        return t

    def eval_g(self, solution: FloatSolution, t: list) -> float:

        g = sin(pi*solution.variables[0]) * sum(list(map(lambda x: -0.9*(x**2) + abs(x)**0.6, t))[1:])

        return g

    def get_name(self):
        return 'MOP1'

class MOP2(MOP1):
    """ Problem MOP2. Continuous problem having a flat Pareto front

    .. note:: Unconstrained problem. The default number of variables and objectives are, respectively, 10 and 2.
    """
    def __init__(self, number_of_variables: int = 10, number_of_objectives=2):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        super().__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        t = self.eval_t(solution)
        g = self.eval_g(solution, t)

        solution.objectives[0] = (1+g)*solution.variables[0]
        solution.objectives[1] = (1+g)*(1-solution.variables[0]**2)

        return solution

    def eval_g(self, solution: FloatSolution, t: list) -> float:

        g = 10*sin(pi*solution.variables[0]) * sum(list(map(lambda x: abs(x)/(1+exp(5*abs(x))), t))[1:])

        return g

    def get_name(self):
        return 'MOP2'

class MOP3(MOP1):
    """ Problem MOP3. Continuous problem having a flat Pareto front

    .. note:: Unconstrained problem. The default number of variables and objectives are, respectively, 10 and 2.
    """
    def __init__(self, number_of_variables: int = 10, number_of_objectives=2):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        super().__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        t = self.eval_t(solution)
        g = self.eval_g(solution, t)

        solution.objectives[0] = (1+g)*cos(0.5*pi*solution.variables[0])
        solution.objectives[1] = (1+g)*sin(0.5*pi*solution.variables[0])

        return solution

    def eval_g(self, solution: FloatSolution, t: list) -> float:

        g = 10*sin(0.5*pi*solution.variables[0]) * sum(list(map(lambda x: abs(x)/(1+exp(5*abs(x))), t))[1:])

        return g

    def get_name(self):
        return 'MOP3'

class MOP4(MOP1):
    """ Problem MOP4. Continuous problem having a flat Pareto front

    .. note:: Unconstrained problem. The default number of variables and objectives are, respectively, 10 and 2.
    """
    def __init__(self, number_of_variables: int = 10, number_of_objectives=2):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        super().__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        t = self.eval_t(solution)
        g = self.eval_g(solution, t)

        solution.objectives[0] = (1+g)*solution.variables[0]
        solution.objectives[1] = (1+g)*(1-(solution.variables[0]**0.5) * (cos(2*pi*solution.variables[0])**2))

        return solution

    def eval_g(self, solution: FloatSolution, t: list) -> float:

        g = 10*sin(pi*solution.variables[0]) * sum(list(map(lambda x: abs(x)/(1+exp(5*abs(x))), t))[1:])

        return g

    def get_name(self):
        return 'MOP4'

class MOP5(MOP1):
    """ Problem MOP5. Continuous problem having a flat Pareto front

    .. note:: Unconstrained problem. The default number of variables and objectives are, respectively, 10 and 2.
    """
    def __init__(self, number_of_variables: int = 10, number_of_objectives=2):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        super().__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        t = self.eval_t(solution)
        g = self.eval_g(solution, t)

        solution.objectives[0] = (1+g)*cos(0.5*pi*solution.variables[0])
        solution.objectives[1] = (1+g)*(1-sqrt(solution.variables[0]))

        return solution

    def eval_g(self, solution: FloatSolution, t: list) -> float:

        g = 2*abs(cos(pi*solution.variables[0])) * sum(list(map(lambda x: -0.9*(x**2) + abs(x)**0.6, t))[1:])

        return g

    def get_name(self):
        return 'MOP5'

class MOP6(MOP1):
    """ Problem MOP6. Continuous problem having a flat Pareto front

    .. note:: Unconstrained problem. The default number of variables and objectives are, respectively, 10 and 2.
    """
    def __init__(self, number_of_variables: int = 10, number_of_objectives=3):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        super().__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        t = self.eval_t(solution)
        g = self.eval_g(solution, t)

        solution.objectives[0] = (1+g)*solution.variables[0]*solution.variables[1]
        solution.objectives[1] = (1+g)*solution.variables[0]*(1-solution.variables[1])
        solution.objectives[2] = (1+g)*(1-solution.variables[0])

        return solution

    def eval_t(self, solution: FloatSolution) -> list:
        t = [1 for _ in range(self.number_of_variables-2)]

        for i in range(self.number_of_variables-2):
            t[i] = solution.variables[i+2] - solution.variables[0]*solution.variables[1]

        return t

    def eval_g(self, solution: FloatSolution, t: list) -> float:

        g = 2*sin(pi*solution.variables[0]) * sum(list(map(lambda x: -0.9*(x**2) + abs(x)**0.6, t))[2:])

        return g

    def get_name(self):
        return 'MOP6'

class MOP7(MOP6):
    """ Problem MOP7. Continuous problem having a flat Pareto front

    .. note:: Unconstrained problem. The default number of variables and objectives are, respectively, 10 and 2.
    """
    def __init__(self, number_of_variables: int = 10, number_of_objectives=3):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        super().__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        t = self.eval_t(solution)
        g = self.eval_g(solution, t)

        solution.objectives[0] = (1+g)*cos(0.5*pi*solution.variables[0])*cos(0.5*pi*solution.variables[1])
        solution.objectives[1] = (1+g)*cos(0.5*pi*solution.variables[0])*sin(0.5*pi*solution.variables[1])
        solution.objectives[2] = (1+g)*sin(0.5*pi*solution.variables[0])

        return solution


    def get_name(self):
        return 'MOP7'
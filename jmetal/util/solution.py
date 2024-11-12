import logging
import os
from pathlib import Path
from typing import List

from jmetal.core.solution import FloatSolution, Solution
from jmetal.util.archive import NonDominatedSolutionsArchive, Archive

LOGGER = logging.getLogger('jmetal')


"""
对目标函数值和目标函数进行输出
1. 输出到屏幕
2. 输出到文件
"""


def get_non_dominated_solutions(solutions: List[Solution]) -> List[Solution]:
    archive: Archive = NonDominatedSolutionsArchive()

    for solution in solutions:
        archive.add(solution)

    return archive.solution_list


def read_solutions(filename: str) -> List[FloatSolution]:
    """ Reads a reference front from a file.

    :param filename: File path where the front is located.
    """
    front = []

    if Path(filename).is_file():
        with open(filename) as file:
            for line in file:
                vector = [float(x) for x in line.split()]

                solution = FloatSolution([], [], len(vector))
                solution.objectives = vector

                front.append(solution)
    else:
        LOGGER.warning('Reference front file was not found at {}'.format(filename))

    return front


# 将每一个解写入文件
def print_variables_to_desk_file(solutions, filename: str):
    LOGGER.info('Output file (variables): ' + filename)

    if type(solutions) is not list:
        solutions = [solutions]

    desktop_path = r"C:\Users\Hangyu\Desktop\Data\Var\\"
    full_path = desktop_path + str(filename) + '.txt'
    with open(full_path, 'w') as of:
        for solution in solutions:
            for variables in solution.variables:
                of.write(str(variables) + " ")
            of.write("\n")

# 将每一个解写入文件
def print_variables_to_file(solutions, filename: str):
    LOGGER.info('Output file (variables): ' + filename)

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        pass

    if type(solutions) is not list:
        solutions = [solutions]

    # with open(r'../TB1/data/'+filename, 'w') as of:
    with open(filename, 'w') as of:
    # with open(r'C:\Users\Hangyu\Desktop\JmetalTB\examples\experiment/'+filename, 'w') as of:
        for solution in solutions:
            for variables in solution.variables:
                of.write(str(variables) + " ")
            of.write("\n")

# 将每一个解打印到屏幕
def print_variables_to_screen(solutions):
    if type(solutions) is not list:
        solutions = [solutions]

    for solution in solutions:
        print(solution.variables[0])

# 将每个个体的目标函数写入文件
def print_function_values_to_file(solutions, filename: str):
    LOGGER.info('Output file (function values): ' + filename)

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        pass

    if type(solutions) is not list:
        solutions = [solutions]

    with open(filename, 'w') as of:
        for solution in solutions:
            for i in range(len(solution.objectives)):
                function_value = solution.objectives[i]
                if i == 2:
                    of.write(str(function_value) + ' ')
                else:
                    of.write(str(function_value) + ' ')
            of.write('\n')


# 将获得的所有最优解写入文件
def print_object_to_file(solutions, filename: str):
    # LOGGER.info('Output file (function values): ' + filename)

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        pass

    if type(solutions) is not list:
        solutions = [solutions]

    with open(filename, 'w') as of:
        for solution in solutions:
            for function_value in solution:
                of.write(str(function_value) + ' ')
            of.write('\n')


# 将每个个体的目标函数打印到屏幕
def print_function_values_to_screen(solutions):
    if type(solutions) is not list:
        solutions = [solutions]

    for solution in solutions:
        print(str(solutions.index(solution)) + ": ", sep='  ', end='', flush=True)
        print(solution.objectives, sep='  ', end='', flush=True)
        print()

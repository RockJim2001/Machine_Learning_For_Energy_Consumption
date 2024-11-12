from jmetal.algorithm.multiobjective.ibea import IBEA
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.solution import read_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

from MOEAs.problems.problem import Current_problem
import matplotlib.pyplot as plt

if __name__ == '__main__':
    '''
    ['LinearRegressor', 'KNeighborsRegressor', 'RandomForestRegressor', 'DecisionTreeRegressor',
              'SupportVectorMachine', 'MultilayerPerceptronRegressor', 'ExtraTreeRegressor',
              'XGBoostRegressor', 'LGBMRegressor', 'GBDTRegressor', 'CATBoostRegressor',
              'LSTMRegressor', 'AdaBoostRegressor', 'BaggingRegressor', 'LASSORegressor']
    '''

    problem_name = 'XGBoostRegressor'
    problem = Current_problem(ML_method=problem_name)

    algorithm = IBEA(
        problem=problem,
        kappa=1.,
        population_size=50,
        offspring_population_size=50,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=25000)
    )

    algorithm.run()
    front = algorithm.get_result()

    # Save results to file
    print_function_values_to_file(front, 'Objective.' + algorithm.label + '.csv')
    print_variables_to_file(front, 'Variable.' + algorithm.label + '.csv')

    print(f'Algorithm: ${algorithm.get_name()}')
    print(f'Problem: ${problem.get_name()}')
    print(f'Computing time: ${algorithm.total_computing_time}')

    # todo 绘图 和 HV 指标评价
    length = len(front.objectives)
    for item in front.objectives:
        pass

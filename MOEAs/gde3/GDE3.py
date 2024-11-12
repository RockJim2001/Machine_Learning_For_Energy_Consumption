from jmetal.algorithm.multiobjective.gde3 import GDE3
from jmetal.util.solution import read_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByKeyboard

from MOEAs.problems.problem import Current_problem

if __name__ == '__main__':
    '''
    ['LinearRegressor', 'KNeighborsRegressor', 'RandomForestRegressor', 'DecisionTreeRegressor',
              'SupportVectorMachine', 'MultilayerPerceptronRegressor', 'ExtraTreeRegressor',
              'XGBoostRegressor', 'LGBMRegressor', 'GBDTRegressor', 'CATBoostRegressor',
              'LSTMRegressor', 'AdaBoostRegressor', 'BaggingRegressor', 'LASSORegressor']
    '''

    problem_name = 'XGBoostRegressor'
    problem = Current_problem(ML_method=problem_name)
    max_evaluations = 50

    algorithm = GDE3(
        problem=problem,
        population_size=50,
        cr=0.5,
        f=0.5,
        termination_criterion=StoppingByKeyboard()
    )

    algorithm.run()
    front = algorithm.get_result()

    # Save results to file
    print_function_values_to_file(front, 'Objective.' + algorithm.label + '.csv')
    print_variables_to_file(front, 'Variable.' + algorithm.label + '.csv')

    print(f'Algorithm: ${algorithm.get_name()}')
    print(f'Problem: ${problem.get_name()}')
    print(f'Computing time: ${algorithm.total_computing_time}')

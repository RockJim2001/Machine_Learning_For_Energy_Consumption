from jmetal.algorithm.multiobjective.omopso import OMOPSO
from jmetal.operator import UniformMutation
from jmetal.operator.mutation import NonUniformMutation
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.solution import read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

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

    mutation_probability = 1.0 / problem.number_of_variables
    max_evaluations = 50
    swarm_size = 50

    algorithm = OMOPSO(
        problem=problem,
        swarm_size=swarm_size,
        epsilon=0.0075,
        uniform_mutation=UniformMutation(probability=mutation_probability, perturbation=0.5),
        non_uniform_mutation=NonUniformMutation(mutation_probability, perturbation=0.5,
                                                max_iterations=int(max_evaluations / swarm_size)),
        leaders=CrowdingDistanceArchive(100),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )

    algorithm.run()
    front = algorithm.get_result()

    # Save results to file
    print_function_values_to_file(front, 'Objective.' + algorithm.label + '.csv')
    print_variables_to_file(front, 'Variable.'+ algorithm.label + '.csv')

    print(f'Algorithm: ${algorithm.get_name()}')
    print(f'Problem: ${problem.get_name()}')
    print(f'Computing time: ${algorithm.total_computing_time}')

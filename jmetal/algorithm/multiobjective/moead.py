import copy
import random
import time
from math import ceil
import math
from typing import TypeVar, List, Generator
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.svm import SVC

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.operator import Mutation, Crossover
from jmetal.core.problem import Problem
from jmetal.operator import DifferentialEvolutionCrossover, NaryRandomSolutionSelection
from jmetal.util.aggregative_function import AggregativeFunction
from jmetal.util.constraint_handling import feasibility_ratio, \
    overall_constraint_violation_degree, is_feasible
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.evaluator import Evaluator
from jmetal.util.point import NadirPoint
from jmetal.util.uniformpoint import uniformpoint
from jmetal.util.neighborhood import WeightVectorNeighborhood
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.termination_criterion import TerminationCriterion, StoppingByEvaluations
from jmetal.util.replacement import RankingAndDensityEstimatorReplacement, RemovalPolicyType
from jmetal.operator import PolynomialMutation, MOEAD_M2M_Crossover,MOEAD_M2M_Mutation
from jmetal.util.comparator import DominanceComparator
from jmetal.core.quality_indicator import HyperVolume, InvertedGenerationalDistance

S = TypeVar('S')
R = List[S]

def update2(aggregative_function, dimension, solutions: []) -> None:
    aggregative_function.nadir_point = NadirPoint(dimension)
    for solution in solutions:
        vector = solution.objectives
        aggregative_function.nadir_point.update(vector)
    # print(f"更新天底点:{aggregative_function.nadir_point.get_point()}")

class MOEAD(GeneticAlgorithm):
    '''非EP版本'''
    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 crossover,
                 aggregative_function: AggregativeFunction,
                 neighbourhood_selection_probability: float,
                 max_number_of_replaced_solutions: int,
                 neighbor_size: int,
                 weight_files_path: str,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator):
        """
        :param max_number_of_replaced_solutions: (eta in Zhang & Li paper).
        :param neighbourhood_selection_probability: 在邻域中与解交配的概率（而不是整个种群中） (Delta in Zhang & Li paper).
        """
        # 基类GeneticAlgorithm的构造函数
        super(MOEAD, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=1,
            mutation=mutation,
            crossover=crossover,
            # 随机选择两个
            selection=NaryRandomSolutionSelection(2),
            population_evaluator=population_evaluator,
            population_generator=population_generator,
            termination_criterion=termination_criterion
        )

        self.max_number_of_replaced_solutions = max_number_of_replaced_solutions
        self.fitness_function = aggregative_function
        self.neighbourhood = WeightVectorNeighborhood(
            number_of_weight_vectors=population_size,
            neighborhood_size=neighbor_size,
            weight_vector_size=problem.number_of_objectives,
            weights_path=weight_files_path
        )
        self.neighbourhood_selection_probability = neighbourhood_selection_probability
        self.permutation = None
        self.current_subproblem = 0
        self.neighbor_type = None  # 字符串类型，'NEIGHBOR'或‘POPULATION’
        self.counter = 0  # 用于记录迭代次数gen
        # self.generation_counter = 0 # 迭代次数gen


    def init_progress(self) -> None:
        # 设置初始评价次数
        self.evaluations = self.population_size
        # 根据初始种群设置理想点
        for solution in self.solutions:
            self.fitness_function.update(solution.objectives)
        # 将原问题分成N个子问题
        self.permutation = Permutation(self.population_size)

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def update_progress(self):
        '''写每一次迭代需要做的事情'''
        super().update_progress()  # 继承父代，不是重写！
        if self.evaluations % 10000 == 0:
            print(f'当前评估次数为{self.evaluations}次')
        if self.evaluations % self.population_size == 0:
            update2(self.fitness_function, self.problem.number_of_objectives, self.solutions)

    def selection(self, population: List[S]):

        # 获取第i个子问题，get_next_value()后，指针会自动向后移动
        # counter == 0, 迭代次数就多一次
        self.current_subproblem, self.counter = self.permutation.get_next_value()
        # # 记录迭代次数
        # if self.counter == 0:
        #     self.generation_counter += 1
        # 选取邻域范围
        self.neighbor_type = self.choose_neighbor_type()

        if self.neighbor_type == 'NEIGHBOR':
            # 获取当前问题的邻域
            neighbors = self.neighbourhood.get_neighbors(self.current_subproblem, population)
            # 领域中随机选择两个索引k, l。
            mating_population = self.selection_operator.execute(neighbors)
        else: #‘POPULATION’
            # 整个种群中随机选择两个索引k, l。
            mating_population = self.selection_operator.execute(population)
        # 返回邻域和其自己构成的列表（元素为解）
        mating_population.append(population[self.current_subproblem])
        return mating_population

    def reproduction(self, mating_population: List[S]) -> List[S]:
        '''交叉变异操作  返回后代个体，offspring_population'''
        # 将当前子问题对应的解赋给 self.crossover_operator.current_individual
        # 差分进化算法中使用
        self.crossover_operator.current_individual = self.solutions[self.current_subproblem]

        offspring_population = self.crossover_operator.execute(mating_population)  # 返回交叉产生的新个体 [child]
        self.mutation_operator.execute(offspring_population[0])  # 返回变异产生的新个体

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        # 用reproduction（交叉变异）产生的个体替换 population 从而产生 offspring_population
        new_solution = offspring_population[0]
        # 更新理想点
        self.fitness_function.update(new_solution.objectives)
        # 更新邻域解
        new_population = self.update_current_subproblem_neighborhood(new_solution, population)
        # 返回新种群
        return new_population

    def update_current_subproblem_neighborhood(self, new_solution, population):
        # 更新邻域解对B(k)中的每个解进行遍历 若新解y的切比雪夫小于旧解x，则用新解替换旧解，达到最大替换次数时停止。
        permuted_neighbors_indexes = self.generate_permutation_of_neighbors(self.current_subproblem)
        replacements = 0 # 计算替换次数

        # 遍历B(k)
        for i in range(len(permuted_neighbors_indexes)):
            k = permuted_neighbors_indexes[i]  # 索引
            # Tschebycheff aggregative fuction
            f1 = self.fitness_function.compute(population[k].objectives, self.neighbourhood.weight_vectors[k])
            f2 = self.fitness_function.compute(new_solution.objectives, self.neighbourhood.weight_vectors[k])

            # 新解替换旧解，并记录替换次数
            if f2 < f1:
                population[k] = copy.deepcopy(new_solution)
                replacements += 1
            # 达到替换次数，break (一般为替换两次)
            if replacements >= self.max_number_of_replaced_solutions:
                break

        return population

    def generate_permutation_of_neighbors(self, subproblem_id):
        if self.neighbor_type == 'NEIGHBOR':
            # 获取当前子问题i的邻域列表B(k)（存储个体解的下标）
            neighbors = self.neighbourhood.get_neighborhood()[subproblem_id] # 返回当前子问题i的邻域B(k),nparray列表
            permuted_array = copy.deepcopy(neighbors.tolist())  # 转换为列表形式
        else:
            # 整个种群为当前子问题i的邻域,随机选择一个子问题
            permuted_array = Permutation(self.population_size).get_permutation()

        return permuted_array

    def choose_neighbor_type(self):
        rnd = random.random()

        if rnd < self.neighbourhood_selection_probability:
            neighbor_type = 'NEIGHBOR'
        else:
            neighbor_type = 'POPULATION'

        return neighbor_type

    def get_name(self):
        return 'MOEAD'

    def get_result(self):
        return self.solutions

class MOEAD_DRA(MOEAD):
    def __init__(self, problem, population_size, mutation, crossover, aggregative_function,
                 neighbourhood_selection_probability, max_number_of_replaced_solutions, neighbor_size,
                 weight_files_path, termination_criterion=store.default_termination_criteria,
                 population_generator=store.default_generator, population_evaluator=store.default_evaluator):
        super(MOEAD_DRA, self).__init__(problem, population_size, mutation, crossover, aggregative_function,
                                        neighbourhood_selection_probability, max_number_of_replaced_solutions,
                                        neighbor_size, weight_files_path,
                                        termination_criterion=termination_criterion,
                                        population_generator=population_generator,
                                        population_evaluator=population_evaluator)

        self.saved_values = []
        self.utility = [1.0 for _ in range(population_size)]
        self.frequency = [0.0 for _ in range(population_size)]
        self.generation_counter = 0  # gen
        self.order = []  # I
        self.current_order_index = 0  # k  for k in I:

    def init_progress(self):
        super().init_progress()

        self.evaluations = self.population_size
        for solution in self.solutions:
            self.fitness_function.update(solution.objectives)
        # 初始化过程 MOEAD-DRA 新实现的过程
        self.saved_values = [copy.copy(solution) for solution in self.solutions]
        self.order = self.__tour_selection(10) # 10次锦标赛选择
        self.current_order_index = 0
        #
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def update_progress(self):
        '''写每一次迭代需要做的事情'''
        super().update_progress()  # 继承父代，不是重写！

        if self.evaluations % 10000 == 0:
            print(f'当前评估次数为{self.evaluations}次')
        # k += 1
        self.current_order_index += 1
        # 如果遍历完I了， 恢复初始值，准备下一轮
        if self.current_order_index == (len(self.order)):
            self.order = self.__tour_selection(10)
            self.current_order_index = 0
        # gen += 1
        self.generation_counter += 1
        # 每50代更新一次效用值，更新的效用值会存入self.utility列表中
        if self.generation_counter % 50 == 0:
            self.__utility_function()

    def selection(self, population: List[S]):
        # 确定当前的子问题
        self.current_subproblem = self.order[self.current_order_index]
        # ？ ？ 下面两句注释掉（没有对运行结果产生什么影响）
        #self.current_order_index += 1
        #self.frequency[self.current_subproblem] += 1

        self.neighbor_type = self.choose_neighbor_type()

        if self.neighbor_type == 'NEIGHBOR':
            neighbors = self.neighbourhood.get_neighbors(self.current_subproblem, population)
            mating_population = self.selection_operator.execute(neighbors)
        else:
            mating_population = self.selection_operator.execute(population)

        mating_population.append(population[self.current_subproblem])  # 把自己也放入交配池

        return mating_population

    def get_name(self):
        return 'MOEAD-DRA'

    def __utility_function(self):
        '''私有方法：子问题的效用函数值，根据效用值为子问题分配计算资源'''
        for i in range(len(self.solutions)):
            f1 = self.fitness_function.compute(self.solutions[i].objectives, self.neighbourhood.weight_vectors[i])    # 当前解
            f2 = self.fitness_function.compute(self.saved_values[i].objectives, self.neighbourhood.weight_vectors[i]) # 历史解
            if f2 != 0:
                delta = (f2 - f1) / f2  # jmetal 中源代码为f2 - f1， 但原论文中为(f2 - f1) / f2
            else:
                delta = (f2 - f1) / 10e-6  # jmetal 中源代码为f2 - f1， 但原论文中为(f2 - f1) / f2
            if delta > 0.001:
                self.utility[i] = 1.0
            else:
                utility_value = (0.95 + (0.05 * delta / 0.001)) * self.utility[i]
                self.utility[i] = utility_value if utility_value < 1.0 else 1.0

            self.saved_values[i] = copy.copy(self.solutions[i])

    def __tour_selection(self, depth):
        # depth 是轮盘赌的次数
        # 返回下标索引集合I
        selected = [i for i in range(self.problem.number_of_objectives)]
        candidate = [i for i in range(self.problem.number_of_objectives, self.population_size)]

        while len(selected) < int(self.population_size / 5.0):
            # 随机设置当前最优个体在candidate中的索引
            best_idd = int(random.random() * len(candidate))
            # 最优个体在种群中的索引
            best_sub = candidate[best_idd]
            for i in range(1, depth):
                i2 = int(random.random() * len(candidate))  # 或 random.randint(0, len(candidate))
                s2 = candidate[i2]
                if self.utility[s2] > self.utility[best_sub]:
                    best_idd = i2
                    best_sub = s2
            selected.append(best_sub)
            del candidate[best_idd]

        return selected

class MOEADIEpsilon(MOEAD):
    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 crossover: DifferentialEvolutionCrossover,
                 aggregative_function: AggregativeFunction,
                 neighbourhood_selection_probability: float,
                 max_number_of_replaced_solutions: int,
                 neighbor_size: int,
                 weight_files_path: str,
                 termination_criterion: TerminationCriterion = StoppingByEvaluations(300000),
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator):
        """
        :param max_number_of_replaced_solutions: (eta in Zhang & Li paper).
        :param neighbourhood_selection_probability: Probability of mating with a solution in the neighborhood rather
               than the entire population (Delta in Zhang & Li paper).
        """
        super(MOEADIEpsilon, self).__init__(
            problem=problem,
            population_size=population_size,
            mutation=mutation,
            crossover=crossover,
            aggregative_function=aggregative_function,
            neighbourhood_selection_probability=neighbourhood_selection_probability,
            max_number_of_replaced_solutions=max_number_of_replaced_solutions,
            neighbor_size=neighbor_size,
            weight_files_path=weight_files_path,
            population_evaluator=population_evaluator,
            population_generator=population_generator,
            termination_criterion=termination_criterion
        )
        self.constraints = []
        self.epsilon_k = 0
        self.phi_max = -1e30
        self.epsilon_zero = 0
        self.tc = 800
        self.tao = 0.05
        self.rk = 0
        self.generation_counter = 0
        self.archive = []

    def init_progress(self) -> None:
        super().init_progress()

        # for k in range(self.population_size):
        #    self.constraints[k] = get_overall_constraint_violation_degree(self.permutation[k])
        self.constraints = [overall_constraint_violation_degree(self.solutions[i])
                            for i in range(0, self.population_size)]

        sorted(self.constraints)
        self.epsilon_zero = abs(self.constraints[int(ceil(0.05 * self.population_size))])

        if self.phi_max < abs(self.constraints[0]):
            self.phi_max = abs(self.constraints[0])

        self.rk = feasibility_ratio(self.solutions)
        self.epsilon_k = self.epsilon_zero

    def update_progress(self) -> None:
        super().update_progress()

        if self.evaluations % self.population_size == 0:
            self.update_external_archive()
            self.generation_counter += 1
            self.rk = feasibility_ratio(self.solutions)
            if self.generation_counter >= self.tc:
                self.epsilon_k = 0
            else:
                if self.rk < 0.95:
                    self.epsilon_k = (1 - self.tao) * self.epsilon_k
                else:
                    self.epsilon_k = self.phi_max * (1 + self.tao)

    def update_current_subproblem_neighborhood(self, new_solution, population):
        if self.phi_max < overall_constraint_violation_degree(new_solution):
            self.phi_max = overall_constraint_violation_degree(new_solution)

        permuted_neighbors_indexes = self.generate_permutation_of_neighbors(self.current_subproblem)
        replacements = 0

        for i in range(len(permuted_neighbors_indexes)):
            k = permuted_neighbors_indexes[i]

            f1 = self.fitness_function.compute(population[k].objectives, self.neighbourhood.weight_vectors[k])
            f2 = self.fitness_function.compute(new_solution.objectives, self.neighbourhood.weight_vectors[k])

            cons1 = abs(overall_constraint_violation_degree(self.solutions[k]))
            cons2 = abs(overall_constraint_violation_degree(new_solution))

            if cons1 < self.epsilon_k and cons2 <= self.epsilon_k:
                if f2 < f1:
                    population[k] = copy.deepcopy(new_solution)
                    replacements += 1
            elif cons1 == cons2:
                if f2 < f1:
                    population[k] = copy.deepcopy(new_solution)
                    replacements += 1
            elif cons2 < cons1:
                population[k] = copy.deepcopy(new_solution)
                replacements += 1

            if replacements >= self.max_number_of_replaced_solutions:
                break

        return population

    def update_external_archive(self):
        feasible_solutions = []
        for solution in self.solutions:
            if is_feasible(solution):
                feasible_solutions.append(copy.deepcopy(solution))

        if len(feasible_solutions) > 0:
            feasible_solutions = feasible_solutions + self.archive
            ranking = FastNonDominatedRanking()
            ranking.compute_ranking(feasible_solutions)

            first_rank_solutions = ranking.get_subfront(0)
            if len(first_rank_solutions) <= self.population_size:
                self.archive = []
                for solution in first_rank_solutions:
                    self.archive.append(copy.deepcopy(solution))
            else:
                crowding_distance = CrowdingDistance()
                while len(first_rank_solutions) > self.population_size:
                    crowding_distance.compute_density_estimator(first_rank_solutions)
                    first_rank_solutions = sorted(first_rank_solutions, key=lambda x: x.attributes['crowding_distance'],
                                                  reverse=True)
                    first_rank_solutions.pop()

                self.archive = []
                for solution in first_rank_solutions:
                    self.archive.append(copy.deepcopy(solution))

    def get_result(self):
        return self.archive

class MOEAD_M2M(MOEAD):

    def __init__(self, problem, population_size, mutation, crossover, selection,
                 neighbourhood_selection_probability, weight_files_path,max_number_of_replaced_solutions=2,
                 K=10,max_gen=3000,aggregative_function=None, normalize=False,
                 neighbor_size=10,termination_criterion=store.default_termination_criteria,
                 population_generator=store.default_generator, population_evaluator=store.default_evaluator):
        super().__init__(problem, population_size, mutation, crossover,
                         neighbourhood_selection_probability=neighbourhood_selection_probability,
                         aggregative_function=aggregative_function,
                         max_number_of_replaced_solutions=max_number_of_replaced_solutions,
                         neighbor_size=neighbor_size, weight_files_path=weight_files_path,
                         termination_criterion=termination_criterion,
                         population_generator=population_generator,
                         population_evaluator=population_evaluator)

        self.K = K
        self.S = self.population_size // self.K
        self.R = []
        self.Q = []
        self.generation_counter = 0  # gen
        self.max_gen = max_gen
        # 对每个K问题中的S个解进行索引
        self.current_order_index_x = 0
        #  对K个子问题进行索引
        self.current_order_index_k = 0
        # 选择交叉变异算子
        self.selection_operator = selection
        # 是否需要正则化:
        self.normalize = normalize
        # 修正种群规模，保证种群规模为K*S
        if self.population_size // self.K != int(self.population_size / self.K):
            self.population_size = self.K * self.S
        self.offspring_population_size = 2


    def init_progress(self) -> None:
        # 评估次数 = 初始种群
        self.evaluations = self.population_size
        # 初始化过程 MOEAD-M2M 新实现的过程
        self.solutions = Group(self.solutions, self.K, self.S).get_offspring()
        # 初始化理想点
        self.ideapoint = [float("inf")] * len(self.solutions[0].objectives)
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    # 父类选择
    def selection(self, population: List[S]):
        # 获取当前子问题
        self.current_subproblem = self.current_order_index_k
        # 选择邻域范围
        self.neighbor_type = self.choose_neighbor_type()
        if self.neighbor_type == 'NEIGHBOR':
            neighbors = self.solutions[self.current_subproblem * self.S : (self.current_subproblem+1) * self.S]
            mating_population = self.selection_operator.execute(neighbors)
        else: #‘POPULATION’
            mating_population = self.selection_operator.execute(population)

        # 将自身加入其中
        mating_population.append(population[self.current_subproblem * self.S + self.current_order_index_x])
        return mating_population

    def reproduction(self, mating_population: List[S]) -> List[S]:

        #self.crossover_operator.current_individual = self.solutions[self.current_subproblem * self.S + self.current_order_index_x]
        offspring_population = self.crossover_operator.execute(mating_population)

        for i in range(len(offspring_population)):
            self.mutation_operator.execute(offspring_population[i])

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        # new_solution = offspring_population[0]
        new_solution = offspring_population
        # 理想点更新
        # self.fitness_function.update(new_solution.objectives)
        # R := R U {z}
        # self.R.append(new_solution)
        self.R.extend(new_solution)
        return self.solutions  #new_population = self.solutions

    def update_progress(self) -> None:
        if self.evaluations % 10000 == 0:
            print(f'当前评估次数为{self.evaluations}次')

        # 和原种群合并
        self.current_order_index_x += 1

        # 第内层循环 for k in range(S)
        if self.current_order_index_x == self.S:
            self.current_order_index_x = 0
            # 子问题索引+1
            self.current_order_index_k += 1

        # 第外层循环 for k in range(K)
        if self.current_order_index_k == self.K:
            self.current_order_index_k = 0

            self.generation_counter += 1
            # self.crossover_operator = MOEAD_M2M_Crossover(self.generation_counter, self.max_gen)
            # self.mutation_operator = MOEAD_M2M_Mutation(self.generation_counter, self.max_gen, 1/10)

            # 将Q中解分配到K个子问题中
            self.solutions = Group(self.solutions + self.R, self.K, self.S).get_offspring()
            self.R = []

            # 更新当前理想点
            for i in range(self.population_size):
                vector = self.solutions[i].objectives
                self.ideapoint = [y if y < x else x for x, y in zip(self.ideapoint, vector)]
            # print(self.evaluations, self.ideapoint)

        # 更新当前的记录
        self.evaluations += self.offspring_population_size
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def get_result(self):
        if self.stopping_condition_is_met():
            # 返回第一前沿
            front = FastNonDominatedRanking().compute_ranking(self.solutions)[0]
            return front
        else:
            return self.solutions

    def get_name(self):
        return 'MOEAD-M2M'

class MOEAD_CRA(MOEAD_M2M):

    def __init__(self, problem, population_size, mutation, crossover, selection,
                 neighbourhood_selection_probability, weight_files_path, max_number_of_replaced_solutions=2,
                 K=10, L=100, max_gen=3000, aggregative_function=None, normalize=False,neighbor_size=10,
                 termination_criterion=store.default_termination_criteria,
                 population_generator=store.default_generator, population_evaluator=store.default_evaluator):
        super().__init__(problem, population_size, mutation, crossover, selection,
                         neighbourhood_selection_probability=neighbourhood_selection_probability,
                         aggregative_function=aggregative_function,
                         max_number_of_replaced_solutions=max_number_of_replaced_solutions,
                         neighbor_size=neighbor_size,
                         K=K,
                         normalize=normalize,
                         max_gen=max_gen,
                         weight_files_path=weight_files_path,
                         termination_criterion=termination_criterion,
                         population_generator=population_generator,
                         population_evaluator=population_evaluator)

        self.A = [] # 外部档案集, 规模为N
        self.generation_counter = 0  # 迭代次数
        self.order = []  # 子问题索引
        self.L = L  # 学习期
        self.offspring_population_size = 2  # 子代个体数目
        self.utility = [1.0 for _ in range(self.K)]  # C 选择效用值
        self.saved_C_values = [1 for i in range(self.K)]  # 记录上一代 C 选择效用值
        self.current_lbd_values = [0 for i in range(self.K)]  # 记录当前每个子问题中贡献解的个数
        self.saved_lbd_values = [0 for i in range(self.K)]  # 记录上一代每个子问题中贡献解的个数
        self.solutions_split = []
        self.selection_operator = selection

    def init_progress(self):
        # 评估次数 = 初始种群
        self.evaluations = self.population_size
        # 初始化过程 MOEAD-M2M 新实现的过程
        self.solutions = Group(self.solutions, self.K, self.S).get_offspring()
        # 初始化过程 MOEAD-CRA 新实现的过程
        self.saved_values = [copy.copy(solution) for solution in self.solutions]
        self.order = [i for i in range(self.K)]
        random.shuffle(self.order)
        # 初始化理想点
        self.ideapoint = [float("inf")] * len(self.solutions[0].objectives)
        # 更新评估值
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def selection(self, population: List[S]):
        # 获取当前子问题(重写self.current_subproblem)
        # 选择当前的current_order_index_k子问题执行交叉变异产生后代操作
        self.current_subproblem = self.order[self.current_order_index_k]

        self.neighbor_type = self.choose_neighbor_type()
        if self.neighbor_type == 'NEIGHBOR': # 从当前子种群中选1个个体
            neighbors = self.solutions[self.current_subproblem * self.S : (self.current_subproblem+1) * self.S]
            mating_population = self.selection_operator.execute(neighbors)
        else: #‘POPULATION’从整个种群中选1个个体
            mating_population = self.selection_operator.execute(population)
        # 将自身加入其中
        mating_population.append(population[self.current_subproblem * self.S + self.current_order_index_x])
        return mating_population

    def update_progress(self):
        '''写每一次迭代需要做的事情'''
        #-----------MOEAD-M2M算法-------------#
        if self.evaluations % 10000 == 0:
            print(f'当前评估次数为{self.evaluations}次')

        self.current_order_index_x += 1

        # 第内层循环 for k in range(S)
        if self.current_order_index_x == self.S:
            self.current_order_index_x = 0
            # 子问题索引+1   k for k in I:
            # -----------MOEAD-CRA算法-------------#
            self.current_order_index_k += 1

        # 第外层循环 for k in range(K)
        if self.current_order_index_k == self.K:
            self.current_order_index_k = 0
            self.generation_counter += 1 # 迭代次数+1
            # self.crossover_operator = MOEAD_M2M_Crossover(self.generation_counter, self.max_gen)
            # self.mutation_operator = MOEAD_M2M_Mutation(self.generation_counter, self.max_gen, 1 / 10)

            # 将Q中解分配到K个子问题中
            self.Q = self.solutions + self.R
            self.solutions, self.solutions_split = Group(self.solutions + self.R, self.K, self.S).get_offspring(split="True")

            # 根据效用值选择子问题，准备下一轮迭代
            if self.generation_counter < self.L:
                self.order = [i for i in range(self.K)]
                # 随机打乱顺序
                random.shuffle(self.order)
            else:
                self.order = self.__tour_selection(3) # 3轮锦标赛

            # 维护外部种群A，将Q中最好的N个解选择出来; 每个子种群的贡献解个数
            self.A = self.__update_external_archive()

            # 每L代更新一次效用值，更新的效用值会存入self.utility列表中
            if self.generation_counter % self.L == 0:
                self.__utility_function()

            # 输出当前理想点
            for i in range(self.population_size):
                vector = self.solutions[i].objectives
                self.ideapoint = [y if y < x else x for x, y in zip(self.ideapoint, vector)]
            # print(self.evaluations, self.ideapoint)

            self.R = []

        # -----------算法公用-------------#
        self.evaluations += self.offspring_population_size
        # 更新当前的记录
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def get_name(self):
        return 'MOEAD-CRA'

    def get_result(self):
        if self.stopping_condition_is_met():
            front = FastNonDominatedRanking().compute_ranking(self.Q)[0]
            return front
        else:
            return self.solutions

    def __utility_function(self):
        '''私有方法：子问题的效用函数值，根据效用值为子问题分配计算资源'''
        for i in range(self.K):
            try:
                delta = (self.current_lbd_values[i] - self.saved_lbd_values[i]) / (
                            (self.population_size + 5) - self.saved_lbd_values[i])  # 分母+2 防止分母为0
            except:
                print(f"第{i}个子问题当前贡献长度为={self.saved_lbd_values[i]}")
                print(f"第{i}个子问题上一代贡献长度为={self.current_lbd_values[i]}")
                print(f'种群规模为：{self.population_size}')
            # finally:
            #     print(f"第{i}个子问题当前贡献长度为={self.saved_lbd_values[i]}")
            #     print(f"第{i}个子问题上一代贡献长度为={self.current_lbd_values[i]}")
            #     print(f'种群规模为：{self.population_size}')
            utility_value = (1 + delta) * self.saved_C_values[i]
            self.utility[i] = utility_value
            self.saved_C_values[i] = utility_value
        self.saved_lbd_values = copy.deepcopy(self.current_lbd_values)

    def __tour_selection(self, depth):
        '''depth轮锦标赛算法, 原文中采用的是锦标赛算法'''
        # depth 是轮盘赌的次数
        # 返回下标索引集合I
        selected = []
        candidate = [i for i in range(self.K)]

        while len(selected) < int(self.K):
            # 随机设置当前最优个体在candidate中的索引
            # best_idd = int(random.random() * len(candidate))
            best_idd = random.randint(0, self.K - 1)
            # 最优个体在种群中的索引
            best_sub = candidate[best_idd]
            for i in range(1, depth):
                i2 = random.randint(0, self.K - 1) # 或int(random.random() * len(candidate))
                s2 = candidate[i2]
                if self.utility[s2] > self.utility[best_sub]:
                    best_idd = i2
                    best_sub = s2
            selected.append(best_sub)
            #del candidate[best_idd]
        return selected

    def __update_external_archive(self):
        "维护外部种群A，将Q中最好的N个解选择出来; 以及每个子种群的贡献解个数"
        # 非支配排序
        ranking = FastNonDominatedRanking()
        density_estimator = CrowdingDistance()
        r = RankingAndDensityEstimatorReplacement(ranking, density_estimator, RemovalPolicyType.ONE_SHOT)
        r.ranking.compute_ranking(self.Q)
        # 种群合并 通过非支配排序和拥挤距离产生后代
        result_list = r.sequential_truncation(0, self.population_size)

        # 记录当前每个子问题中贡献解的个数
        # 防止每代的结果叠加，先置空
        self.current_lbd_values = [0] * self.K
        # 计算每个种群贡献度
        for i in range(self.population_size):
            for j in range(self.K):
                for k in range(len(self.solutions_split[j])):
                    if result_list[i] == self.solutions_split[j][k]:
                        #self.solutions_split[j].pop(k)  # 不一定能加快运行速度
                        self.current_lbd_values[j] += 1
        # for k in range(self.K):
        #     for j in range(0, self.S):
        #         if self.solutions[k * self.S + j] in result_list:
        #             self.current_lbd_values[k] += 1
        return result_list

class MOEAD_SVM(MOEAD):
    def __init__(self, problem, population_size, mutation, crossover, aggregative_function,
                 neighbourhood_selection_probability, max_number_of_replaced_solutions, neighbor_size,
                 weight_files_path, termination_criterion=store.default_termination_criteria,
                 population_generator=store.default_generator, population_evaluator=store.default_evaluator):
        super(MOEAD_SVM, self).__init__(problem, population_size, mutation, crossover, aggregative_function,
                                        neighbourhood_selection_probability, max_number_of_replaced_solutions,
                                        neighbor_size, weight_files_path,
                                        termination_criterion=termination_criterion,
                                        population_generator=population_generator,
                                        population_evaluator=population_evaluator)

        self.offspring_population_size = 2  # 产生
        self.generation_counter = 0  # 迭代次数gen
        self.data_pos = [] # 正样本集
        self.data_neg = [] # 负样本集
        self.theta = 0.9   # 模型信任度
        self.gamma = 1     # SVM中RBF参数
        self.C = 100       # SVM中RBF参数

    def step(self):
        # 剔除对每个解都评价的功能, 只对SVM筛选出有希望的解进行评价
        mating_population = self.selection(self.solutions)
        offspring_population = self.reproduction(mating_population)
        # offspring_population = self.evaluate(offspring_population)
        self.solutions = self.replacement(self.solutions, offspring_population)  # 返回下一代种群

    def update_progress(self):
        '''写每一次迭代需要做的事情'''
        # 剔除了父类原函数中有增加评估次数的过程(该功能移到replacement中, svm判断为有希望的解后才能增加评估次数)
        if self.evaluations % 10000 == 0:
            print(f'当前评估次数为{self.evaluations}次')

        # 每次迭代训练一个SVM模型
        if self.generation_counter >= 1 and self.counter == 0:
            # print(len(self.data_neg), len(self.data_pos))
            self.classifier = self.__trainSVM(self.gamma, self.C)  # 模型训练

        # 每次迭代更新天底点, 更新正负样本集, 更新迭代次数
        if (self.counter == 0):
            update2(self.fitness_function, self.problem.number_of_objectives, self.solutions)  # 更新天底点
            self.data_pos = []  # 置空正样本集
            if len(self.data_neg) >= 300:
                self.data_neg = self.data_neg[-300:]
            self.generation_counter += 1

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def reproduction(self, mating_population: List[S]) -> List[S]:
        '''交叉变异操作  返回后代个体，offspring_population'''
        # 将当前子问题对应的解赋给 self.crossover_operator.current_individual
        # 差分进化算法中使用
        # self.crossover_operator.current_individual = self.solutions[self.current_subproblem]
        offspring_population = self.crossover_operator.execute(mating_population)  # 返回交叉产生的新个体 [child]
        self.mutation_operator.execute(offspring_population[0])  # 返回变异产生的新个体
        self.mutation_operator.execute(offspring_population[1])  # 返回变异产生的新个体
        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        # 用 reproduction(交叉变异) 产生的个体替换 population 从而产生 offspring_population
        new_population = self.solutions
        for i in range(self.offspring_population_size):
            new_solution = offspring_population[i]
            if self.generation_counter > 1:
                label = self.classifier.predict(np.array(new_solution.variables[0]).reshape(1,-1))
                r = random.random()
                # 当label为1 或 随机数r>0.9对解进行评估
                if label == 1 or r >= self.theta:
                    # 更新理想点
                    _ = self.evaluate([new_solution])
                    # 构建负样本集合(违反约束且不在集合中), 100是设定的penalty值
                    if (new_solution.objectives[1] > 100) and (new_solution not in self.data_neg):
                        self.data_neg.append(new_solution) # 负样本
                    else:
                        self.data_pos.append(new_solution) # 正样本
                    self.fitness_function.update(new_solution.objectives) # 更新理想点
                    # 更新邻域解
                    new_population = self.update_current_subproblem_neighborhood(new_solution, population)
                    # 评估次数+1
                    self.evaluations += 1
            else:
                _ = self.evaluate([new_solution]) # 对解进行评估
                if (new_solution.objectives[1] > 100):
                    self.data_neg.append(new_solution)
                else:
                    self.data_pos.append(new_solution)
                self.fitness_function.update(new_solution.objectives)
                # 更新邻域解
                new_population = self.update_current_subproblem_neighborhood(new_solution, population)
                # 评估次数+1
                self.evaluations += 1

        return new_population

    def get_name(self):
        return 'MOEAD-SVM'

    def __trainSVM(self,gamma=1, C=100):
        '''训练SVM分类器模型'''
        # 当前点作为正样本, 上一代的点作为负样本
        data_pos = np.array([solution.variables[0] for solution in self.data_pos])  # 正样本
        data_neg = np.array([solution.variables[0] for solution in self.data_neg])  # 负样本
        # 正负样本合并
        trainX = np.concatenate((data_pos, data_neg))
        # [1,1,1,0,0,0] = [1,1,1] + [0,0,0]
        trainY = np.concatenate((np.ones(len(data_pos)), np.zeros(len(data_neg))))
        classifier = SVC(gamma=gamma, C=C)
        classifier.fit(trainX, trainY)

        return classifier

class MOEAD_DRA_CloudComputing(MOEAD_DRA):
    def __init__(self, problem, population_size, mutation, crossover, aggregative_function,
                 neighbourhood_selection_probability, max_number_of_replaced_solutions, neighbor_size,
                 weight_files_path, calculate_IGD=False, termination_criterion=store.default_termination_criteria,
                 population_generator=store.default_generator, population_evaluator=store.default_evaluator):
        super(MOEAD_DRA, self).__init__(problem, population_size, mutation, crossover, aggregative_function,
                                        neighbourhood_selection_probability, max_number_of_replaced_solutions,
                                        neighbor_size, weight_files_path,
                                        termination_criterion=termination_criterion,
                                        population_generator=population_generator,
                                        population_evaluator=population_evaluator)

        self.offspring_population_size = 2
        self.saved_values = []
        self.utility = [1.0 for _ in range(population_size)]
        self.frequency = [0.0 for _ in range(population_size)]
        self.generation_counter = 0  # gen
        self.order = []  # I
        self.current_order_index = 0  # k  for k in I:
        # 计算并保存IGD
        self.calculate_IGD = calculate_IGD
        self.IGD_value = []

    def init_progress(self):
        # 设置初始评价次数
        self.evaluations = self.population_size
        # 根据初始种群设置理想点
        for solution in self.solutions:
            self.fitness_function.update(solution.objectives)
        update2(self.fitness_function, self.problem.number_of_objectives, self.solutions)

        # 将原问题分成N个子问题
        self.permutation = Permutation(self.population_size)
        # 初始化过程 MOEAD-DRA 新实现的过程
        self.saved_values = [copy.copy(solution) for solution in self.solutions] # old function value (用于计算效用值delta)
        self.order = self.__tour_selection(10)  # 10次锦标赛选择选择的子问题下标索引
        self.current_order_index = 0

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def selection(self, population: List[S]):
        # 确定当前的子问题
        self.current_subproblem = self.order[self.current_order_index]
        # ？ ？ 下面两句注释掉（没有对运行结果产生什么影响）
        #self.current_order_index += 1
        #self.frequency[self.current_subproblem] += 1

        self.neighbor_type = self.choose_neighbor_type()

        if self.neighbor_type == 'NEIGHBOR':
            neighbors = self.neighbourhood.get_neighbors(self.current_subproblem, population)
            mating_population = self.selection_operator.execute(neighbors)
        else:
            mating_population = self.selection_operator.execute(population)

        return mating_population

    def update_progress(self):
        '''写每一次迭代需要做的事情'''
        super().update_progress()  # 继承父代，不是重写! (包含更新天底点)
        # k += 1
        self.current_order_index += 1 # 子问题索引集order(原文中为I)中的下标
        # 如果遍历完I了， 恢复初始值，准备下一轮
        if self.current_order_index == (len(self.order)):
            self.order = self.__tour_selection(10) # 根据效用值重新选择子问题
            self.current_order_index = 0 # 子问题索引归零
            self.generation_counter += 1 # gen += 1
        # 每50代更新一次效用值，更新的效用值会存入self.utility列表中
        if self.generation_counter % 50 == 0:
            self.__utility_function()

        # 标准化并计算IGD
        if self.calculate_IGD:
            if self.evaluations % 50 == 0:
                value = self.cal_IGD()
                self.IGD_value.append(value)

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        # 用reproduction（交叉变异）产生的个体替换 population 从而产生 offspring_population
        for i in range(self.offspring_population_size):
            new_solution = offspring_population[i]
            # 更新理想点
            self.fitness_function.update(new_solution.objectives)
            # 更新邻域解
            new_population = self.update_current_subproblem_neighborhood(new_solution, population)
        # 返回新种群
        return new_population

    def get_name(self):
        return 'DRA-ColudCpt'

    # 私有函数不能继承，只能重写
    def __utility_function(self):
        '''私有方法：子问题的效用函数值，根据效用值为子问题分配计算资源'''
        for i in range(len(self.solutions)):
            f1 = self.fitness_function.compute(self.solutions[i].objectives, self.neighbourhood.weight_vectors[i])    # 当前解
            f2 = self.fitness_function.compute(self.saved_values[i].objectives, self.neighbourhood.weight_vectors[i]) # 历史解
            if f2 != 0:
                delta = (f2 - f1) / f2  # jmetal 中源代码为f2 - f1， 但原论文中为(f2 - f1) / f2
            else:
                delta = (f2 - f1) / 10e-6  # jmetal 中源代码为f2 - f1， 但原论文中为(f2 - f1) / f2
            if delta > 0.001:
                self.utility[i] = 1.0
            else:
                utility_value = (0.95 + (0.05 * delta / 0.001)) * self.utility[i]
                self.utility[i] = utility_value if utility_value < 1.0 else 1.0

            self.saved_values[i] = copy.copy(self.solutions[i])

    def __tour_selection(self, depth):
        # depth 是轮盘赌的次数
        # 返回下标索引集合I
        selected = [i for i in range(self.problem.number_of_objectives)]
        candidate = [i for i in range(self.problem.number_of_objectives, self.population_size)]

        while len(selected) < int(self.population_size / 5.0):
            # 随机设置当前最优个体在candidate中的索引
            best_idd = int(random.random() * len(candidate))
            # 最优个体在种群中的索引
            best_sub = candidate[best_idd]
            for i in range(1, depth):
                i2 = int(random.random() * len(candidate))  # 或 random.randint(0, len(candidate))
                s2 = candidate[i2]
                if self.utility[s2] > self.utility[best_sub]:
                    best_idd = i2
                    best_sub = s2
            selected.append(best_sub)
            del candidate[best_idd]

        return selected

class MOEAD_FJSP(MOEAD):
    '''相比于MOEAD, MOEAD_FJSP只修改了选择部分(变为2个父代个体，MOEA/D中用的差分进化需要3个个体)'''
    def __init__(self, problem, population_size, mutation, crossover, aggregative_function,
                 neighbourhood_selection_probability, max_number_of_replaced_solutions, neighbor_size,
                 weight_files_path, termination_criterion=store.default_termination_criteria,
                 population_generator=store.default_generator, population_evaluator=store.default_evaluator):
        super().__init__(problem, population_size, mutation, crossover, aggregative_function,
                         neighbourhood_selection_probability, max_number_of_replaced_solutions,
                         neighbor_size, weight_files_path,
                         termination_criterion=termination_criterion,
                         population_generator=population_generator,
                         population_evaluator=population_evaluator)


    def selection(self, population: List[S]):
        '''返回mating_population'''
        # 获取第i个子问题，get_next_value()后，指针会自动向后移动
        self.current_subproblem,counter = self.permutation.get_next_value()
        # 选取邻域范围
        self.neighbor_type = self.choose_neighbor_type()

        if self.neighbor_type == 'NEIGHBOR':
            # B(k) 中随机选择两个索引k, l。
            neighbors = self.neighbourhood.get_neighbors(self.current_subproblem, population)
            mating_population = self.selection_operator.execute(neighbors)
        else:
            # 整个种群中随机选择两个索引k, l。
            mating_population = self.selection_operator.execute(population)

        return mating_population



class MOMALD_FJSP(MOEAD_FJSP):
    def __init__(self, problem, population_size, mutation, crossover, aggregative_function,
                 neighbourhood_selection_probability, max_number_of_replaced_solutions, neighbor_size,
                 weight_files_path, local_prob, local_search_type, Loop, n_clusters, initial_rules, flag_ep,
                 calculate_IGD=False, termination_criterion=store.default_termination_criteria,
                 population_generator=store.default_generator, population_evaluator=store.default_evaluator):
        super().__init__(problem, population_size, mutation, crossover, aggregative_function,
                         neighbourhood_selection_probability, max_number_of_replaced_solutions,
                         neighbor_size, weight_files_path,
                         termination_criterion=termination_criterion,
                         population_generator=population_generator,
                         population_evaluator=population_evaluator)
        self.comparator = DominanceComparator()
        # 是否采用启发式方法进行种群初始化
        self.local_prob = local_prob
        # 采取领域搜索的类型（0,1,2，3）0：不采取领域搜索策略 1：采取基于关键路径 2：基于ML的邻域搜索 3:1+2的领域搜索
        self.local_search_type = local_search_type
        # 采取启发式规则初始种群(0,1)  0: 采取随机初始种群方式 1：采取随机初始化 + 启发式规则的方式
        self.initial_rules = initial_rules
        # 是否采用外部种群来维护解集(True,False)  True:采用EP  False:不采用GP
        self.flag_ep = flag_ep
        # 外部种群
        self.EP = []
        # 计算IGD的标识
        self.calculate_IGD = calculate_IGD
        # 保存IGD的值
        self.IGD_value = []
        # 局部搜索次数
        self.Loop = Loop
        # 聚类的个数
        self.n_clusters = n_clusters
        # 问题前沿
        self.reference_front = np.array(
            [self.problem.reference_front[i].objectives for i in range(len(self.problem.reference_front))])
        # 执行领域搜索的概率
        self.Pl = 0.5
        # 累计分数值记录
        self.sum_Dp1 = 0
        self.sum_Dp2 = 0
        self.sum_Dp3 = 0
        self.alpha = 0.2
        # 执行局部搜索的频率
        self.T = 200

    def create_initial_solutions(self) -> List[S]:
        return [self.problem.create_solution() for _ in range(self.population_size)]

    def update_EP(self,new_solution):
        '''更新外部种群'''
        temp = []
        if self.EP == []:
            # 获取rank1个体
            temp = FastNonDominatedRanking().compute_ranking(self.solutions)[0]
            for sol in temp:
                if sol.objectives not in [self.EP[i].objectives for i in range(len(self.EP))]:
                    self.EP.append(sol)
        else:
            ep = False
            delete = []
            for k in range(len(self.EP)):
                if (new_solution.objectives == self.EP[k].objectives):  # 如果有一样的就不用算了
                    ep = True
                    break
                if self.comparator.compare(self.EP[k],new_solution)==1:  # flag = 1:new_solution更好，-1：sol更好，0：一样好
                    delete.append(self.EP[k])
                elif ep == False and (self.comparator.compare(new_solution,self.EP[k]) == -1): # new_solution被支配了
                    ep = True
                    break  # 后面就不用看了，最好也是互不支配
            if len(delete) != 0:
                for sol in delete:
                    self.EP.remove(sol)
            if ep == False:
                self.EP.append(new_solution)

            # 超出容量限制，根据拥挤距离删除
            if len(self.EP) > self.population_size:
                ranking = FastNonDominatedRanking()
                density_estimator = CrowdingDistance()
                r = RankingAndDensityEstimatorReplacement(ranking, density_estimator, RemovalPolicyType.ONE_SHOT)
                r.ranking.compute_ranking(self.EP)
                # 种群合并 通过非支配排序和拥挤距离产生后代
                self.EP = r.sequential_truncation(0, self.population_size)

    def update_progress(self):
        '''写每一次迭代需要做的事情'''
        super().update_progress()  # 继承父代，并对父代进行扩展
        # 标准化并计算IGD
        if self.calculate_IGD:
            if self.evaluations % 100 == 0:
                value = self.cal_IGD()
                self.IGD_value.append(value)

        # 执行局部搜索策略
        if self.evaluations % self.population_size == 0:
            self.local_search()

    def local_search(self):
        flag = self.local_search_type
        # 为每一个个体赋予编号
        for i in range(self.population_size):
            self.solutions[i].attributes['index'] = i
        if flag == 0: # 不采取领域搜索策略
            pass
        elif flag == 1: # 只采取关键路径搜索策略
            pass
        elif flag == 2: # 只采取PR搜索策略
            pass
        elif flag == 3: # 采取基于关键路径 + 基于ML的邻域搜索

            for loop in range(self.Loop):

                front_all = FastNonDominatedRanking().compute_ranking(self.solutions)[0]
                front = copy.copy(front_all)

                temp_objective = [p.objectives for p in front]
                # 将目标函数值相同的个体删除
                for r in range(len(temp_objective) - 1, -1, -1):
                    for c in range(r):
                        if temp_objective[r] == temp_objective[c]:
                            del temp_objective[r]
                            del front[r]
                            break
                # 随机对20%的个体执行局部搜索
                local_num = int(len(front) * 0.2)
                local_idx = random.sample([i for i in range(len(front))], local_num)

                if random.random() < self.Pl:
                    # LS1 对关键路径进行局部搜索
                    for i in local_idx:
                        solution = front[i]
                        current_idx = solution.attributes['index']  # 获得当前解的编号
                        new_solution = copy.deepcopy(solution)
                        new_solution.variables = self.problem.critcal_based_search(solution.variables)
                        self.population_evaluator.evaluate_solution(new_solution, self.problem)  # 对解进行评估
                        # 如果新解支配旧解，则进行替换
                        if self.comparator.compare(solution, new_solution) == 1:  # flag = 1:new_solution更好，-1：sol更好，0：一样好
                            # 替换solution(暂时没有替换)
                            self.fitness_function.update(new_solution.objectives)
                            self.update_neighborhood_for_local_search(new_solution, current_idx)
                            if self.flag_ep:
                                self.update_EP(new_solution)

                else:
                    # LS2 执行patch-relinked操作
                    density_estimator = CrowdingDistance()
                    density_estimator.compute_density_estimator(self.solutions)
                    front = FastNonDominatedRanking().compute_ranking(self.solutions)[0]
                    # 对种群进行分类
                    X = np.array([solution.objectives for solution in self.solutions])
                    kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=1).fit(X)
                    labels = kmeans_model.labels_
                    # 将个体分配到对应的组中
                    group = [[] for i in range(len(set(labels)))]
                    max_crowd = []
                    for i in range(self.population_size):
                        group[labels[i] - 1].append(self.solutions[i])
                    # 计算每组中拥挤距离最大的个体，且不选择前沿中的个体
                    for i in range(len(set(labels))):
                        density_estimator.sort(group[i])
                        j = 0
                        while group[i][j] in front:
                            if j == len(group[i]) - 1:
                                break
                            else:
                                j += 1
                        max_crowd.append(group[i][j])

                    for sol in max_crowd:
                        pareto_solution = random.choice(front)
                        # 随机从前沿中选取一个，进行path-relinked操作
                        off1, off2 = self.problem.pathlinked_based_search(pareto_solution.variables, sol.variables)
                        a = [off1, off2]
                        # for k in range(2):
                        for i in [random.choice([0, 1])]:
                            # todo 选择一个支配的，互不支配选择一个离原点近的
                            current_idx = sol.attributes['index']  # 获得当前解的编号
                            new_solution = copy.copy(sol)
                            new_solution.variables = a[i]
                            self.population_evaluator.evaluate_solution(new_solution, self.problem)  # 对解进行评估
                            if self.comparator.compare(sol, new_solution) == 1:  # 1:new_solution更好，-1：sol更好，0：一样好
                                # normalObj = [(new_solution.objectives[k] - self.Min[k]) / (self.Max[k] - self.Min[k])
                                #              for k in range(len(new_solution.objectives))]
                                # self.fitness_function.update(normalObj)
                                self.fitness_function.update(new_solution.objectives)

                                self.update_neighborhood_for_local_search(new_solution, current_idx)

                                # 存在外部种群，则对外部种群进行更新
                                if self.flag_ep:
                                    self.update_EP(new_solution)

    def update_neighborhood_for_local_search(self, new_solution, current_subproblem):
        # 更新邻域解对B(k)中的每个解进行遍历 若新解y的切比雪夫小于旧解x，则用新解替换旧解，达到最大替换次数时停止。
        permuted_neighbors_indexes = self.generate_permutation_of_neighbors(current_subproblem)
        replacements = 0 # 计算替换次数
        # print(current_subproblem,self.current_subproblem)

        # 遍历B(k)
        for k in permuted_neighbors_indexes:
            # Tschebycheff aggregative fuction
            f1 = self.fitness_function.compute(self.solutions[k].objectives, self.neighbourhood.weight_vectors[k])
            f2 = self.fitness_function.compute(new_solution.objectives, self.neighbourhood.weight_vectors[k])

            # 新解替换旧解，并记录替换次数
            if f2 < f1:
                self.solutions[k].objectives = new_solution.objectives
                self.solutions[k].variables = new_solution.variables
                replacements += 1
            # 达到替换次数，break
            if replacements >= self.max_number_of_replaced_solutions:
                break

    def update_probability(self):
        self.Pl = 0.5

    def normal_distance(self, solution):
        # 计算solution和理想点之间的欧氏距离
        # 理想点
        ideal_normal = self.fitness_function.get_ideal_point()
        # 目标值标准化
        normalObj = [(solution.objectives[i] - self.Min[i]) / (self.Max[i] - self.Min[i]) for i
                     in range(len(solution.objectives))]

        # 计算两点之间的欧氏距离
        distance = math.sqrt((normalObj[0] - ideal_normal[0]) ** 2 + (normalObj[1] - ideal_normal[1]) ** 2
                             + (normalObj[2] - ideal_normal[2]) ** 2)

        return distance

    def cal_IGD(self):

        front = FastNonDominatedRanking().compute_ranking(self.solutions)[0]
        algorithm_front = np.array(
            [[front[i].objectives[j] for j in range(len(front[0].objectives))] for i in range(len(front))])
        reference_front = copy.copy(self.reference_front)
        M = len(self.solutions[0].objectives)
        ideal_point = self.fitness_function.get_ideal_point()
        nadir_point = self.fitness_function.get_nadir_point()

        # 标准化当前pareto前沿
        for i in range(M):
            algorithm_front[:, i] = (algorithm_front[:, i] - ideal_point[i]) / (nadir_point[i] - ideal_point[i])
        # 标准化参考pareto前沿
        for i in range(M):
            up = max(reference_front[:, i])
            low = min(reference_front[:, i])
            reference_front[:, i] = (reference_front[:, i] - low) / (up - low)

        value = InvertedGenerationalDistance(reference_front).compute(algorithm_front)

        return value

    def get_IGD_result(self):
        return self.IGD_value

    def get_result(self):
        if self.stopping_condition_is_met():
            if self.flag_ep:
                return self.EP    # 带EP版本的MOEA/D
            else:
                front = FastNonDominatedRanking().compute_ranking(self.solutions)[0]
                return front      # 不带EP版本的MOEA/D
        else:
            return self.solutions

    def get_name(self):
        return ' MLMOMA '

class MALD_FJSP(MOEAD_FJSP):
    def __init__(self, problem, population_size, mutation, crossover, aggregative_function,
                 neighbourhood_selection_probability, max_number_of_replaced_solutions, neighbor_size,
                 weight_files_path, local_prob, local_search_type, Loop, n_clusters, initial_rules, flag_ep,
                 calculate_IGD=False, termination_criterion=store.default_termination_criteria,
                 population_generator=store.default_generator, population_evaluator=store.default_evaluator):
        super().__init__(problem, population_size, mutation, crossover, aggregative_function,
                         neighbourhood_selection_probability, max_number_of_replaced_solutions,
                         neighbor_size, weight_files_path,
                         termination_criterion=termination_criterion,
                         population_generator=population_generator,
                         population_evaluator=population_evaluator)
        self.comparator = DominanceComparator()
        # 是否采用启发式方法进行种群初始化
        self.local_prob = local_prob
        # 采取领域搜索的类型（0,1,2，3）0：不采取领域搜索策略 1：采取基于关键路径 2：基于ML的邻域搜索 3:1+2的领域搜索
        self.local_search_type = local_search_type
        # 采取启发式规则初始种群(0,1)  0: 采取随机初始种群方式 1：采取随机初始化 + 启发式规则的方式
        self.initial_rules = initial_rules
        # 是否采用外部种群来维护解集(True,False)  True:采用EP  False:不采用GP
        self.flag_ep = flag_ep
        # 外部种群
        self.EP = []
        # 计算IGD的标识
        self.calculate_IGD = calculate_IGD
        # 保存IGD的值
        self.IGD_value = []
        # 局部搜索次数
        self.Loop = Loop
        # 聚类的个数
        self.n_clusters = n_clusters
        # 问题前沿
        self.reference_front = np.array(
            [self.problem.reference_front[i].objectives for i in range(len(self.problem.reference_front))])
        # 执行领域搜索的概率
        self.Pl = 0.5
        # 累计分数值记录
        self.sum_Dp1 = 0
        self.sum_Dp2 = 0
        self.sum_Dp3 = 0
        self.alpha = 0.2
        # 执行局部搜索的频率
        self.T = self.population_size
        self.pl = self.population_size

    def create_initial_solutions(self) -> List[S]:
        return [self.problem.create_solution() for _ in range(self.population_size)]

    def update_EP(self,new_solution):
        '''更新外部种群'''
        temp = []
        if self.EP == []:
            # 获取rank1个体
            temp = FastNonDominatedRanking().compute_ranking(self.solutions)[0]
            for sol in temp:
                if sol.objectives not in [self.EP[i].objectives for i in range(len(self.EP))]:
                    self.EP.append(sol)
        else:
            ep = False
            delete = []
            for k in range(len(self.EP)):
                if (new_solution.objectives == self.EP[k].objectives):  # 如果有一样的就不用算了
                    ep = True
                    break
                if self.comparator.compare(self.EP[k],new_solution)==1:  # flag = 1:new_solution更好，-1：sol更好，0：一样好
                    delete.append(self.EP[k])
                elif ep == False and (self.comparator.compare(new_solution,self.EP[k]) == -1): # new_solution被支配了
                    ep = True
                    break  # 后面就不用看了，最好也是互不支配
            if len(delete) != 0:
                for sol in delete:
                    self.EP.remove(sol)
            if ep == False:
                self.EP.append(new_solution)

            # 超出容量限制，根据拥挤距离删除
            if len(self.EP) > self.population_size:
                ranking = FastNonDominatedRanking()
                density_estimator = CrowdingDistance()
                r = RankingAndDensityEstimatorReplacement(ranking, density_estimator, RemovalPolicyType.ONE_SHOT)
                r.ranking.compute_ranking(self.EP)
                # 种群合并 通过非支配排序和拥挤距离产生后代
                self.EP = r.sequential_truncation(0, self.population_size)

    def update_progress(self):
        '''写每一次迭代需要做的事情'''
        super().update_progress()  # 继承父代，并对父代进行扩展

        # # 标准化并计算IGD
        # if self.calculate_IGD:
        #     if self.evaluations % 100 == 0:
        #         value = self.cal_IGD()
        #         self.IGD_value.append(value)
        #
        # # 执行局部搜索策略
        # if self.evaluations % self.pl == 0:
        #     self.local_search()
        #     # todo 更新局部搜索频率
        #     self.update_probability()

    def local_search(self):
        flag = self.local_search_type
        # 为每一个个体赋予编号
        for i in range(self.population_size):
            self.solutions[i].attributes['index'] = i
        if flag == 0: # 不采取领域搜索策略
            pass
        elif flag == 1: # 只采取关键路径搜索策略
            for loop in range(self.Loop):
                # LS1 对关键路径进行局部搜索
                front = FastNonDominatedRanking().compute_ranking(self.solutions)[0]
                temp_objective = [p.objectives for p in front]
                # 将目标函数值相同的个体删除
                for r in range(len(temp_objective) - 1, -1, -1):
                    for c in range(r):
                        if temp_objective[r] == temp_objective[c]:
                            del temp_objective[r]
                            del front[r]
                            break
                # 随机对30%的个体执行局部搜索
                local_num = int(len(front) * 0.3)
                local_idx = random.sample([i for i in range(len(front))], local_num)  # 返回含有local_num个个体的列表

                for i in local_idx:
                    solution = front[i]
                    current_idx = solution.attributes['index']  # 获得当前解的编号
                    new_solution = copy.deepcopy(solution)
                    new_solution.variables = self.problem.critcal_based_search(solution.variables)
                    self.population_evaluator.evaluate_solution(new_solution, self.problem)  # 对解进行评估
                    # 如果新解支配旧解，则进行替换
                    if self.comparator.compare(solution,
                                               new_solution) == 1:  # flag = 1:new_solution更好，-1：sol更好，0：一样好
                        self.fitness_function.update(new_solution.objectives)
                        self.update_neighborhood_for_local_search(new_solution, current_idx)
                        if self.flag_ep:
                            self.update_EP(new_solution)

        elif flag == 2: # 只采取PR搜索策略
            for loop in range(self.Loop):
                # LS2 执行patch-relinked操作
                density_estimator = CrowdingDistance()
                # density_estimator.compute_density_estimator(self.solutions)  #应该在每个group中计算
                front = FastNonDominatedRanking().compute_ranking(self.solutions)[0]
                # 对种群进行分类
                X = np.array([solution.objectives for solution in self.solutions])
                kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=1).fit(X)
                labels = kmeans_model.labels_
                # 将个体分配到对应的组中
                group = [[] for i in range(len(set(labels)))]
                max_crowd = []
                for i in range(self.population_size):
                    group[labels[i] - 1].append(self.solutions[i])
                # 计算每组中拥挤距离最大的个体，且不选择前沿中的个体
                for i in range(len(set(labels))):
                    density_estimator.compute_density_estimator(group[i])
                    density_estimator.sort(group[i])
                    j = 0
                    if group[i][j] not in front:
                        max_crowd.append(group[i][j])
                    else:
                        j += 1

                # 对拥挤距离最大的个体执行pathrelink操作
                for sol in max_crowd:
                    # 随机从前沿中选取一个，进行path-relinked操作
                    pareto_solution = random.choice(front)
                    off1, off2 = self.problem.pathlinked_based_search(pareto_solution.variables, sol.variables)
                    a = [off1, off2]

                    # for k in range(2):
                    for i in [random.choice([0, 1])]:
                        current_idx = sol.attributes['index']  # 获得当前解的编号
                        new_solution = copy.deepcopy(sol)
                        new_solution.variables = a[i]
                        self.population_evaluator.evaluate_solution(new_solution, self.problem)  # 对解进行评估
                        if self.comparator.compare(sol, new_solution) == 1:  # 1:new_solution更好，-1：sol更好，0：一样好
                            self.fitness_function.update(new_solution.objectives)
                            self.update_neighborhood_for_local_search(new_solution, current_idx)
                            # 存在外部种群，则对外部种群进行更新
                            if self.flag_ep:
                                self.update_EP(new_solution)

        elif flag == 3:
            for loop in range(self.Loop):
                # LS2 执行patch-relinked操作
                density_estimator = CrowdingDistance()
                # density_estimator.compute_density_estimator(self.solutions)  #应该在每个group中计算
                front = FastNonDominatedRanking().compute_ranking(self.solutions)[0]
                # 执行投影距离最大的局部搜素策略
                # 获取当前理想点
                ideal_normal = self.fitness_function.get_ideal_point()
                # 计算两点间距离的匿名函数
                D = lambda X, Y: math.sqrt((X[0] - Y[0]) ** 2 + (X[1] - Y[1]) ** 2 + (X[2] - Y[2]) ** 2)
                # 储存每个点投影向量的长度
                dis_list = []
                # 计算每个点和理想点之间的距离 和 两向量夹角的cos值
                for i in range(self.population_size):
                    solution = self.solutions[i]
                    point = solution.objectives
                    cosVal = self.cos(ideal_normal, point)
                    dis = cosVal * D(ideal_normal, point)
                    dis_list.append([dis, i])  # [距离，在种群中的位置]
                dis_list.sort(key=lambda x: x[0], reverse=True)
                # 选出前(n1-n2)个投影距离最大的解执行PR操作
                max_dis = []  # 记录执行PR操作个体
                for i in range(17):
                    max_dis.append(self.solutions[dis_list[i][1]])
                # 对投影距离最大的个体执行pathrelink操作
                for sol in max_dis:
                    # 随机从前沿中选取一个，进行path-relinked操作
                    # todo 这里没用重新对pareto前沿进行排序
                    pareto_solution = random.choice(front)
                    off1, off2 = self.problem.pathlinked_based_search(pareto_solution.variables, sol.variables)
                    a = [off1, off2]
                    # for k in range(2):
                    for i in [random.choice([0, 1])]:
                        current_idx = sol.attributes['index']  # 获得当前解的编号
                        new_solution = copy.deepcopy(sol)
                        new_solution.variables = a[i]
                        self.population_evaluator.evaluate_solution(new_solution, self.problem)  # 对解进行评估
                        if self.comparator.compare(sol, new_solution) == 1:  # 1:new_solution更好，-1：sol更好，0：一样好
                            self.fitness_function.update(new_solution.objectives)
                            self.update_neighborhood_for_local_search(new_solution, current_idx)
                            # 存在外部种群，则对外部种群进行更新
                            if self.flag_ep:
                                self.update_EP(new_solution)

        elif flag == 4: # 采取基于关键路径 + 基于ML的邻域搜索

            for loop in range(self.Loop):
                n1 = 0 # LS1执行次数
                n2 = 0 # LS2执行次数
                n3 = 0 # LS3执行次数
                if random.random() < self.Pl:
                    # LS1 对关键路径进行局部搜索
                    front = FastNonDominatedRanking().compute_ranking(self.solutions)[0]
                    temp_objective = [p.objectives for p in front]
                    # 将目标函数值相同的个体删除
                    for r in range(len(temp_objective) - 1, -1, -1):
                        for c in range(r):
                            if temp_objective[r] == temp_objective[c]:
                                del temp_objective[r]
                                del front[r]
                                break
                    # 随机对30%的个体执行局部搜索
                    local_num = int(len(front) * 0.3)
                    local_idx = random.sample([i for i in range(len(front))], local_num)  # 返回含有local_num个个体的列表

                    for i in local_idx:
                        solution = front[i]
                        current_idx = solution.attributes['index']  # 获得当前解的编号
                        new_solution = copy.deepcopy(solution)
                        new_solution.variables = self.problem.critcal_based_search(solution.variables)
                        self.population_evaluator.evaluate_solution(new_solution, self.problem)  # 对解进行评估
                        # 如果新解支配旧解，则进行替换
                        if self.comparator.compare(solution, new_solution) == 1:  # flag = 1:new_solution更好，-1：sol更好，0：一样好
                            self.fitness_function.update(new_solution.objectives)
                            self.update_neighborhood_for_local_search(new_solution, current_idx)
                            if self.flag_ep:
                                self.update_EP(new_solution)

                        # 记录局部搜索的分数和次数
                        # n1 += 1
                        # if self.local_prob:
                        #     dp = self.normal_distance(solution)
                        #     dp_new = self.normal_distance(new_solution)
                        #     self.sum_Dp1 += max(0, dp - dp_new) / dp

                else:
                    # LS2 执行patch-relinked操作
                    density_estimator = CrowdingDistance()
                    density_estimator.compute_density_estimator(self.solutions)  #应该在每个group中计算
                    front = FastNonDominatedRanking().compute_ranking(self.solutions)[0]
                    # 对种群进行分类
                    X = np.array([solution.objectives for solution in self.solutions])
                    kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=1).fit(X)
                    labels = kmeans_model.labels_
                    # 将个体分配到对应的组中
                    group = [[] for i in range(len(set(labels)))]
                    max_crowd = []
                    for i in range(self.population_size):
                        group[labels[i] - 1].append(self.solutions[i])
                    # 计算每组中拥挤距离最大的个体，且不选择前沿中的个体
                    for i in range(len(set(labels))):
                        #density_estimator.compute_density_estimator(group[k])
                        density_estimator.sort(group[i])
                        j = 0
                        if group[i][j] not in front:
                            max_crowd.append(group[i][j])
                        else:
                            j += 1

                    # 对拥挤距离最大的个体执行pathrelink操作
                    for sol in max_crowd:
                        # 随机从前沿中选取一个，进行path-relinked操作
                        pareto_solution = random.choice(front)
                        off1, off2 = self.problem.pathlinked_based_search(pareto_solution.variables, sol.variables)
                        a = [off1, off2]

                        #for k in range(2):
                        for i in [random.choice([0, 1])]:
                            current_idx = sol.attributes['index']  # 获得当前解的编号
                            new_solution = copy.deepcopy(sol)
                            new_solution.variables = a[i]
                            self.population_evaluator.evaluate_solution(new_solution, self.problem)  # 对解进行评估
                            if self.comparator.compare(sol, new_solution) == 1:  # 1:new_solution更好，-1：sol更好，0：一样好
                                self.fitness_function.update(new_solution.objectives)
                                self.update_neighborhood_for_local_search(new_solution, current_idx)
                                # 存在外部种群，则对外部种群进行更新
                                if self.flag_ep:
                                    self.update_EP(new_solution)
                            # 记录局部搜索的分数和次数
                            # n2 += 1
                            # if self.local_prob:
                            #     dp = self.normal_distance(sol)
                            #     dp_new = self.normal_distance(new_solution)
                            #     self.sum_Dp2 += max(0, dp - dp_new) / dp

                # 执行投影距离最大的局部搜素策略
                if (n1 - n2) > 0:
                    # 获取当前前沿
                    front = FastNonDominatedRanking().compute_ranking(self.solutions)[0]
                    # 获取当前理想点
                    ideal_normal = self.fitness_function.get_ideal_point()
                    # 计算两点间距离的匿名函数
                    D = lambda X, Y: math.sqrt((X[0] - Y[0]) ** 2 + (X[1] - Y[1]) ** 2 + (X[2] - Y[2]) ** 2)
                    # 储存每个点投影向量的长度
                    dis_list = []
                    # 计算每个点和理想点之间的距离 和 两向量夹角的cos值
                    for i in range(self.population_size):
                        solution = self.solutions[i]
                        point = solution.objectives
                        cosVal = self.cos(ideal_normal, point)
                        dis = cosVal * D(ideal_normal, point)
                        dis_list.append([dis, i])  # [距离，在种群中的位置]
                    dis_list.sort(key=lambda x: x[0], reverse=True)
                    # 选出前(n1-n2)个投影距离最大的解执行PR操作
                    max_dis = []  # 记录执行PR操作个体
                    i = 0
                    j = 0
                    while i <= min((n1 - n2), 10):
                        _sol = self.solutions[dis_list[j][1]]
                        if _sol not in front:
                            max_dis.append(self.solutions[dis_list[i][1]])
                            i += 1
                        j += 1
                        if j == self.population_size:
                            break
                            # 对投影距离最大的个体执行pathrelink操作
                    for sol in max_dis:
                        # 随机从前沿中选取一个，进行path-relinked操作
                        pareto_solution = random.choice(front)
                        off1, off2 = self.problem.pathlinked_based_search(pareto_solution.variables, sol.variables)
                        a = [off1, off2]
                        # for k in range(2):
                        for i in [random.choice([0, 1])]:
                            current_idx = sol.attributes['index']  # 获得当前解的编号
                            new_solution = copy.deepcopy(sol)
                            new_solution.variables = a[i]
                            self.population_evaluator.evaluate_solution(new_solution, self.problem)  # 对解进行评估
                            if self.comparator.compare(sol, new_solution) == 1:  # 1:new_solution更好，-1：sol更好，0：一样好
                                self.fitness_function.update(new_solution.objectives)
                                self.update_neighborhood_for_local_search(new_solution, current_idx)
                                # 存在外部种群，则对外部种群进行更新
                                if self.flag_ep:
                                    self.update_EP(new_solution)
                            # # 记录局部搜索的分数
                            # if self.local_prob:
                            #     dp = self.normal_distance(sol)
                            #     dp_new = self.normal_distance(new_solution)
                            #     self.sum_Dp3 += max(0, dp - dp_new) / dp

    def cos(self, vector1, vector2):
        # 计算vector1和vector2夹角余弦值
        norm1 = math.sqrt(sum(list(map(lambda x: math.pow(x, 2), vector1))))
        norm2 = math.sqrt(sum(list(map(lambda x: math.pow(x, 2), vector2))))
        return sum([vector1[i] * vector2[i] for i in range(0, len(vector1))]) / (norm1 * norm2)

    def update_neighborhood_for_local_search(self, new_solution, current_subproblem):
        # 更新邻域解对B(k)中的每个解进行遍历 若新解y的切比雪夫小于旧解x，则用新解替换旧解，达到最大替换次数时停止。
        permuted_neighbors_indexes = self.generate_permutation_of_neighbors(current_subproblem)
        replacements = 0 # 计算替换次数
        # print(current_subproblem,self.current_subproblem)

        # 遍历B(k)
        for k in permuted_neighbors_indexes:
            # Tschebycheff aggregative fuction
            f1 = self.fitness_function.compute(self.solutions[k].objectives, self.neighbourhood.weight_vectors[k])
            f2 = self.fitness_function.compute(new_solution.objectives, self.neighbourhood.weight_vectors[k])

            # 新解替换旧解，并记录替换次数
            if f2 < f1:
                self.solutions[k].objectives = new_solution.objectives
                self.solutions[k].variables = new_solution.variables
                replacements += 1
            # 达到替换次数，break
            if replacements >= self.max_number_of_replaced_solutions:
                break

    def update_probability(self):
        contri = self.sum_Dp1 + self.sum_Dp2 + self.sum_Dp3
        if contri >= 1:
            self.Pl = 3*self.T
        elif 0.5 <= contri < 1:
            self.pl = 2*self.T
        else:
            self.pl = self.T

    def normal_distance(self, solution):
        # 计算solution和理想点之间的欧氏距离
        # 理想点
        ideal_point = self.fitness_function.get_ideal_point()
        nadir_point = self.fitness_function.get_nadir_point()
        # 目标值标准化
        normalObj = [(solution.objectives[i] - ideal_point[i]) / (nadir_point[i] - ideal_point[i]) for i
                     in range(len(solution.objectives))]
        # 计算两点之间的欧氏距离
        distance = math.sqrt(normalObj[0]** 2 + normalObj[1]** 2 + normalObj[2]** 2)
        return distance

    def cal_IGD(self):

        front = FastNonDominatedRanking().compute_ranking(self.solutions)[0]
        algorithm_front = np.array([[front[i].objectives[j] for j in range(len(front[0].objectives))] for i in range(len(front))])
        reference_front = copy.copy(self.reference_front)
        M = len(self.solutions[0].objectives)
        ideal_point = self.fitness_function.get_ideal_point()
        nadir_point = self.fitness_function.get_nadir_point()

        # 标准化当前pareto前沿
        for i in range(M):
            algorithm_front[:, i] = (algorithm_front[:, i] - ideal_point[i]) / (nadir_point[i] - ideal_point[i])
        # 标准化参考pareto前沿
        for i in range(M):
            up = max(reference_front[:, i])
            low = min(reference_front[:, i])
            reference_front[:, i] = (reference_front[:, i] - low) / (up - low)

        value = InvertedGenerationalDistance(reference_front).compute(algorithm_front)

        return value

    def get_IGD_result(self):
        return self.IGD_value

    def get_result(self):
        if self.stopping_condition_is_met():
            if self.flag_ep:
                return self.EP    # 带EP版本的MOEA/D
            else:
                front = FastNonDominatedRanking().compute_ranking(self.solutions)[0]
                return front      # 不带EP版本的MOEA/D
        else:
            return self.solutions

    def get_name(self):
        return 'MALD'

class MALD_FJSP1(MALD_FJSP):
    def __init__(self, problem, population_size, mutation, crossover, aggregative_function,
                 neighbourhood_selection_probability, max_number_of_replaced_solutions, neighbor_size,
                 weight_files_path, local_prob, local_search_type, Loop, n_clusters, fq, PR_num, initial_rules, flag_ep,
                 calculate_IGD=False, termination_criterion=store.default_termination_criteria,
                 population_generator=store.default_generator, population_evaluator=store.default_evaluator):
        super().__init__(problem, population_size, mutation, crossover, aggregative_function,
                         neighbourhood_selection_probability, max_number_of_replaced_solutions,
                         neighbor_size, weight_files_path, local_prob, local_search_type, Loop, n_clusters,
                         initial_rules, flag_ep,
                         termination_criterion=termination_criterion,
                         population_generator=population_generator,
                         population_evaluator=population_evaluator)

        # 执行领域搜索的概率
        self.pl = fq
        # PR局部搜索时的中间解个数
        self.np = PR_num
        # 自动调整局部搜索频率
        self.adaptive = False
        self.calculate_IGD = calculate_IGD

    def local_search(self):
        flag = self.local_search_type
        # 为每一个个体赋予编号
        for i in range(self.population_size):
            self.solutions[i].attributes['index'] = i
        if flag == 0:  # 不采取领域搜索策略
            pass
        elif flag == 4:  # 采取基于关键路径 + 基于ML的邻域搜索

            for loop in range(self.Loop):
                n1 = 0  # LS1执行次数
                n2 = 0  # LS2执行次数
                n3 = 0  # LS3执行次数

                # LS1 对关键路径进行局部搜索
                front = FastNonDominatedRanking().compute_ranking(self.solutions)[0]
                temp_objective = [p.objectives for p in front]
                # 将目标函数值相同的个体删除
                for r in range(len(temp_objective) - 1, -1, -1):
                    for c in range(r):
                        if temp_objective[r] == temp_objective[c]:
                            del temp_objective[r]
                            del front[r]
                            break
                # 随机对30%的个体执行局部搜索
                local_num = int(len(front) * 0.2)
                local_idx = random.sample([i for i in range(len(front))], local_num)  # 返回含有local_num个个体的列表

                for i in local_idx:
                    solution = front[i]
                    current_idx = solution.attributes['index']  # 获得当前解的编号
                    new_solution = copy.deepcopy(solution)
                    new_solution.variables = self.problem.critcal_based_search(solution.variables)
                    self.population_evaluator.evaluate_solution(new_solution, self.problem)  # 对解进行评估
                    # 如果新解支配旧解，则进行替换
                    if self.comparator.compare(solution,
                                               new_solution) == 1:  # flag = 1:new_solution更好，-1：sol更好，0：一样好
                        self.fitness_function.update(new_solution.objectives)
                        self.update_neighborhood_for_local_search(new_solution, current_idx)
                        if self.flag_ep:
                            self.update_EP(new_solution)

                    # 记录局部搜索的分数和次数
                    n1 += 1
                    # dp = self.normal_distance(solution)
                    # dp_new = self.normal_distance(new_solution)
                    # self.sum_Dp1 += max(0, dp - dp_new) / dp


                # LS2 执行patch-relinked操作
                density_estimator = CrowdingDistance()
                density_estimator.compute_density_estimator(self.solutions)  # 应该在每个group中计算
                front = FastNonDominatedRanking().compute_ranking(self.solutions)[0]
                # 对种群进行分类
                X = np.array([solution.objectives for solution in self.solutions])
                kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=1).fit(X)
                labels = kmeans_model.labels_
                # 将个体分配到对应的组中
                group = [[] for i in range(len(set(labels)))]
                max_crowd = []
                for i in range(self.population_size):
                    group[labels[i] - 1].append(self.solutions[i])
                # 计算每组中拥挤距离最大的个体，且不选择前沿中的个体
                for i in range(len(set(labels))):
                    # density_estimator.compute_density_estimator(group[k])
                    density_estimator.sort(group[i])
                    j = 0
                    if group[i][j] not in front:
                        max_crowd.append(group[i][j])
                    else:
                        j += 1

                # 对拥挤距离最大的个体执行pathrelink操作
                for sol in max_crowd:
                    # 随机从前沿中选取一个，进行path-relinked操作
                    pareto_solution = random.choice(front)
                    # if self.np == 1:
                    #     off1, off2 = self.problem.pathlinked_based_search(pareto_solution.variables, sol.variables,self.np)
                    #     a = [off1, off2]
                    # else:
                    a = self.problem.Pathlinked_based_search(pareto_solution.variables, sol.variables,self.np)
                    for i in range(self.np):
                    #for k in [random.choice([0, 1])]:
                        current_idx = sol.attributes['index']  # 获得当前解的编号
                        new_solution = copy.deepcopy(sol)
                        new_solution.variables = a[i]
                        self.population_evaluator.evaluate_solution(new_solution, self.problem)  # 对解进行评估
                        if self.comparator.compare(sol, new_solution) == 1:  # 1:new_solution更好，-1：sol更好，0：一样好
                            self.fitness_function.update(new_solution.objectives)
                            self.update_neighborhood_for_local_search(new_solution, current_idx)
                            # 存在外部种群，则对外部种群进行更新
                            if self.flag_ep:
                                self.update_EP(new_solution)
                        # 记录局部搜索的分数和次数
                        n2 += 1
                        # dp = self.normal_distance(sol)
                        # dp_new = self.normal_distance(new_solution)
                        # self.sum_Dp2 += max(0, dp - dp_new) / dp

                # 执行投影距离最大的局部搜素策略
                if (n1 - n2) > 0:
                    # 获取当前前沿
                    front = FastNonDominatedRanking().compute_ranking(self.solutions)[0]
                    # 获取当前理想点
                    ideal_normal = self.fitness_function.get_ideal_point()
                    # 计算两点间距离的匿名函数
                    D = lambda X, Y: math.sqrt((X[0] - Y[0]) ** 2 + (X[1] - Y[1]) ** 2 + (X[2] - Y[2]) ** 2)
                    # 储存每个点投影向量的长度
                    dis_list = []
                    # 计算每个点和理想点之间的距离 和 两向量夹角的cos值
                    for i in range(self.population_size):
                        solution = self.solutions[i]
                        point = solution.objectives
                        cosVal = self.cos(ideal_normal, point)
                        dis = cosVal * D(ideal_normal, point)
                        dis_list.append([dis, i])  # [距离，在种群中的位置]
                    dis_list.sort(key=lambda x: x[0], reverse=True)
                    # 选出前(n1-n2)个投影距离最大的解执行PR操作
                    max_dis = []  # 记录执行PR操作个体
                    i = 0
                    j = 0
                    while i <= min((n1 - n2), 10):
                        _sol = self.solutions[dis_list[j][1]]
                        if _sol not in front:
                            max_dis.append(self.solutions[dis_list[i][1]])
                            i += 1
                        j += 1
                        if j == self.population_size:
                            break
                            # 对投影距离最大的个体执行pathrelink操作
                    for sol in max_dis:
                        # 随机从前沿中选取一个，进行path-relinked操作
                        pareto_solution = random.choice(front)
                        off1, off2 = self.problem.pathlinked_based_search(pareto_solution.variables, sol.variables,self.np)
                        a = [off1, off2]
                        # for k in range(2):
                        for i in [random.choice([0, 1])]:
                            current_idx = sol.attributes['index']  # 获得当前解的编号
                            new_solution = copy.deepcopy(sol)
                            new_solution.variables = a[i]
                            self.population_evaluator.evaluate_solution(new_solution, self.problem)  # 对解进行评估
                            if self.comparator.compare(sol, new_solution) == 1:  # 1:new_solution更好，-1：sol更好，0：一样好
                                self.fitness_function.update(new_solution.objectives)
                                self.update_neighborhood_for_local_search(new_solution, current_idx)
                                # 存在外部种群，则对外部种群进行更新
                                if self.flag_ep:
                                    self.update_EP(new_solution)
                            # 记录局部搜索的分数
                            # dp = self.normal_distance(sol)
                            # dp_new = self.normal_distance(new_solution)
                            # self.sum_Dp3 += max(0, dp - dp_new) / dp
                self.evaluations += (n1+n2+n3)

    def update_probability(self):
        ideal_point = self.fitness_function.get_ideal_point()
        # 计算前后种群的前进量
        before = 0
        after = 0
        obj_after_record = [sol.objectives for sol in self.solutions]
        for i in range(self.population_size):
            obj_before = [self.obj_record[i][j] - ideal_point[j] for j in range(len(ideal_point))]
            # 计算两点之间的欧氏距离
            distance = math.sqrt(obj_before[0] ** 2 + obj_before[1] ** 2 + obj_before[2] ** 2)
            before += distance

            obj_after = [obj_after_record[i][j] - ideal_point[j] for j in range(len(ideal_point))]
            # 计算两点之间的欧氏距离
            distance = math.sqrt(obj_after[0] ** 2 + obj_after[1] ** 2 + obj_after[2] ** 2)
            after += distance
        contri = before - after
        if self.adaptive:
            if contri >= 30:
                self.pl = self.T
            elif 2 <= contri < 30:
                self.pl = 2 * self.T
            else:
                self.pl = 3*self.T

    def update_progress(self):
        '''写每一次迭代需要做的事情'''
        super().update_progress()

        if self.evaluations % 1000 == 0:
            print(f"当前迭代次数：{self.evaluations}")

        self.obj_record = [sol.objectives for sol in self.solutions]
        # 标准化并计算IGD
        if self.calculate_IGD:
            if self.evaluations % 100 == 0:
                value = self.cal_IGD()
                self.IGD_value.append(value)

        # 执行局部搜索策略
        if self.evaluations % self.pl == 0:
            self.local_search()
            # todo 更新局部搜索频率
            self.update_probability()

    def cal_IGD(self):

        front = FastNonDominatedRanking().compute_ranking(self.solutions)[0]
        algorithm_front = np.array([[front[i].objectives[j] for j in range(len(front[0].objectives))] for i in range(len(front))])
        reference_front = copy.copy(self.reference_front)
        M = len(self.solutions[0].objectives)
        ideal_point = self.fitness_function.get_ideal_point()
        nadir_point = self.fitness_function.get_nadir_point()

        # 标准化当前pareto前沿
        for i in range(M):
            algorithm_front[:, i] = (algorithm_front[:, i] - ideal_point[i]) / (nadir_point[i] - ideal_point[i])
        # 标准化参考pareto前沿
        for i in range(M):
            up = max(reference_front[:, i])
            low = min(reference_front[:, i])
            reference_front[:, i] = (reference_front[:, i] - low) / (up - low)

        value = InvertedGenerationalDistance(reference_front).compute(algorithm_front)

        return value

    def get_IGD_result(self):
        return self.IGD_value

    def get_name(self):
        return 'MALD'

class HMOEAD_FJSP(MOEAD_FJSP):
    def __init__(self, problem, population_size, mutation, crossover, aggregative_function,
                 neighbourhood_selection_probability, max_number_of_replaced_solutions, neighbor_size,
                 weight_files_path, termination_criterion=store.default_termination_criteria,
                 population_generator=store.default_generator, population_evaluator=store.default_evaluator):
        super().__init__(problem, population_size, mutation, crossover, aggregative_function,
                         neighbourhood_selection_probability, max_number_of_replaced_solutions,
                         neighbor_size, weight_files_path,
                         termination_criterion=termination_criterion,
                         population_generator=population_generator,
                         population_evaluator=population_evaluator)

        self.selection_operator = NaryRandomSolutionSelection(1)
        self.comparator = DominanceComparator()
        self.calculate_IGD = False
        # 保存IGD的值
        self.IGD_value = []
        # 局部搜索次数
        self.Loop = 5  # 正常为7
        # 邻域搜索的概率
        self.local_probability = 0.5
        # 邻域搜索类型执行次数
        self.local1_num = 0
        self.local2_num = 0
        # 累计分数值记录
        self.sum_Dp1 = 0
        self.sum_Dp2 = 0
        self.alpha = 0.01
        self.ON = True
        # 问题前沿
        self.reference_front = np.array(
            [self.problem.reference_front[i].objectives for i in range(len(self.problem.reference_front))])

    def init_progress(self):
        super().init_progress()
        update2(self.fitness_function, self.problem.number_of_objectives, self.solutions)

    def selection(self, population: List[S]):
        '''返回mating_population'''
        # 获取第i个子问题，get_next_value()后，指针会自动向后移动
        mating_population = []
        self.current_subproblem,counter = self.permutation.get_next_value()
        # 选取邻域范围
        self.neighbor_type = self.choose_neighbor_type()
        if self.neighbor_type == 'NEIGHBOR':
            # B(k) 中随机选择两个索引k, l。
            neighbors = self.neighbourhood.get_neighbors(self.current_subproblem, population)
            mating_population = self.selection_operator.execute(neighbors)
        else:
            # 整个种群中随机选择两个索引k, l。
            mating_population = self.selection_operator.execute(population)
        # 返回邻域和其自己构成的列表（元素为解）
        mating_population.append(population[self.current_subproblem])
        return mating_population

    def reproduction(self, mating_population: List[S]) -> List[S]:
        # 返回合作机制产生的2个新个体 [child1，child2]
        offspring_population = self.crossover_operator.execute(mating_population)  # 返回合作机制产生的新个体 [child]
        # 比较两个解的支配关系，谁好取谁，一样好则随机取
        self.population_evaluator.evaluate_solution(offspring_population[0], self.problem)  # 对后代1进行评估
        self.population_evaluator.evaluate_solution(offspring_population[1], self.problem)  # 对后代2进行评估
        flag = self.comparator.compare(offspring_population[0], offspring_population[1])  # flag = 1:1更好，-1：0更好，0：一样好
        if flag == 1:
            return [offspring_population[1]]
        elif flag == -1:
            return [offspring_population[0]]
        elif flag == 0:
            return [offspring_population[random.random() < 0.5]]

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        # 用reproduction（交叉变异）产生的个体替换 population 从而产生 offspring_population
        new_solution = offspring_population[0]
        # 执行局部搜索操作
        new_solution = self.local_search(new_solution)
        # 更新理想点
        self.fitness_function.update(new_solution.objectives)
        # 更新邻域解
        new_population = self.update_current_subproblem_neighborhood(new_solution, population)
        # 返回新种群
        return new_population

    def update_progress(self):
        '''写每一次迭代需要做的事情'''
        super().update_progress()  # 继承父代，并对父代进行扩展

        # # 标准化并计算IGD
        # if self.calculate_IGD:
        #     if self.evaluations % 50 == 0:
        #         value = self.cal_IGD()
        #         self.IGD_value.append(value)

        if self.evaluations % self.population_size == 0:
            #self.update_probability()
            self.local1_num = self.local2_num = 0
            self.sum_Dp1 = self.sum_Dp2 = 0

    def local_search(self, solution):
        local_search_solution = copy.copy(solution)

        def local_search_1(local_search_solution):
            # 选择某个关键工序，更换其加工设备和人员
            local_search_solution.variables = self.problem.LS1(local_search_solution.variables)
            # 若完工时间缩短则保留，否则pass
            self.population_evaluator.evaluate_solution(solution, self.problem)  # 评估旧解
            self.population_evaluator.evaluate_solution(local_search_solution, self.problem)  # 评估新解
            if solution.objectives[0] <= local_search_solution.objectives[0]:
                return solution
            else:
                return local_search_solution

        def local_search_2(local_search_solution):
            # 关键路径上用的最多的机器，替换到最少机器上加工，若调整后设备工件数小于调整前的设备，则更换。
            local_search_solution.variables = self.problem.LS2(local_search_solution.variables)
            return local_search_solution

        # todo 降低问题复杂度
        for _ in range(self.Loop):
            dp = self.normal_distance(local_search_solution)
            #print("dp",dp,ideal_point,local_search_solution.objectives)

            # 计算归一化空间中的欧式距离
            if random.random() < self.local_probability:
                new_solution = local_search_1(local_search_solution)  # 产生解
                self.population_evaluator.evaluate_solution(new_solution, self.problem)  # 评估解
                self.evaluations += 1
                self.local1_num += 1
                dp_new = self.normal_distance(new_solution)
                #print("dp_new",dp_new,ideal_point, new_solution.objectives)
                self.sum_Dp1 += max(0, dp_new - dp) / dp
                #self.sum_Dp1 += max(0, dp-dp_new) / dp_new

            else:
                new_solution = local_search_2(local_search_solution)  # 产生解
                self.population_evaluator.evaluate_solution(new_solution, self.problem)  # 评估解
                self.evaluations += 1
                self.local2_num += 1
                dp_new = self.normal_distance(new_solution)
                self.sum_Dp2 += max(0, dp_new - dp) / dp
                #self.sum_Dp2 += max(0, dp-dp_new) / dp_new

            if self.comparator.compare(local_search_solution, new_solution) == 1:  # flag = 1:solution更好，-1：0更好，0：一样好
                local_search_solution = new_solution
        return local_search_solution

    def update_probability(self):
        local1_score = self.sum_Dp1 / self.local1_num
        local2_score = self.sum_Dp2 / self.local2_num
        self.local_probability = self.alpha + ((1 - 2 * self.alpha) * local1_score) / (local1_score + local2_score)

    def normal_distance(self, solution):
        # 计算solution和理想点之间的欧氏距离
        # 理想点
        ideal_point = self.fitness_function.get_ideal_point()
        nadir_point = self.fitness_function.get_nadir_point()
        # 目标值标准化
        normalObj = [(solution.objectives[i] - ideal_point[i]) / (nadir_point[i] - ideal_point[i]) for i
                     in range(len(solution.objectives))]
        # 计算两点之间的欧氏距离
        distance = math.sqrt(normalObj[0] ** 2 + normalObj[1] ** 2 + normalObj[2] ** 2)
        return distance

    def get_name(self):
        return 'HMOEAD'

# 对0 ~ length-1 之间的数进行随机排列; 返回子问题的索引值
class Permutation:

    def __init__(self, length: int):
        self.counter = 0
        self.length = length
        self.permutation = np.random.permutation(length)

    def get_next_value(self):
        next_value = self.permutation[self.counter]
        self.counter += 1

        # 指向最后面时，重新生成一个排列，并将索引设置为0
        if self.counter == self.length:
            self.permutation = np.random.permutation(self.length)
            self.counter = 0

        return next_value, self.counter

    def get_permutation(self):
        return self.permutation.tolist()

class Group():
    "将种群中的个体划分至对应的区域"
    def __init__(self, population: List[S], K: int, S: int):
        # 此时经过了交叉变异，self.population为2倍
        self.population = copy.deepcopy(population)  # 分配前的种群
        self.offspring = []  # 分配后的种群
        self.K = K  # 子种群个数
        self.S = S  # 子种群规模
        self.M = len(self.population[0].objectives) # 目标函数个数
        self.weight = uniformpoint(self.K,self.M) # 权重向量
        # print(len(self.weight))
        self.sub_pop = [[] for _ in range(self.K)]
        self.popsize = self.K * self.S

    def normal(self):
        # 去除不符合约束的解
        pop_temp = []
        for i in range(len(self.population)):
            if self.population[i].objectives[1] > 100:
                pass
            else:
                pop_temp.append(self.population[i])
        self.population = pop_temp


        # 当前种群目标的最大值和最小值，用于归一化
        pop_obj = []
        max_point = [float("-inf")] * self.M
        min_point = [float("inf")] * self.M
        for solution in self.population:
            vector = solution.objectives
            max_point = [y if y > x else x for x, y in zip(max_point, vector)]
            min_point = [y if y < x else x for x, y in zip(min_point, vector)]

        # 目标函数归一化
        for solution in self.population:
            vector = solution.objectives
            vector_temp = [None] * self.M
            for i in range(self.M):
                vector_temp[i] = (vector[i] - min_point[i]) / (max_point[i] - min_point[i])
            pop_obj.append(vector_temp)
        return pop_obj

    def cal_distance(self):
        '''计算余弦距离'''
        pop_obj = self.normal()
        # pop_obj = [self.population[i].objectives for i in range(len(self.population))]
        cosine_distance = cosine_similarity(self.weight, pop_obj)  # numpy
        return cosine_distance

    def allocate(self):
        '''个体关联至距离它最近的子向量上'''
        cosine_distance = self.cal_distance()
        idx = cosine_distance.argmax(0).tolist()  # 获取每列的最大值索引
        for i in range(len(idx)):
            self.sub_pop[idx[i]].append(self.population[i])
        return self.sub_pop

    def repair_population(self,split="False"):
        for i in range(self.K):
            # 子种群规模小于预定大小
            if len(self.sub_pop[i]) < self.S:
                "随机从Q中选出s-|Pk|个解放入Pk中"
                add = random.choices(self.population, k=self.S - len(self.sub_pop[i]))
                self.sub_pop[i].extend(add)
            # 子种群规模超过预定大小
            elif len(self.sub_pop[i]) > self.S:
                "非支配排序，拥挤距离筛选"
                ranking = FastNonDominatedRanking()
                density_estimator = CrowdingDistance()
                r = RankingAndDensityEstimatorReplacement(ranking, density_estimator, RemovalPolicyType.ONE_SHOT)
                r.ranking.compute_ranking(self.sub_pop[i])
                # 种群合并 通过非支配排序和拥挤距离产生后代
                self.sub_pop[i] = r.sequential_truncation(0, self.S)

        for i in range(self.K):
            self.offspring += self.sub_pop[i]
        return self.offspring

    def get_offspring(self,split="False"):
        self.sub_pop = self.allocate()
        if split == "False":
            return self.repair_population(split)
        elif split == "True":
            return self.repair_population(split), self.sub_pop
        else:
            print("split——输入非法")







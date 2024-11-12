import time
import copy
import random
from typing import TypeVar, List, Generator

try:
    import dask
    from distributed import as_completed, Client
except ImportError:
    pass

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.algorithm import DynamicAlgorithm, Algorithm
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem, DynamicProblem
from jmetal.operator import BinaryTournamentSelection,RandomSolutionSelection
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.evaluator import Evaluator
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.replacement import RankingAndDensityEstimatorReplacement, RemovalPolicyType
from jmetal.util.comparator import DominanceComparator, Comparator, MultiComparator
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.util.comparator import EpsilonDominanceComparator
from sklearn.cluster import KMeans
import numpy as np


S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: NSGA-II
   :synopsis: NSGA-II (Non-dominance Sorting Genetic Algorithm II) implementation.
"""


class NSGAII(GeneticAlgorithm[S, R]):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection = BinaryTournamentSelection(
                     MultiComparator([FastNonDominatedRanking.get_comparator(),
                                      CrowdingDistance.get_comparator()])),
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = store.default_comparator):
        """

        NSGA-II is a genetic algorithm (GA), k.e. it belongs to the evolutionary algorithms (EAs)
        family. The implementation of NSGA-II provided in jMetalPy follows the evolutionary
        algorithm template described in the algorithm module (:py:mod:`jmetal.core.algorithm`).

        .. note:: A steady-state version of this algorithm can be run by setting the offspring size to 1.

        :param problem: The problem to solve.
        :param population_size: Size of the population.
        :param mutation: Mutation operator (see :py:mod:`jmetal.operator.mutation`).
        :param crossover: Crossover operator (see :py:mod:`jmetal.operator.crossover`).
        :param selection: Selection operator (see :py:mod:`jmetal.operator.selection`).
        """
        super(NSGAII, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            termination_criterion=termination_criterion,
            population_evaluator=population_evaluator,
            population_generator=population_generator
        )
        self.dominance_comparator = dominance_comparator

    # 只有替换地方需要重写父类方法，根据拥挤距离和非支配排名来产生子代
    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
        """ 该方法通过非支配排序和拥挤距离选择，将当前种群和后代种群结合起来，生成下一代种群。
        :param population: Parent population.
        :param offspring_population: Offspring population.
        :return: 实施了非支配排序和拥挤距离的子代种群
        """
        ranking = FastNonDominatedRanking(self.dominance_comparator)
        density_estimator = CrowdingDistance()
        r = RankingAndDensityEstimatorReplacement(ranking, density_estimator, RemovalPolicyType.ONE_SHOT)
        # 种群合并 通过非支配排序和拥挤距离产生后代
        solutions = r.replace(population, offspring_population)

        return solutions

    def update_progress(self) -> None:
        super(NSGAII, self).update_progress()
        # if self.evaluations % 1000 == 0:
        #     print("当前评估次数为：",self.evaluations)

    def get_result(self) -> R:
        '''返回算法求解到的Pareto前沿值'''
        front = FastNonDominatedRanking().compute_ranking(self.solutions)[0]
        return front
        # return self.solutions

    def get_name(self) -> str:
        return 'NSGAII'

# 2022 03 13 算法
# 改进包括: 算法流程改进, 自适应交叉变异概率, 精英保留策略
class NSGAII_improve(NSGAII[S, R]):
    '''A memetic algorithm for multi-objective flexible job-shop problem with worker flexibility (2018)'''
    def __init__(self,
                 problem: DynamicProblem[S],
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection = RandomSolutionSelection(),
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: DominanceComparator = DominanceComparator()):
        super(NSGAII_improve, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            population_evaluator=population_evaluator,
            population_generator=population_generator,
            termination_criterion=termination_criterion,
            dominance_comparator=dominance_comparator)

    # 选择
    def selection(self, population: List[S]):
        # 不用选择，直接返回个体
        return population

    # 产生新解（后代）
    def reproduction(self, mating_population: List[S]) -> List[S]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()
        offspring_population1 = []  # 储存交叉产生的个体 offspring1
        offspring_population2 = []  # 储存变异产生的个体 offspring2
        # 交叉操作产生offspring_population1
        for i in range(0, self.offspring_population_size, number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                parents.append(mating_population[i + j])
            offspring_population1.extend(self.crossover_operator.execute(parents))
        # 变异操作产生offspring_population2，变异后的种群规模不够，则随机产生剩余解
        offspring = copy.copy(self.solutions)
        for solution in offspring:
            self.mutation_operator.execute(solution)
            offspring_population2.append(solution)
        if len(offspring_population2) < self.offspring_population_size:
            individuals = [self.population_generator.new(self.problem) for _ in range(self.offspring_population_size-len(offspring_population2))]
            offspring_population2.extend(individuals)

        # 合并子代和父代
        offspring_population = offspring_population1 + offspring_population2
        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:

        temp_population = population + offspring_population

        # 对20%的精英个体进行保留, 不进行交叉变异
        local_num = int(len(temp_population)*0.2)
        local_idx = random.sample([i for i in range(len(temp_population))],local_num)
        for i in local_idx:
            individual = temp_population[i]
            temp_population[i] = individual

        ranking = FastNonDominatedRanking(self.dominance_comparator)
        density_estimator = CrowdingDistance()
        r = RankingAndDensityEstimatorReplacement(ranking, density_estimator, RemovalPolicyType.ONE_SHOT)
        # 种群合并 通过非支配排序和拥挤距离产生后代
        solutions = r.replace(temp_population[:self.population_size], temp_population[self.population_size:])
        return solutions

    def get_name(self) -> str:
        return 'NSGAII_improve'

class MA(NSGAII[S, R]):
    '''A memetic algorithm for multi-objective flexible job-shop problem with worker flexibility (2018)'''
    def __init__(self,
                 problem: DynamicProblem[S],
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection = RandomSolutionSelection(),
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: DominanceComparator = DominanceComparator()):
        super(MA, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            population_evaluator=population_evaluator,
            population_generator=population_generator,
            termination_criterion=termination_criterion,
            dominance_comparator=dominance_comparator)

    # 选择
    def selection(self, population: List[S]):
        # 不用选择，直接返回个体
        return population

    # 产生新解（后代）
    def reproduction(self, mating_population: List[S]) -> List[S]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()
        offspring_population1 = []  # 储存交叉产生的个体
        offspring_population2 = []  # 储存变异产生的个体
        # 交叉操作产生offspring_population1
        for i in range(0, self.offspring_population_size, number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                parents.append(mating_population[i + j])
            offspring_population1.extend(self.crossover_operator.execute(parents))
        # 变异操作产生offspring_population2，变异后的种群规模不够，则随机产生剩余解
        offspring = copy.copy(self.solutions)
        for solution in offspring:
            self.mutation_operator.execute(solution)
            offspring_population2.append(solution)
        if len(offspring_population2) < self.offspring_population_size:
            individuals = [self.population_generator.new(self.problem) for _ in range(self.offspring_population_size-len(offspring_population2))]
            offspring_population2.extend(individuals)

        # 合并子代和父代
        offspring_population = offspring_population1 + offspring_population2
        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
        # 合并种群
        #temp_population = population + offspring_population
        temp_population = offspring_population
        temp_objective = [p.objectives for p in temp_population]
        # 将目标函数值相同的个体删除
        for r in range(len(temp_objective) - 1, -1, -1):
            for c in range(r):
                if temp_objective[r] == temp_objective[c]:
                    del temp_objective[r]
                    del temp_population[r]
                    break
        # 随机对20%的个体执行局部搜索
        local_num = int(len(temp_population)*0.1)  # 正常值为0.2
        local_idx = random.sample([i for i in range(len(temp_population))],local_num)
        for i in local_idx:
            individual = temp_population[i]
            new_individual = self.local_search(individual)
            temp_population[i] = new_individual

        ranking = FastNonDominatedRanking(self.dominance_comparator)
        density_estimator = CrowdingDistance()
        r = RankingAndDensityEstimatorReplacement(ranking, density_estimator, RemovalPolicyType.ONE_SHOT)
        # 种群合并 通过非支配排序和拥挤距离产生后代
        solutions = r.replace(temp_population[:self.population_size], temp_population[self.population_size:])
        # print(len(temp_population),len(solutions))
        return solutions

    def local_search(self,individual):
        new_individual = copy.copy(individual)
        new_individual.variables = self.problem.critcal_based_search(individual.variables)
        self.population_evaluator.evaluate_solution(new_individual, self.problem)  # 对解进行评估
        self.evaluations += 1
        return new_individual

    def update_progress(self) -> None:
        super().update_progress()
        if self.evaluations % self.population_size == 0:
            print("当前迭代次数为：",self.evaluations)

    def get_name(self) -> str:
        return 'MA'

class DynamicNSGAII(NSGAII[S, R], DynamicAlgorithm):

    def __init__(self,
                 problem: DynamicProblem[S],
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection = BinaryTournamentSelection(
                     MultiComparator([FastNonDominatedRanking.get_comparator(),
                                      CrowdingDistance.get_comparator()])),
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: DominanceComparator = DominanceComparator()):
        super(DynamicNSGAII, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            population_evaluator=population_evaluator,
            population_generator=population_generator,
            termination_criterion=termination_criterion,
            dominance_comparator=dominance_comparator)
        self.completed_iterations = 0
        self.start_computing_time = 0
        self.total_computing_time = 0

    def restart(self):
        self.solutions = self.evaluate(self.solutions)

    def update_progress(self):
        if self.problem.the_problem_has_changed():
            self.restart()
            self.problem.clear_changed()

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

        self.evaluations += self.offspring_population_size

    def stopping_condition_is_met(self):
        if self.termination_criterion.is_met:
            observable_data = self.get_observable_data()
            observable_data['TERMINATION_CRITERIA_IS_MET'] = True
            self.observable.notify_all(**observable_data)

            self.restart()
            self.init_progress()

            self.completed_iterations += 1

class DistributedNSGAII(Algorithm[S, R]):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 number_of_cores: int,
                 client,
                 selection: Selection = BinaryTournamentSelection(
                     MultiComparator([FastNonDominatedRanking.get_comparator(),
                                      CrowdingDistance.get_comparator()])),
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 dominance_comparator: DominanceComparator = DominanceComparator()):
        super(DistributedNSGAII, self).__init__()
        self.problem = problem
        self.population_size = population_size
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection
        self.dominance_comparator = dominance_comparator

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.number_of_cores = number_of_cores
        self.client = client

    def create_initial_solutions(self) -> List[S]:
        return [self.problem.create_solution() for _ in range(self.number_of_cores)]

    def evaluate(self, solutions: List[S]) -> List[S]:
        return self.client.map(self.problem.evaluate, solutions)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def get_observable_data(self) -> dict:
        ctime = time.time() - self.start_computing_time

        return {'PROBLEM': self.problem,
                'EVALUATIONS': self.evaluations,
                'SOLUTIONS': self.get_result(),
                'COMPUTING_TIME': ctime}

    def init_progress(self) -> None:
        self.evaluations = self.number_of_cores

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def step(self) -> None:
        pass

    def update_progress(self):
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def run(self):
        """ Execute the algorithm. """
        self.start_computing_time = time.time()

        create_solution = dask.delayed(self.problem.create_solution)
        evaluate_solution = dask.delayed(self.problem.evaluate)

        task_pool = as_completed([], with_results=True)

        for _ in range(self.number_of_cores):
            new_solution = create_solution()
            new_evaluated_solution = evaluate_solution(new_solution)
            future = self.client.compute(new_evaluated_solution)

            task_pool.add(future)

        batches = task_pool.batches()

        auxiliar_population = []
        while len(auxiliar_population) < self.population_size:
            batch = next(batches)
            for _, received_solution in batch:
                auxiliar_population.append(received_solution)

                if len(auxiliar_population) < self.population_size:
                    break

            # submit as many new tasks as we collected
            for _ in batch:
                new_solution = create_solution()
                new_evaluated_solution = evaluate_solution(new_solution)
                future = self.client.compute(new_evaluated_solution)

                task_pool.add(future)

        self.init_progress()

        # perform an algorithm step to create a new solution to be evaluated
        while not self.stopping_condition_is_met():
            batch = next(batches)

            for _, received_solution in batch:
                offspring_population = [received_solution]

                # replacement
                ranking = FastNonDominatedRanking(self.dominance_comparator)
                density_estimator = CrowdingDistance()

                r = RankingAndDensityEstimatorReplacement(ranking, density_estimator, RemovalPolicyType.ONE_SHOT)
                auxiliar_population = r.replace(auxiliar_population, offspring_population)

                # selection
                mating_population = []
                for _ in range(2):
                    solution = self.selection_operator.execute(auxiliar_population)
                    mating_population.append(solution)

                # Reproduction and evaluation
                new_task = self.client.submit(reproduction, mating_population, self.problem,
                                              self.crossover_operator, self.mutation_operator)
                task_pool.add(new_task)

                # update progress
                self.evaluations += 1
                self.solutions = auxiliar_population

                self.update_progress()

                if self.stopping_condition_is_met():
                    break

        self.total_computing_time = time.time() - self.start_computing_time

        # at this point, computation is done
        for future, _ in task_pool:
            future.cancel()

    def get_result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return 'dNSGA-II'

def reproduction(mating_population: List[S], problem, crossover_operator, mutation_operator) -> S:
    offspring_pool = []
    for parents in zip(*[iter(mating_population)] * 2):
        offspring_pool.append(crossover_operator.execute(parents))

    offspring_population = []
    for pair in offspring_pool:
        for solution in pair:
            mutated_solution = mutation_operator.execute(solution)
            offspring_population.append(mutated_solution)

    return problem.evaluate(offspring_population[0])

# 大创论文算法
class INSGAII(NSGAII[S, R]):
    def __init__(self,
                 problem: DynamicProblem[S],
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection = RandomSolutionSelection(),
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: DominanceComparator = DominanceComparator()):
        super(INSGAII, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            population_evaluator=population_evaluator,
            population_generator=population_generator,
            termination_criterion=termination_criterion,
            dominance_comparator=dominance_comparator)

        self.n_clusters = 5

    # 选择
    def selection(self, population: List[S]):
        # 不用选择，直接返回个体
        return population

    # 产生新解（后代）
    def reproduction(self, mating_population: List[S]) -> List[S]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()
        offspring_population1 = []  # 储存交叉产生的个体
        offspring_population2 = []  # 储存变异产生的个体

        # 聚类算法，对种群进行分类
        X = np.array([solution.objectives for solution in mating_population])
        kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=1).fit(X)
        labels = kmeans_model.labels_
        # 将个体分配到对应的组中
        group = [[] for i in range(self.n_clusters)]
        for i in range(self.population_size):
            group[labels[i] - 1].append(mating_population[i])

        # 求5个类平均大小
        group_len = []
        for i in range(self.n_clusters):
            group_len.append(len(group[i]))
        max_idx = group_len.index(max(group_len)) # 最大长度的下标索引
        mean = 0 # 除了最大的聚类 其余平均长度
        for i in range(self.n_clusters):
            if i != max_idx:
                mean += group_len[i]
        mean /= (self.n_clusters-1)

        # 计算空缺，并用交叉变异算子来填补空缺
        gap_num = group_len[max_idx] - int(mean)
        crossover_pop = []
        mutation_pop = []

        for k in range(gap_num):
            # 产生后代(交叉变异)
            for i in range(0, self.offspring_population_size, number_of_parents_to_combine):
                # 从最大的中选一个个体
                parents = []
                p1 = random.choice(group[max_idx])
                parents.append(p1)

                for j in range(number_of_parents_to_combine-1):
                    # 从其余中选取一个
                    idx = random.randint(0, self.n_clusters - 1)
                    while idx == max_idx:
                        idx = random.randint(0, self.n_clusters - 1)
                    p2 = random.choice(group[idx])
                    parents.append(p2)
                    crossover_pop.extend(self.crossover_operator.execute(parents))

            # 变异操作产生offspring
            for solution in crossover_pop:
                self.mutation_operator.execute(solution)
                mutation_pop.append(solution)

        # 删除max-mean个, 再合并
        for i in range(gap_num):
            rand_idx = random.randint(0,len(group[max_idx])-1)
            group[max_idx].pop(rand_idx)
        group[max_idx].extend(mutation_pop)

        # 将新种群赋值给mating_population
        new_population = []
        for i in range(self.n_clusters):
            new_population.extend(group[i])

        # 交叉操作产生offspring_population1
        for i in range(0, self.offspring_population_size, number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                parents.append(new_population[i + j])

            offspring_population1.extend(self.crossover_operator.execute(parents))

        # 变异操作产生offspring_population2，变异后的种群规模不够，则随机产生剩余解
        # offspring = copy.copy(self.solutions)
        # for solution in offspring:
        #     self.mutation_operator.execute(solution)
        #     offspring_population2.append(solution)
        #
        # if len(offspring_population2) < self.offspring_population_size:
        #     individuals = [self.population_generator.new(self.problem) for _ in range(self.offspring_population_size-len(offspring_population2))]
        #     offspring_population2.extend(individuals)

        offspring_population2 = []
        for i in range(0, self.offspring_population_size, number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                parents.append(mating_population[i + j])

            offspring = self.crossover_operator.execute(parents)

            for solution in offspring:
                self.mutation_operator.execute(solution)
                offspring_population2.append(solution)
                if len(offspring_population2) >= self.offspring_population_size:
                    break

        # 合并子代和父代
        offspring_population = offspring_population1 + offspring_population2
        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
        """ 该方法通过非支配排序和拥挤距离选择，将当前种群和后代种群结合起来，生成下一代种群。
        :param population: Parent population.
        :param offspring_population: Offspring population.
        :return: 实施了非支配排序和拥挤距离的子代种群
        """
        ranking = FastNonDominatedRanking(self.dominance_comparator)
        density_estimator = CrowdingDistance()
        r = RankingAndDensityEstimatorReplacement(ranking, density_estimator, RemovalPolicyType.ONE_SHOT)
        # 种群合并 通过非支配排序和拥挤距离产生后代
        solutions = r.replace(population, offspring_population)
        return solutions

    def get_name(self) -> str:
        return 'INSGAII'
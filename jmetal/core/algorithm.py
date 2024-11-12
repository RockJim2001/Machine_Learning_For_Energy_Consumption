import logging
import threading  # 多线程模块
import time
from abc import abstractmethod, ABC
from typing import TypeVar, Generic, List

from jmetal.config import store
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution

LOGGER = logging.getLogger('jmetal')

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: algorithm
   :synopsis: Templates for algorithms.
"""

# 所有算法的基类
class Algorithm(Generic[S, R], threading.Thread, ABC):

    def __init__(self):
        threading.Thread.__init__(self)
        self.solutions: List[S] = []  # 种群
        self.evaluations = 0          # 评估次数
        self.start_computing_time = 0 # 开始计算时间
        self.total_computing_time = 0 # 总共计算时间

        self.observable = store.default_observable

    @abstractmethod
    def create_initial_solutions(self) -> List[S]:
        """ 创建初始种群列表 Creates the initial list of solutions of a metaheuristic. """
        pass

    @abstractmethod
    def evaluate(self, solution_list: List[S]) -> List[S]:
        """ 对种群进行评估 Evaluates a solution list. """
        pass

    @abstractmethod
    def init_progress(self) -> None:
        """ 算法初始化 Initialize the algorithm. """
        pass

    @abstractmethod
    def stopping_condition_is_met(self) -> bool:
        """ 是否满足终止条件 The stopping condition is met or not. """
        pass

    @abstractmethod
    def step(self) -> None:
        """(选择，产生子类，评估，更新) Performs one iteration/step of the algorithm's loop. """
        pass

    @abstractmethod
    def update_progress(self) -> None:
        """ (更新评估次数) Update the progress after each iteration. """
        pass

    @abstractmethod
    def get_observable_data(self) -> dict:
        """获取跟踪值 Get observable data, with the information that will be send to all observers each time. """
        pass

    def run(self):
        """执行算法  Execute the algorithm. """
        self.start_computing_time = time.time()
        # 种群初始化（初始化种群列表）
        self.solutions = self.create_initial_solutions()
        # 种群适应度值评估
        self.solutions = self.evaluate(self.solutions)
        LOGGER.debug('Initializing progress')

        # 算法初始化
        self.init_progress()
        LOGGER.debug('Running main loop until termination criteria is met')
        while not self.stopping_condition_is_met():
            # 选择，产生子代，评估，更新种群
            self.step()
            # 更新评价次数
            self.update_progress()

        # 计算运行时间
        self.total_computing_time = time.time() - self.start_computing_time

    @abstractmethod
    def get_result(self) -> R:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class DynamicAlgorithm(Algorithm[S, R], ABC):

    @abstractmethod
    def restart(self) -> None:
        pass


class EvolutionaryAlgorithm(Algorithm[S, R], ABC):

    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 offspring_population_size: int):
        super(EvolutionaryAlgorithm, self).__init__()
        self.problem = problem
        self.population_size = population_size
        self.offspring_population_size = offspring_population_size

    @abstractmethod
    def selection(self, population: List[S]) -> List[S]:
        """选择父代个体 Select the best-fit individuals for reproduction (parents). """
        pass

    @abstractmethod
    def reproduction(self, population: List[S]) -> List[S]:
        """交叉变异产生子代个体 Breed new individuals through crossover and mutation operations to give birth to offspring. """
        pass

    @abstractmethod
    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        """ 替换 Replace least-fit population with new individuals. """
        pass

    def get_observable_data(self) -> dict:
        return {'PROBLEM': self.problem,
                'EVALUATIONS': self.evaluations,
                'SOLUTIONS': self.get_result(),
                'COMPUTING_TIME': time.time() - self.start_computing_time}

# 初始化评估次数
    def init_progress(self) -> None:
        self.evaluations = self.population_size

        observable_data = self.get_observable_data()   # 返回一个字典
        self.observable.notify_all(**observable_data)

# 选择，产生子类，评估，更新
    def step(self):
        mating_population = self.selection(self.solutions)
        offspring_population = self.reproduction(mating_population)
        offspring_population = self.evaluate(offspring_population)
        self.solutions = self.replacement(self.solutions, offspring_population)  # 返回了下一代种群
    
# 当前评估次数
    def update_progress(self) -> None:
        self.evaluations += self.offspring_population_size
        # 更新当前的记录
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    @property
    def label(self) -> str:
        return f'{self.get_name()}.{self.problem.get_name()}'


# 粒子群算法
class ParticleSwarmOptimization(Algorithm[FloatSolution, List[FloatSolution]], ABC):

    def __init__(self,
                 problem: Problem[S],
                 swarm_size: int):
        super(ParticleSwarmOptimization, self).__init__()
        self.problem = problem
        self.swarm_size = swarm_size

    @abstractmethod
    def initialize_velocity(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def initialize_particle_best(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def initialize_global_best(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_velocity(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_particle_best(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_position(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def perturbation(self, swarm: List[FloatSolution]) -> None:
        pass

    def get_observable_data(self) -> dict:
        return {'PROBLEM': self.problem,
                'EVALUATIONS': self.evaluations,
                'SOLUTIONS': self.get_result(),
                'COMPUTING_TIME': time.time() - self.start_computing_time}

    def init_progress(self) -> None:
        self.evaluations = self.swarm_size

        self.initialize_velocity(self.solutions)
        self.initialize_particle_best(self.solutions)
        self.initialize_global_best(self.solutions)

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def step(self):
        self.update_velocity(self.solutions)
        self.update_position(self.solutions)
        self.perturbation(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_global_best(self.solutions)
        self.update_particle_best(self.solutions)

    def update_progress(self) -> None:
        self.evaluations += self.swarm_size

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    @property
    def label(self) -> str:
        return f'{self.get_name()}.{self.problem.get_name()}'

from abc import abstractmethod, ABC
from typing import TypeVar, List

import numpy as np
from numpy.linalg import LinAlgError
from scipy import special

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.config import store
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.comparator import Comparator, MultiComparator
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: NSGA-III
   :platform: Unix, Windows
   :synopsis: NSGA-III (Non-dominance Sorting Genetic Algorithm III) implementation.

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>, Julian Blank <blankjul@egr.msu.edu>
"""


class ReferenceDirectionFactory(ABC):

    def __init__(self, n_dim: int, scaling=None) -> None:
        self.n_dim = n_dim
        self.scaling = scaling

    def compute(self):
        if self.n_dim == 1:
            return np.array([[1.0]])
        else:
            ref_dirs = self._compute()
            if self.scaling is not None:
                ref_dirs = ref_dirs * self.scaling + ((1 - self.scaling) / self.n_dim)
            return ref_dirs

    @abstractmethod
    def _compute(self):
        pass


class UniformReferenceDirectionFactory(ReferenceDirectionFactory):

    def __init__(self, n_dim: int, scaling=None, n_points: int = None, n_partitions: int = None) -> None:
        super().__init__(n_dim, scaling)
        if n_points is not None:
            self.n_partitions = self.get_partition_closest_to_points(n_points, n_dim)
        else:
            if n_partitions is None:
                raise Exception("Either provide number of partitions or number of points.")
            else:
                self.n_partitions = n_partitions

    def _compute(self):
        return self.uniform_reference_directions(self.n_partitions, self.n_dim)

    def uniform_reference_directions(self, n_partitions: int, n_dim: int):
        ref_dirs = []
        ref_dir = np.full(n_dim, np.inf)
        self.__uniform_reference_directions(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)

    def __uniform_reference_directions(self, ref_dirs, ref_dir, n_partitions: int, beta: int, depth: int):
        if depth == len(ref_dir) - 1:
            ref_dir[depth] = beta / (1.0 * n_partitions)
            ref_dirs.append(ref_dir[None, :])
        else:
            for i in range(beta + 1):
                ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
                self.__uniform_reference_directions(ref_dirs, np.copy(ref_dir), n_partitions, beta - i,
                                                    depth + 1)

    @staticmethod
    def get_partition_closest_to_points(n_points, n_dim):
        # in this case the do method will always return one values anyway
        if n_dim == 1:
            return 0

        n_partitions = 1
        _n_points = UniformReferenceDirectionFactory.get_n_points(n_partitions, n_dim)
        while _n_points <= n_points:
            n_partitions += 1
            _n_points = UniformReferenceDirectionFactory.get_n_points(n_partitions, n_dim)

        return n_partitions - 1

    @staticmethod
    def get_n_points(n_partitions, n_dim):
        return int(special.binom(n_dim + n_partitions - 1, n_partitions))


def get_extreme_points(F, n_objs, ideal_point, extreme_points=None):
    """
        Calculate the Achievement Scalarization Function which is used for the extreme point decomposition.
        计算用于极点分解的ASF函数
    """
    asf = np.eye(n_objs)
    asf[asf == 0] = 1e6

    # add the old extreme points to never loose them for normalization
    # 添加旧的极端点，以使它们在归一化过程中不致丢失。
    _F = F
    if extreme_points is not None:
        _F = np.concatenate([extreme_points, _F], axis=0)

    # use __F because we substitute small values to be 0
    # 使用__F，因为我们将小的数值替换为0
    __F = _F - ideal_point
    __F[__F < 1e-3] = 0

    # update the extreme points for the normalization having the highest asf value each
    # 更新每一个具有最高ASF值极端点用于归一化
    F_asf = np.max(__F * asf[:, None, :], axis=2)  # 在None上添加一个维度
    idx = np.argmin(F_asf, axis=1) # [49,2,30]     # ASF函数最小值
    extreme_points = _F[idx, :]                    # 极端点

    return extreme_points


def get_nadir_point(extreme_points, ideal_point, worst_point, worst_of_front, worst_of_population):
    """ Calculate the axis intersects for a set of individuals and its extremes (construct hyperplane). """
    '''计算超平面的系数，及超平面在坐标轴上的截距'''
    try:
        # find the intercepts using gaussian elimination
        # A * M = b
        M = extreme_points - ideal_point        # 极值点 - 理想点  (系数矩阵)
        b = np.ones(extreme_points.shape[1])    # 生成1*m的全1矩阵，m为目标个数 (系数矩阵)
        plane = np.linalg.solve(M, b)           # 获取超平面的系数
        intercepts = 1 / plane                  # 获取超平面和坐标轴的截距

        nadir_point = ideal_point + intercepts  # nadir_point = 理想点 + 截距

        if not np.allclose(np.dot(M, plane), b) or np.any(intercepts <= 1e-6) or np.any(nadir_point > worst_point):
            raise LinAlgError()
    except LinAlgError:
        nadir_point = worst_of_front            # nadir_point 为当前前沿的最差点

    b = nadir_point - ideal_point <= 1e-6       # 截距 < 0
    nadir_point[b] = worst_of_population[b]     # nadir_point 为当前种群最差

    return nadir_point


def niching(pop: List[S], n_remaining: int, niche_count, niche_of_individuals, dist_to_niche):
    '''返回临界层中存活的个体的索引'''
    survivors = []

    # boolean array of elements that are considered for each iteration
    mask = np.full(len(pop), True)

    while len(survivors) < n_remaining: # 没从临界层中挑选足够个体
        # number of individuals to select in this iteration
        n_select = n_remaining - len(survivors)

        # all niches where new individuals can be assigned to and the corresponding niche count
        next_niches_list = np.unique(niche_of_individuals[mask])
        next_niche_count = niche_count[next_niches_list]

        # the minimum niche count
        # 选取当前关联个体最少的参考点关联的个体数
        min_niche_count = next_niche_count.min()

        # all niches with the minimum niche count (truncate if randomly select more niches than remaining individuals)
        next_niches = next_niches_list[np.where(next_niche_count == min_niche_count)[0]]
        next_niches = next_niches[np.random.permutation(len(next_niches))[:n_select]]

        for next_niche in next_niches:
            # indices of individuals that are considered and assign to next_niche
            next_ind = np.where(np.logical_and(niche_of_individuals == next_niche, mask))[0]

            # shuffle to break random_search tie (equal perp. dist) or select randomly
            np.random.shuffle(next_ind)

            if niche_count[next_niche] == 0:
                next_ind = next_ind[np.argmin(dist_to_niche[next_ind])]
                is_closest = True
            else:
                # already randomized through shuffling
                next_ind = next_ind[0]
                is_closest = False

            # add the selected individual to the survivors
            # 将个体加入存活种群中
            mask[next_ind] = False
            pop[next_ind].attributes['is_closest'] = is_closest
            survivors.append(int(next_ind))

            # increase the corresponding niche count
            # 对应的关联参考点 关联个体的个数+1
            niche_count[next_niche] += 1

    return survivors


def associate_to_niches(F, niches, ideal_point, nadir_point, utopian_epsilon: float = 0.0):
    """ Associate each solution to a reference point.
        将每个点关联到均匀分布的参考点上去
    """
    utopian_point = ideal_point - utopian_epsilon

    denom = nadir_point - utopian_point  # 天底点 - 理想点
    denom[denom == 0] = 1e-12

    # normalize by ideal point and intercepts
    # 根据理想点和截距归一化种群
    N = (F - utopian_point) / denom
    def compute_perpendicular_distance(N, ref_dirs):
        '''计算垂直距离'''
        # tile(平铺)  沿y轴扩大len(N)倍，沿x轴扩大1倍(不变) —— 用于计算每个点到种群中个体的最小值
        u = np.tile(ref_dirs, (len(N), 1))
        # np.repeat() 将N按行复制len(ref_dirs)份
        v = np.repeat(N, len(ref_dirs), axis=0)
        # linalg=linear（线性）+algebra（代数），norm默认表示求2范数。
        # axis=1表示按行向量处理，求多个行向量的范数(每行元素平方和开根号，即每行获得一个元素)
        norm_u = np.linalg.norm(u, axis=1)
        # v * u 对应位置相乘
        scalar_proj = np.sum(v * u, axis=1) / norm_u
        proj = scalar_proj[:, None] * u / norm_u[:, None]
        val = np.linalg.norm(proj - v, axis=1)
        matrix = np.reshape(val, (len(N), len(ref_dirs)))
        return matrix

    dist_matrix = compute_perpendicular_distance(N, niches)

    niche_of_individuals = np.argmin(dist_matrix, axis=1) # 获取离每个参考点最近的个体的索引
    dist_to_niche = dist_matrix[np.arange(F.shape[0]), niche_of_individuals] # 获取到参考点的最小距离
    # niche_of_individuals 和 dist_to_niche 均为len(N)*len(ref_dirs)的矩阵
    # niche_of_individuals 中存的是索引, dist_to_niche中存的是距离
    return niche_of_individuals, dist_to_niche


def compute_niche_count(n_niches: int, niche_of_individuals):
    # n_niches 参考点；niche_of_individuals 个体
    niche_count = np.zeros(n_niches, dtype=np.int)
    # np.unique: 去除数组中的重复数字，并进行排序之后输出(index); count为每个数字出现的次数。
    index, count = np.unique(niche_of_individuals, return_counts=True)
    niche_count[index] = count # 每个个体index关联的数目count

    return niche_count


class NSGAIII(NSGAII):

    def __init__(self,
                 reference_directions,
                 problem: Problem,
                 mutation: Mutation,
                 crossover: Crossover,
                 population_size: int = None,
                 selection: Selection = BinaryTournamentSelection(
                     MultiComparator([FastNonDominatedRanking.get_comparator(),
                                      CrowdingDistance.get_comparator()])),
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,    # 随机产生
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = store.default_comparator): # 支配比较
        self.reference_directions = reference_directions.compute()

        if not population_size:
            population_size = len(self.reference_directions)
        if self.reference_directions.shape[1] != problem.number_of_objectives:
            raise Exception('Dimensionality of reference points must be equal to the number of objectives')

        super(NSGAIII, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=population_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            termination_criterion=termination_criterion,
            population_evaluator=population_evaluator,
            population_generator=population_generator,
            dominance_comparator=dominance_comparator
        )

        self.extreme_points = None  # 极值点
        self.ideal_point = np.full(self.problem.number_of_objectives, np.inf)  # idea point  [inf, inf, ...]
        self.worst_point = np.full(self.problem.number_of_objectives, -np.inf) # nadir point [-inf, -inf, ...]

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        """ Implements NSGA-III environmental selection based on reference points as described in:

        * Deb, K., & Jain, H. (2014). An Evolutionary Many-Objective Optimization
          Algorithm Using Reference-Point-Based Nondominated Sorting Approach,
          Part I: Solving Problems With Box Constraints. IEEE Transactions on
          Evolutionary Computation, 18(4), 577–601. doi:10.1109/TEVC.2013.2281535.
        """
        F = np.array([s.objectives for s in population])

        # find or usually update the new ideal point - from feasible solutions
        # note that we are assuming minimization here!
        # 设置理想点
        self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

        # calculate the fronts of the population
        # 获取非支配前沿
        ranking = FastNonDominatedRanking(self.dominance_comparator)
        ranking.compute_ranking(population + offspring_population, k=self.population_size)

        # fronts = 每层个体组成的列表 [[第1层个体],[第2层个体],[第3层个体],...,[临界层个体]]
        # non_dominated: 非支配集，第0层前沿
        fronts, non_dominated = ranking.ranked_sublists, ranking.get_subfront(0)

        # find the extreme points for normalization
        # 获取在每个目标上的极值点
        self.extreme_points = get_extreme_points(F=np.array([s.objectives for s in non_dominated]),
                                                 n_objs=self.problem.number_of_objectives,
                                                 ideal_point=self.ideal_point,
                                                 extreme_points=self.extreme_points)

        # find the intercepts for normalization and do backup if gaussian elimination fails
        # 获取截距用于归一化
        worst_of_population = np.max(F, axis=0) # 当前种群中每个目标上的最差点
        worst_of_front = np.max(np.array([s.objectives for s in non_dominated]), axis=0) # 当前前沿中每个目标上的最差点

        nadir_point = get_nadir_point(extreme_points=self.extreme_points,
                                      ideal_point=self.ideal_point,
                                      worst_point=self.worst_point,
                                      worst_of_population=worst_of_population,
                                      worst_of_front=worst_of_front)

        # consider only the population until we come to the splitting front
        # pop为临界层个体组成的列表
        pop = np.concatenate(ranking.ranked_sublists)
        F = np.array([s.objectives for s in pop]) # 此时F中个体数目多于种群大小，要执行NSGA-III的筛选操作

        # update the front indices for the current population
        # 更新当前种群前沿索引
        counter = 0
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                fronts[i][j] = counter  # 第i层的第j个个体
                counter += 1            # 每赋值给一个个体后都加1
        last_front = np.array(fronts[-1]) # 获取临界层

        # associate individuals to niches
        # niches：合适的位置
        niche_of_individuals, dist_to_niche = associate_to_niches(F=F,
                                                                  niches=self.reference_directions,
                                                                  ideal_point=self.ideal_point,
                                                                  nadir_point=nadir_point)

        # if we need to select individuals to survive
        # 选择存活体
        if len(pop) > self.population_size:
            # if there is only one front
            # 若只有一个前沿
            if len(fronts) == 1:
                until_last_front = np.array([], dtype=np.int)
                niche_count = np.zeros(len(self.reference_directions), dtype=np.int) # 每个参考点关联的个体数
                n_remaining = self.population_size
            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1]) # 除了临界层外的所有层(已存活个体)
                # 除临界层外所有个体与参考点的匹配数
                niche_count = compute_niche_count(len(self.reference_directions),
                                                  niche_of_individuals[until_last_front])
                n_remaining = self.population_size - len(until_last_front)  # 需要从临界层中选择的个体数


            S_idx = niching(pop=pop[last_front],
                            n_remaining=n_remaining, # 需要从临界层中挑选的个数
                            niche_count=niche_count,
                            niche_of_individuals=niche_of_individuals[last_front],
                            dist_to_niche=dist_to_niche[last_front])

            # 下一代种群的索引survivors_idx和种群pop
            survivors_idx = np.concatenate((until_last_front, last_front[S_idx].tolist()))
            pop = pop[survivors_idx]

        return list(pop)

    def get_result(self):
        """ Return only non dominated solutions."""
        ranking = FastNonDominatedRanking(self.dominance_comparator)
        ranking.compute_ranking(self.solutions, k=self.population_size)

        return ranking.get_subfront(0)

    def get_name(self) -> str:
        return 'NSGA3'

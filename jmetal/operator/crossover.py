import copy
import random
from typing import List

from jmetal.core.operator import Crossover
from jmetal.core.solution import Solution, FloatSolution, BinarySolution, PermutationSolution, IntegerSolution, \
    CompositeSolution
from jmetal.util.ckecking import Check

"""
module:: crossover
"""


class NullCrossover(Crossover[Solution, Solution]):
    def __init__(self):
        super(NullCrossover, self).__init__(probability=0.0)

    def execute(self, parents: List[Solution]) -> List[Solution]:
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))

        return parents

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'Null crossover'

# 部分匹配交叉（找到两点，两点之间片段交叉）
class PMXCrossover(Crossover[PermutationSolution, PermutationSolution]):
    def __init__(self, probability: float):
        super(PMXCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[PermutationSolution]) -> List[PermutationSolution]:
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        permutation_length = offspring[0].number_of_variables

        rand = random.random()
        if rand <= self.probability:
            cross_points = sorted([random.randint(0, permutation_length) for _ in range(2)])

            def _repeated(element, collection):
                c = 0
                for e in collection:
                    if e == element:
                        c += 1
                return c > 1

            def _swap(data_a, data_b, cross_points):
                c1, c2 = cross_points
                new_a = data_a[:c1] + data_b[c1:c2] + data_a[c2:]
                new_b = data_b[:c1] + data_a[c1:c2] + data_b[c2:]
                return new_a, new_b

            def _map(swapped, cross_points):
                n = len(swapped[0])
                c1, c2 = cross_points
                s1, s2 = swapped
                map_ = s1[c1:c2], s2[c1:c2]
                for i_chromosome in range(n):
                    if not c1 < i_chromosome < c2:
                        for i_son in range(2):
                            while _repeated(swapped[i_son][i_chromosome], swapped[i_son]):
                                map_index = map_[i_son].index(swapped[i_son][i_chromosome])
                                swapped[i_son][i_chromosome] = map_[1 - i_son][map_index]
                return s1, s2

            swapped = _swap(parents[0].variables, parents[1].variables, cross_points)
            mapped = _map(swapped, cross_points)

            offspring[0].variables, offspring[1].variables = mapped

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'Partially Matched crossover'


class CXCrossover(Crossover[PermutationSolution, PermutationSolution]):
    def __init__(self, probability: float):
        super(CXCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[PermutationSolution]) -> List[PermutationSolution]:
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[1]), copy.deepcopy(parents[0])]
        rand = random.random()

        if rand <= self.probability:
            for i in range(parents[0].number_of_variables):
                idx = random.randint(0, len(parents[0].variables[i]) - 1)
                curr_idx = idx
                cycle = []

                while True:
                    cycle.append(curr_idx)
                    curr_idx = parents[0].variables[i].index(parents[1].variables[i][curr_idx])

                    if curr_idx == idx:
                        break

                for j in range(len(parents[0].variables[i])):
                    if j in cycle:
                        offspring[0].variables[i][j] = parents[0].variables[i][j]
                        offspring[1].variables[i][j] = parents[0].variables[i][j]

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'Cycle crossover'

#模拟二进制交叉
class SBXCrossover(Crossover[FloatSolution, FloatSolution]):
    __EPS = 1.0e-14

    def __init__(self, probability: float, distribution_index: float = 20.0):
        super(SBXCrossover, self).__init__(probability=probability)
        self.distribution_index = distribution_index
        if distribution_index < 0:
            raise Exception("The distribution index is negative: " + str(distribution_index))

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        Check.that(issubclass(type(parents[0]), FloatSolution), "Solution type invalid: " + str(type(parents[0])))
        Check.that(issubclass(type(parents[1]), FloatSolution), "Solution type invalid")
        Check.that(len(parents) == 2, 'The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        rand = random.random()

        if rand <= self.probability:
            for i in range(parents[0].number_of_variables):
                value_x1, value_x2 = parents[0].variables[i], parents[1].variables[i]

                if random.random() <= 0.5:
                    if abs(value_x1 - value_x2) > self.__EPS:
                        if value_x1 < value_x2:
                            y1, y2 = value_x1, value_x2
                        else:
                            y1, y2 = value_x2, value_x1

                        lower_bound, upper_bound = parents[0].lower_bound[i], parents[1].upper_bound[i]

                        beta = 1.0 + (2.0 * (y1 - lower_bound) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        rand = random.random()
                        if rand <= (1.0 / alpha):
                            betaq = pow(rand * alpha, (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))
                        beta = 1.0 + (2.0 * (upper_bound - y2) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        if rand <= (1.0 / alpha):
                            betaq = pow((rand * alpha), (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))

                        if c1 < lower_bound:
                            c1 = lower_bound
                        if c2 < lower_bound:
                            c2 = lower_bound
                        if c1 > upper_bound:
                            c1 = upper_bound
                        if c2 > upper_bound:
                            c2 = upper_bound

                        if random.random() <= 0.5:
                            offspring[0].variables[i] = c2
                            offspring[1].variables[i] = c1
                        else:
                            offspring[0].variables[i] = c1
                            offspring[1].variables[i] = c2
                    else:
                        offspring[0].variables[i] = value_x1
                        offspring[1].variables[i] = value_x2
                else:
                    offspring[0].variables[i] = value_x1
                    offspring[1].variables[i] = value_x2
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'SBX crossover'

#整数模拟二进制交叉
class IntegerSBXCrossover(Crossover[IntegerSolution, IntegerSolution]):
    __EPS = 1.0e-14

    def __init__(self, probability: float, distribution_index: float = 20.0):
        super(IntegerSBXCrossover, self).__init__(probability=probability)
        self.distribution_index = distribution_index

    def execute(self, parents: List[IntegerSolution]) -> List[IntegerSolution]:
        Check.that(issubclass(type(parents[0]), IntegerSolution), "Solution type invalid")
        Check.that(issubclass(type(parents[1]), IntegerSolution), "Solution type invalid")
        Check.that(len(parents) == 2, 'The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        rand = random.random()

        if rand <= self.probability:
            for i in range(parents[0].number_of_variables):
                value_x1, value_x2 = parents[0].variables[i], parents[1].variables[i]

                if random.random() <= 0.5:
                    if abs(value_x1 - value_x2) > self.__EPS:
                        if value_x1 < value_x2:
                            y1, y2 = value_x1, value_x2
                        else:
                            y1, y2 = value_x2, value_x1

                        lower_bound, upper_bound = parents[0].lower_bound[i], parents[1].upper_bound[i]

                        beta = 1.0 + (2.0 * (y1 - lower_bound) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        rand = random.random()
                        if rand <= (1.0 / alpha):
                            betaq = pow(rand * alpha, (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))
                        beta = 1.0 + (2.0 * (upper_bound - y2) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        if rand <= (1.0 / alpha):
                            betaq = pow((rand * alpha), (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))

                        if c1 < lower_bound:
                            c1 = lower_bound
                        if c2 < lower_bound:
                            c2 = lower_bound
                        if c1 > upper_bound:
                            c1 = upper_bound
                        if c2 > upper_bound:
                            c2 = upper_bound

                        if random.random() <= 0.5:
                            offspring[0].variables[i] = int(c2)
                            offspring[1].variables[i] = int(c1)
                        else:
                            offspring[0].variables[i] = int(c1)
                            offspring[1].variables[i] = int(c2)
                    else:
                        offspring[0].variables[i] = value_x1
                        offspring[1].variables[i] = value_x2
                else:
                    offspring[0].variables[i] = value_x1
                    offspring[1].variables[i] = value_x2
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'Integer SBX crossover'

# 单点交叉（返回[子代个体1， 子代个体2]）
class SPXCrossover(Crossover[BinarySolution, BinarySolution]):

    def __init__(self, probability: float):
        super(SPXCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[BinarySolution]) -> List[BinarySolution]:
        Check.that(type(parents[0]) is BinarySolution, "Solution type invalid")
        Check.that(type(parents[1]) is BinarySolution, "Solution type invalid")
        Check.that(len(parents) == 2, 'The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        rand = random.random()

        if rand <= self.probability:
            # 1. Get the total number of bits
            total_number_of_bits = parents[0].get_total_number_of_bits()

            # 2. Calculate the point to make the crossover
            crossover_point = random.randrange(0, total_number_of_bits)

            # 3. Compute the variable containing the crossover bit
            variable_to_cut = 0
            bits_count = len(parents[1].variables[variable_to_cut])
            while bits_count < (crossover_point + 1):
                variable_to_cut += 1
                bits_count += len(parents[1].variables[variable_to_cut])

            # 4. Compute the bit into the selected variable
            diff = bits_count - crossover_point
            crossover_point_in_variable = len(parents[1].variables[variable_to_cut]) - diff

            # 5. Apply the crossover to the variable
            bitset1 = copy.copy(parents[0].variables[variable_to_cut])
            bitset2 = copy.copy(parents[1].variables[variable_to_cut])

            for i in range(crossover_point_in_variable, len(bitset1)):
                swap = bitset1[i]
                bitset1[i] = bitset2[i]
                bitset2[i] = swap

            offspring[0].variables[variable_to_cut] = bitset1
            offspring[1].variables[variable_to_cut] = bitset2

            # 6. Apply the crossover to the other variables
            for i in range(variable_to_cut + 1, parents[0].number_of_variables):
                offspring[0].variables[i] = copy.deepcopy(parents[1].variables[i])
                offspring[1].variables[i] = copy.deepcopy(parents[0].variables[i])

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'Single point crossover'

# 差分进化交叉
class DifferentialEvolutionCrossover(Crossover[FloatSolution, FloatSolution]):
    """ This operator receives two parameters: the current individual and an array of three parent individuals. The
    best and rand variants depends on the third parent, according whether it represents the current of the "best"
    individual or a random_search one. The implementation of both variants are the same, due to that the parent selection is
    external to the crossover operator.
    该运算符接收两个参数：当前个体和三个父个体的数组。3个父亲个体产生1个子代个体
    """

    def __init__(self, CR: float, F: float, K: float = 0.5):
        super(DifferentialEvolutionCrossover, self).__init__(probability=1.0)
        self.CR = CR  # 交叉概率
        self.F = F    # 算子中的量化因子
        self.K = K

        self.current_individual: FloatSolution = None

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        """ Execute the differential evolution crossover ('best/1/bin' variant in jMetal).
        """
        if len(parents) != self.get_number_of_parents(): # 默认为3
            raise Exception('The number of parents is not {}: {}'.format(self.get_number_of_parents(), len(parents)))

        child = copy.deepcopy(self.current_individual)

        # 随机选取1个变量位置进行交叉（即使为满足交叉概率）
        number_of_variables = parents[0].number_of_variables
        rand = random.randint(0, number_of_variables - 1)

        for i in range(number_of_variables):
            if random.random() < self.CR or i == rand:
                value = parents[2].variables[i] + self.F * (parents[0].variables[i] - parents[1].variables[i])
                # 边界检查
                if value < child.lower_bound[i]:
                    value = child.lower_bound[i]
                if value > child.upper_bound[i]:
                    value = child.upper_bound[i]
            else:
                value = child.variables[i]

            child.variables[i] = value

        return [child]

    def get_number_of_parents(self) -> int:
        return 3

    def get_number_of_children(self) -> int:
        return 1

    def get_name(self) -> str:
        return 'Differential Evolution crossover'

class CompositeCrossover(Crossover[CompositeSolution, CompositeSolution]):
    __EPS = 1.0e-14

    def __init__(self, crossover_operator_list:[Crossover]):
        super(CompositeCrossover, self).__init__(probability=1.0)

        Check.is_not_none(crossover_operator_list)
        Check.collection_is_not_empty(crossover_operator_list)

        self.crossover_operators_list = []
        for operator in crossover_operator_list:
            Check.that(issubclass(operator.__class__, Crossover), "Object is not a subclass of Crossover")
            self.crossover_operators_list.append(operator)

    def execute(self, solutions: List[CompositeSolution]) -> List[CompositeSolution]:
        Check.is_not_none(solutions)
        Check.that(len(solutions) == 2, "The number of parents is not two: " + str(len(solutions)))

        offspring1 = []
        offspring2 = []

        number_of_solutions_in_composite_solution = solutions[0].number_of_variables

        for i in range(number_of_solutions_in_composite_solution):
            parents = [solutions[0].variables[i], solutions[1].variables[i]]
            children = self.crossover_operators_list[i].execute(parents)
            offspring1.append(children[0])
            offspring2.append(children[1])

        return [CompositeSolution(offspring1), CompositeSolution(offspring2)]

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'Composite crossover'

# MOEAD-M2M交叉
class MOEAD_M2M_Crossover(Crossover[FloatSolution, FloatSolution]):
    """
    该运算符接收两个参数：当前个体和一个父个体的数组，产生1个子代个体
    """

    def __init__(self, gen: int, MaxGen: int):
        super().__init__(probability=1.0)
        self.current_individual: FloatSolution = None
        self.gen = gen
        self.MaxGen = MaxGen

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        """ Execute the MOEAD-M2M crossover
        """
        if len(parents) != self.get_number_of_parents(): # 默认为2
            raise Exception('The number of parents is not {}: {}'.format(self.get_number_of_parents(), len(parents)))

        child = copy.deepcopy(self.current_individual)

        # 选取1个变量位置进行交叉
        number_of_variables = parents[0].number_of_variables

        rand = 2 * random.random() - 1
        rc = rand * (1 - random.random() ** (-(1 - self.gen / self.MaxGen) ** 0.7))
        for i in range(number_of_variables):
            if random.random() < self.probability:
                value = parents[1].variables[i] + rc * (parents[1].variables[i] - parents[0].variables[i])
                # 边界检查
                if value < child.lower_bound[i]:
                    value = child.lower_bound[i] + 0.5*random.random()*(parents[1].variables[i] - child.lower_bound[i])
                if value > child.upper_bound[i]:
                    value = child.upper_bound[i] - 0.5*random.random()*(child.upper_bound[i] - parents[1].variables[i])

            child.variables[i] = value

        return [child]

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 1

    def get_name(self) -> str:
        return 'MOEAD-M2M crossover'

#--------------------FJSP交叉操作---------------------#

class POX(Crossover[IntegerSolution, IntegerSolution]):
    '''OS vector'''
    def __init__(self, probability: float, parameters):
        super().__init__(probability=probability)
        self.parameters = parameters

    def execute(self, parents: List[IntegerSolution]) -> List[IntegerSolution]:
        Check.that(issubclass(type(parents[0]), IntegerSolution), "Solution type invalid")
        Check.that(issubclass(type(parents[1]), IntegerSolution), "Solution type invalid")
        Check.that(len(parents) == 2, 'The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        rand = random.random()
        if rand <= self.probability:
            length = len(parents[0].variables[0])
            J = self.parameters['jobs']
            jobNb = len(J)
            jobsRange = range(1, jobNb + 1)
            sizeJobset1 = random.randint(0, jobNb)

            jobset1 = random.sample(jobsRange, sizeJobset1) # 从工件集jobsRange中随机获取sizeJobset1个元素，作为一个列表返回

            p1 = offspring[0].variables[0] # 获取OS片段
            p2 = offspring[1].variables[0]

            o1 = []  # 子代1
            p1kept = []
            for i in range(length):
                e = p1[i]  # 获取基因
                if e in jobset1:  # 若属于集合J1
                    o1.append(e)  # 加入o1中
                else:
                    o1.append(-1)  # 加入-1
                    p1kept.append(e)  # 在p1kept中加入e

            o2 = []  # 子代2
            p2kept = []
            for i in range(length):
                e = p2[i]
                if e in jobset1:
                    o2.append(e)
                else:
                    o2.append(-1)
                    p2kept.append(e)

            for i in range(length):
                if o1[i] == -1:
                    o1[i] = p2kept.pop(0)  # 若o1对应基因为-1，则用p2kept[0]替换

            for i in range(length):
                if o2[i] == -1:
                    o2[i] = p1kept.pop(0)

            offspring[0].variables[0] = o1
            offspring[1].variables[0] = o2
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'POX crossover'

class JBX(Crossover[IntegerSolution, IntegerSolution]):
    '''OS vector'''
    def __init__(self, probability: float, parameters):
        super().__init__(probability=probability)
        self.parameters = parameters

    def execute(self, parents: List[IntegerSolution]) -> List[IntegerSolution]:
        Check.that(issubclass(type(parents[0]), IntegerSolution), "Solution type invalid")
        Check.that(issubclass(type(parents[1]), IntegerSolution), "Solution type invalid")
        Check.that(len(parents) == 2, 'The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        rand = random.random()
        if rand <= self.probability:
            length = len(parents[0].variables[0])
            J = self.parameters['jobs']
            jobNumber = len(J)
            jobsRange = range(0, jobNumber)
            sizeJobset1 = random.randint(0, jobNumber)

            jobset1 = random.sample(jobsRange, sizeJobset1)
            jobset2 = [item for item in jobsRange if item not in jobset1]

            p1 = offspring[0].variables[0]  # 获取OS片段
            p2 = offspring[1].variables[0]

            o1 = []
            p1kept = []
            for i in range(length):
                e = p1[i]
                if e in jobset1:
                    o1.append(e)
                    p1kept.append(e)
                else:
                    o1.append(-1)

            o2 = []
            p2kept = []
            for i in range(length):
                e = p2[i]
                if e in jobset2:
                    o2.append(e)
                    p2kept.append(e)
                else:
                    o2.append(-1)

            for i in range(length):
                if o1[i] == -1:
                    o1[i] = p2kept.pop(0)

            for i in range(length):
                if o2[i] == -1:
                    o2[i] = p1kept.pop(0)

            offspring[0].variables[0] = o1
            offspring[1].variables[0] = o2
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'JBX crossover'

class TwoPointCrossover(Crossover[IntegerSolution, IntegerSolution]):
    '''对MS进行交叉操作：TwoPointCrossover'''
    def __init__(self, probability: float, parameters):
        super().__init__(probability=probability)
        self.parameters = parameters

    def execute(self, parents: List[IntegerSolution]) -> List[IntegerSolution]:
        Check.that(issubclass(type(parents[0]), IntegerSolution), "Solution type invalid")
        Check.that(issubclass(type(parents[1]), IntegerSolution), "Solution type invalid")
        Check.that(len(parents) == 2, 'The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        rand = random.random()
        if rand <= self.probability:
            length = len(parents[0].variables[0])
            p1 = offspring[0].variables[1]  # 获取MS片段
            p2 = offspring[1].variables[1]

            pos1, pos2 = sorted(random.sample(range(length),2))

            offspring[0].variables[1] = p1[:pos1] + p2[pos1:pos2] + p1[pos2:]
            offspring[1].variables[1] = p2[:pos1] + p1[pos1:pos2] + p2[pos2:]

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'TwoPointCrossover crossover'

class WSCrossover_1(Crossover[IntegerSolution, IntegerSolution]):
    '''对WS进行交叉操作：TwoPointCrossover'''
    def __init__(self, probability: float, parameters):
        super().__init__(probability=probability)
        self.parameters = parameters

    def execute(self, parents: List[IntegerSolution]) -> List[IntegerSolution]:
        Check.that(issubclass(type(parents[0]), IntegerSolution), "Solution type invalid")
        Check.that(issubclass(type(parents[1]), IntegerSolution), "Solution type invalid")
        Check.that(len(parents) == 2, 'The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        rand = random.random()
        if rand <= self.probability:
            length = len(parents[0].variables[0])
            p1 = offspring[0].variables[2]  # 获取WS片段
            p2 = offspring[1].variables[2]

            pos1, pos2 = sorted(random.sample(range(length),2))

            offspring[0].variables[2] = p1[:pos1] + p2[pos1:pos2] + p1[pos2:]
            offspring[1].variables[2] = p2[:pos1] + p1[pos1:pos2] + p2[pos2:]

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'TwoPointCrossover crossover'

class FJSPCrossover(Crossover[IntegerSolution, IntegerSolution]):
    def __init__(self, probability: float, parameters):
        super().__init__(probability=probability)
        self.parameters = parameters

    def execute(self, parents: List[IntegerSolution]) -> List[IntegerSolution]:
        # OS变异部分
        if random.choice([True, False]):
            # 选取POX对OS vector执行交叉操作
            offspring = POX(self.probability, self.parameters).execute(parents)
        else:
            # 选取JBX对OS vector执行交叉操作
            offspring = JBX(self.probability, self.parameters).execute(parents)
        # 选取TPX对MS vector执行交叉操作
        offspring = TwoPointCrossover(self.probability, self.parameters).execute(offspring)
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'FJSPCrossover'

class Cooperative_search(Crossover[IntegerSolution, IntegerSolution]):
    '''Multi-objective optimization based on decomposition for flexible job
       shop scheduling under time-of-use electricity prices'''

    def __init__(self, probability: float, parameters):
        super().__init__(probability=probability)
        self.parameters = parameters

    def execute(self, parents: List[IntegerSolution]) -> List[IntegerSolution]:
        Check.that(issubclass(type(parents[0]), IntegerSolution), "Solution type invalid")
        Check.that(issubclass(type(parents[1]), IntegerSolution), "Solution type invalid")
        Check.that(len(parents) == 2, 'The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        rand = random.random()
        if rand <= self.probability:

            length = len(parents[0].variables[0])
            jobs = self.parameters["jobsnum"]
            ni = [0] * jobs
            for i in range(jobs):
                ni[i] = len(self.parameters['jobs'][i])
            # 随机选择一个工件
            jobNumber = random.randint(0, jobs-1)

            # 对OS片段进行操作
            os1 = offspring[0].variables[0]  # 获取OS片段
            os2 = offspring[1].variables[0]
            o1 = []
            o2 = []
            for i in range(length):
                if os1[i] == jobNumber:
                    o2.append(jobNumber)
                else:
                    o2.append(-1)
                if os2[i] == jobNumber:
                    o1.append(jobNumber)
                else:
                    o1.append(-1)

            oo2 = copy.copy(o2)
            for i in range(length):
                if (o2[i] == -1):
                    for j in range(length):
                        if o1[j] == -1:
                            o1[j] = -2
                            o2[i] = os2[j]
                            break
            for i in range(length):
                if (o1[i] == -2):
                    for j in range(length):
                        if oo2[j] == -1:
                            oo2[j] = -2
                            o1[i] = os1[j]
                            break

            # 对MS,WS片段进行操作
            ms1 = offspring[0].variables[1]  # 获取MS片段
            ms2 = offspring[1].variables[1]
            ws1 = offspring[0].variables[2]  # 获取WS片段
            ws2 = offspring[1].variables[2]
            m1 = copy.copy(ms1)
            m2 = copy.copy(ms2)
            w1 = copy.copy(ws1)
            w2 = copy.copy(ws2)
            for i in range(sum(ni[:jobNumber]),sum(ni[:jobNumber+1])):
                m2[i] = ms1[i]
                m1[i] = ms2[i]
                w2[i] = ws1[i]
                w1[i] = ws2[i]

            offspring[0].variables = [o1,m1,w1]
            offspring[1].variables = [o2,m2,w2]
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'Cooperative_search'

#--------------------FJSPLF交叉操作---------------------#

class FJSPLFCrossover(Crossover[IntegerSolution, IntegerSolution]):
    def __init__(self, probability: float, parameters):
        super().__init__(probability=probability)
        self.parameters = parameters

    def execute(self, parents: List[IntegerSolution]) -> List[IntegerSolution]:
        # OS变异部分
        if random.choice([True, False]):
            # 选取POX对OS vector执行交叉操作
            offspring = POX(self.probability, self.parameters).execute(parents)
        else:
            # 选取JBX对OS vector执行交叉操作
            offspring = JBX(self.probability, self.parameters).execute(parents)
        # 选取TPX对MS vector执行交叉操作
        offspring = TwoPointCrossover(self.probability, self.parameters).execute(offspring)
        # 选取TPX对WS vector执行交叉操作
        offspring = WSCrossover_1(self.probability, self.parameters).execute(offspring)
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'FJSPLFCrossover'


class MA2018Crossover(Crossover[IntegerSolution, IntegerSolution]):
    def __init__(self, probability: float, parameters):
        super().__init__(probability=probability)
        self.parameters = parameters

    def execute(self, parents: List[IntegerSolution]) -> List[IntegerSolution]:
        # OS变异部分
        if random.choice([True, False]):
            # 选取POX对OS vector执行交叉操作
            offspring = POX(self.probability, self.parameters).execute(parents)
        else:
            # 选取JBX对OS vector执行交叉操作
            offspring = JBX(self.probability, self.parameters).execute(parents)

        # 选取TPX对MS vector执行交叉操作
        offspring = TwoPointCrossover(self.probability, self.parameters).execute(offspring)
        # 选取TPX对WS vector执行交叉操作
        offspring = WSCrossover_1(self.probability, self.parameters).execute(offspring)
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'MA2018Crossover'

class FJSPLF_Compared_Crossover(Crossover[IntegerSolution, IntegerSolution]):
    def __init__(self, probability: float, parameters):
        super().__init__(probability=probability)
        self.parameters = parameters

    def execute(self, parents: List[IntegerSolution]) -> List[IntegerSolution]:
        # OS变异部分
        # 选取POX对OS vector执行交叉操作
        offspring = POX(self.probability, self.parameters).execute(parents)
        # 选取TPX对MS vector执行交叉操作
        offspring = TwoPointCrossover(self.probability, self.parameters).execute(offspring)
        # 选取TPX对WS vector执行交叉操作
        offspring = WSCrossover_1(self.probability, self.parameters).execute(offspring)
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'FJSPLFCrossover'


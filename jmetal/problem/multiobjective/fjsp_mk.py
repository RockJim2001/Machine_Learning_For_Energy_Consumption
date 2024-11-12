import math
import random
import re
import time
from collections import Counter

from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution

'''----------传统的FJSP问题（不考虑人员）----------'''

class FJSPMK(IntegerProblem):
    """ Class representing FJSP_MK Problem. """

    def __init__(self, instance: str = None):
        super().__init__()
        self.obj_directions = [self.MINIMIZE]
        self.number_of_variables = 0  # 无用
        self.number_of_objectives = 3
        self.number_of_constraints = 0
        self.machinesNb = 0  # 机器数
        self.jobsnNb = 0     # 工件数
        self.workerNb = 0    # 工人数
        self.jobs = []       # parameters['jobs']
        self.parameters = self.__read_from_file(instance)
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(self.number_of_objectives)]

    def __read_from_file(self, filename: str):

        if filename is None:
            raise FileNotFoundError('Filename can not be None')

        with open(filename) as file:
            firstLine = file.readline()
            firstLineValues = list(map(int, firstLine.split()[0:2]))
            self.jobsnNb = firstLineValues[0]  # 获取工件数
            self.machinesNb = firstLineValues[1]  # 获取机器数
            self.each_job_oper_num = [[] for _ in range(self.jobsnNb)]

            for i in range(self.jobsnNb):
                currentLine = file.readline()  # 读取每一行
                currentLineValues = list(map(int, currentLine.split()))  # 读取每一行转为整数
                job = []
                j = 1
                while j < len(currentLineValues):
                    operations = []  # 每个工序加工列表
                    path_num = currentLineValues[j]  # 获取本工序的加工路线个数
                    j = j + 1
                    for path in range(path_num):
                        machine = currentLineValues[j]  # 获取加工机器
                        j = j + 1
                        processingTime = currentLineValues[j]  # 获取加工时间
                        j = j + 1
                        operations.append({'machine': machine, 'processingTime': processingTime})
                    job.append(operations)
                self.jobs.append(job)
            file.close()
            info = {'VmNum': self.machinesNb, 'jobs': self.jobs, 'jobsnum': self.jobsnNb}
            return info

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:

        # 计算完工时间
        solution.objectives[0] = self.__makespan(solution.variables)
        # 最大机器负荷，总负荷。
        solution.objectives[1],solution.objectives[2] = self.__maxload(solution.variables)

        return solution

    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(lower_bound=[],upper_bound=[],number_of_objectives=self.number_of_objectives)
        OS = self.__generateOS()
        MS = self.__generateMS()
        new_solution.variables = [OS,MS]
        return new_solution

    '''----------------------编码解码-------------------------'''
    # 工序的编码
    def __generateOS(self):
        OS = []
        i = 0  # 工件索引从0开始
        for job in self.jobs:  # 获取工件数
            for op in job:  # 获取工件的工序数
                OS.append(i)
            i = i + 1
        random.shuffle(OS)
        return OS

    # 机器分配的编码
    def __generateMS(self):
        MS = []
        for job in self.jobs:
            for op in job:
                randomMachine = random.randint(0, len(op) - 1)  # 索引从0开始
                MS.append(randomMachine)
        return MS

    '''----------------------目标函数计算-------------------------'''
    # （柔性）计算最大完工时间
    def __makespan(self,solution):  # 个体（[os],[ms]）
        os, ms = solution
        decoded = self.__decode(os, ms)

        # 获取每台机器上最大完工时间
        max_per_machine = []
        for machine in decoded:
            max_d = 0
            for job in machine:  # job = （‘工件-工序’，加工时间，前序完工时间，最早开始时间）
                end = job[3] + job[1]
                if end > max_d:
                    max_d = end
            max_per_machine.append(max_d)
        makespan = max(max_per_machine)
        return makespan


    # 机器总负荷函数
    def __maxload(self, solution):

        os, ms = solution
        decoded = self.__decode(os, ms)
        mac = [0] * self.parameters['VmNum']  # 记录每台设备上的工作负荷
        for i in range(self.parameters['VmNum']):
            machine_info = decoded[i]
            for item in machine_info:
                mac[i] += item[1]
        sumload = sum(mac)
        maxload = max(mac)
        sumload = round(sumload, 2)
        return maxload, sumload

    '''----------------------解码-------------------------'''

    # 对个体进行解码，分配工件至机器。返回每台机器上加工任务
    def __decode(self, os, ms):
        o = self.jobs
        machine_operations = [[] for i in range(self.machinesNb)]  # [[机器1],[],[]..[机器n]]
        ms_s = self.__split_ms(ms)  # 每个工件的加工机器[[],[],[]..]
        Job_process = [0] * len(ms_s)  # len(ms_s）为工件数,储存第几工件加工第几工序
        Job_before = [0] * len(ms_s)  # 储存工件前一工序的完工时间

        # 对基于工序的编码进行依次解码，并安排相应的加工机器
        for job in os:
            index_machine = ms_s[job][Job_process[job]]  # 获取工件job的第Job_process[job]工序加工机器
            machine = o[job][Job_process[job]][index_machine]['machine'] - 1  # （工件，工序，机器序号）加工机器(索引重1开始)
            prcTime = o[job][Job_process[job]][index_machine]['processingTime']  # 加工时间

            start_cstr = Job_before[job]  # 前工序的加工时间
            # 能动解码
            start = self.__find_first_available_place(start_cstr, prcTime, machine_operations[machine])
            text = "{}-{}".format(job, Job_process[job])  # 工件-工序（索引均为0）

            # （‘工件-工序’，加工时间，前序完工时间，最早开始时间, 完工时间,工件号）
            machine_operations[machine].append((text, prcTime, start_cstr, start, start + prcTime, job))
            # 更新工件加工到第几工序
            Job_process[job] += 1
            Job_before[job] = (start + prcTime)
        return machine_operations  # [[(),(),()],[],[]]

    # 寻找最早可开始加工时间，返回可以最早开始加工的时间
    def __find_first_available_place(self, start_ctr, duration, machine_jobs):
        # 判断机器空闲是否可用
        def is_free(machine_used, start, duration):
            # machine_used 机器使用列表； start开始时间； duration加工时长
            for k in range(start, start + duration):
                if not machine_used[k]:  # 都为True可行，否则返回False
                    return False
            return True

        max_duration_list = []
        max_duration = start_ctr + duration

        # max_duration = start_ctr + duration 或 max(possible starts) + duration
        # 最长时间 = （前工序完工时间+time） 或 （机器可用时间+time）
        if machine_jobs:
            for job in machine_jobs:  # job --('0-1', 6, 0, 0)  （‘’，加工时间，开始时间，）
                max_duration_list.append(job[3] + job[1])

            max_duration = max(max(max_duration_list), start_ctr) + duration

        machine_used = [True] * max_duration  # machine_used  机器可用列表

        # 更新机器可用列表
        for job in machine_jobs:
            start = job[3]
            time = job[1]
            for k in range(start, start + time):
                machine_used[k] = False

        # 寻找满足约束的第一个可用位置
        for k in range(start_ctr, len(machine_used)):
            if is_free(machine_used, k, duration):
                return k  # 返回可以开始加工的位置

    # 分割基于机器分配的编码，划分为每个工件所需的机器
    def __split_ms(self, ms):
        jobs_machine = []  # 储存每个工件每个工序的加工机器(机器列表中位置)[[],[],[]]
        current = 0
        for index, job in enumerate(self.jobs):
            jobs_machine.append(ms[current:current + len(job)])
            current += len(job)
        return jobs_machine

    def get_name(self):
        return 'Traditonal FJSP'


if __name__ == "__main__":
    path = r'C:\Users\Hangyu\PycharmProjects\FJSPP\柔性作业车间调度\FJSP\FJSPBenchmark\Brandimarte_Data\Mk02.fjs'
    solution = FJSPMK(path).create_solution()
    time0 = time.time()
    for i in range(100000):
        FJSPMK(path).evaluate(solution)
        if i %1000 == 0:
            print(i)
    print(time.time() - time0)
    print((time.time() - time0)/100000)

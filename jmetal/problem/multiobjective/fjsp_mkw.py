import random
import time
import copy

from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution
from resources.FJSP_instance.mklf import LF
from jmetal.util.schedule_util import gantt
from jmetal.util.schedule_util.criticalpath import criticalpath,criticalpath_localsearch_lf,alternative_machine_worker,\
    movement_machine_worker,criticalpath_localsearch
from jmetal.util.schedule_util.idletime import find_idle_time

'''----------考虑工人柔性的FJSP问题----------'''


class FJSPMKW(IntegerProblem):

    def __init__(self, instance: str = None):
        super().__init__()
        self.obj_directions = [self.MINIMIZE]
        self.number_of_variables = 0  # 无用
        self.number_of_objectives = 3
        self.number_of_constraints = 0
        self.machinesNb = 0  # 机器数
        self.jobsnNb = 0    # 工件数
        self.workerNb = 0    # 工人数
        self.jobs = []       # parameters['jobs']
        self.parameters = self.__read_from_file(instance)
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(self.number_of_objectives)]
        self.cup = [LF.LF01(),LF.LF02(),LF.LF03(),LF.LF04(),LF.LF05(),LF.LF06(),
                    LF.LF07(),LF.LF08(),LF.LF09(),LF.LF10(),LF.LF11(),LF.LF12()]
        self.lf = self.cup[int(instance[-6:-4]) + 1]
        print(int(instance[-6:-4]) + 1)
        # self.lf = LF.LF01()
    def __read_from_file(self, filename: str):

        if filename is None:
            raise FileNotFoundError('Filename can not be None')

        with open(filename) as file:
            firstLine = file.readline()
            firstLineValues = list(map(int, firstLine.split()[0:2]))
            self.jobsnNb = firstLineValues[0]  # 获取工件数
            self.machinesNb = firstLineValues[1]  # 获取机器数
            self.workerNb = self.machinesNb       # 工人数与机器数一致
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
        # 计算总负荷和关键设备负荷
        solution.objectives[1],solution.objectives[2] = self.__maxload()

        return solution

    def create_solution(self,heuristic_method=False) -> IntegerSolution:
        new_solution = IntegerSolution(lower_bound=[],upper_bound=[],number_of_objectives=self.number_of_objectives)
        OS = self.__generateOS()
        MS = self.__generateMS()
        WS = self.__generateWS(MS)
        if heuristic_method:
            new_solution.variables = self.init_mwr(OS,MS,WS)
        else:
            if random.random() < 0.3:
                new_solution.variables = self.init_mwr(OS, MS, WS)
            else:
                new_solution.variables = [OS,MS,WS]
        return new_solution

    # 剩余加工时间最长的优先加工 (MWR most work remaining)
    def init_mwr(self,os, ms, ws):
        o = self.parameters['jobs']
        Job_process = [0] * self.parameters['jobsnum']  # 储存第几工件加工第几工序

        ni = []  # 存储每个工件的工序数
        for job in self.parameters['jobs']:
            ni.append(len(job))

        vecotr = [[] for i in range(len(self.parameters['jobs']))]

        # 对基于工序的编码进行依次解码，并安排相应的加工机器
        total_opr = len(os)
        for i in range(total_opr):
            job = os[i]
            opr = Job_process[job]
            index_machine = ms[sum(ni[:job]) + opr]  # 获取Oij的加工机器
            machine = o[job][opr][index_machine]['machine'] - 1  # （工件，工序，机器序号）加工机器(索引重1开始)
            people = ws[sum(ni[:job]) + opr]
            basicTime = o[job][opr][index_machine]['processingTime']  # 加工时间
            prcTime = basicTime  # * ratio # 员工操作速度
            vecotr[job].append(((job, opr), prcTime, index_machine, people))
            # 更新工件加工到第几工序
            Job_process[job] += 1

        os_vector = []
        ms_vector = [-1 for _ in range(total_opr)]
        ws_vector = [-1 for _ in range(total_opr)]

        for i in range(total_opr):
            block = None
            maxval = max(item[0][1] for item in vecotr)
            for j in range(self.parameters['jobsnum']):
                if vecotr[j][0][1] == maxval:
                    block = vecotr[j][0]
                    # print(block)
                    del vecotr[j][0]
                    vecotr[j].append((0, 0, 0, 0))
                    break

            job_num = block[0][0]  # 工件编码
            opr_num = block[0][1]  # 工序编码
            mac_num = block[2]  #
            people_num = block[3]
            os_vector.append(job_num)
            ms_vector[sum(ni[:job_num]) + opr_num] = mac_num
            ws_vector[sum(ni[:job_num]) + opr_num] = people_num

        return [os_vector, ms_vector, ws_vector]

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

    # 人员分配的编码
    def __generateWS(self, MS):
        # 为机器选择合适的员工,默认员工可操作所有机器
        WS = []
        for o in MS:
            workerlist = [i for i in range(self.machinesNb)]
            k = random.choice(workerlist)
            WS.append(k)
        return WS

    '''----------------------目标函数计算-------------------------'''
    # 最大完工时间
    def __makespan(self,solution):  # 个体（[os],[ms]）
        self.decoded = self.decode_lf(solution)
        # 获取每台机器上最大完工时间
        max_per_machine = []
        for machine in self.decoded:
            max_d = 0
            for job in machine:  # job = （‘工件-工序’，加工时间，前序完工时间，最早开始时间）
                end = job['startTime'] + job["processingTime"]
                if end > max_d:
                    max_d = end
            max_per_machine.append(max_d)
        makespan = max(max_per_machine)
        return round(makespan,2)

    def get_objects(self,solution):
        makespan = self.__makespan(solution)
        maxload,sumload = self.__maxload()

        return makespan,maxload,sumload

    # 机器最大负荷,和总机器负荷
    def __maxload(self):
        '''
        record = {"operation": text, "processingTime": processTime, "Cij-1": start_cstr, "startTime": start, \
                  "endTime": start + processTime, "job": job, "machine": machine, "people": people}
        machine_operations[machine].append(record)
        '''
        mac = [0]*self.machinesNb  # 记录每台设备上的工作负荷
        info = self.decoded
        for i in range(self.machinesNb):
            machine_info = info[i]
            for item in machine_info:
                mac[i] += item["processingTime"]
        maxload = max(mac)
        sumload = sum(mac)
        return round(maxload,2), round(sumload,2)  # 最大机器负荷,总负荷

    '''----------------------解码-------------------------'''

    # 寻找最早可开始加工时间，返回可以最早开始加工的时间
    def __find_first_available_place(self, start_ctr, processingTime, machine_jobs, people_jobs):
        '''
        start_ctr:前工序完工时间  processingTime：加工时间  machine_jobs：机器所加工的工序  people_jobs：人员所加工的工序
        '''
        # 判断机器人员空闲是否可用
        def is_free(idleStart, idleEnd, processingTime):
            idleStart = max(idleStart,start_ctr) # 空闲时间 和 前序完工时间的最大值
            return idleStart + processingTime <= idleEnd

        def find_idle_time(list1, list2):
            '''寻找list1和list2共同的空闲时间'''
            # 先按第一个数排序，再按第二个排序
            List = sorted(list1 + list2, key=lambda x: (x[0], x[1]))
            # b_idx = 0
            # b_end = len(b)
            i = 0
            while i < len(List) - 1:
                if List[i][0] <= List[i + 1][0] <= List[i][1]:
                    new = (min(List[i][0], List[i + 1][0]), max(List[i][1], List[i + 1][1]))
                    List.pop(i)
                    List.pop(i)
                    List.insert(i, new)
                else:
                    i += 1

            idle = []  # 存储所有的空闲时间
            for i in range(len(List) - 1):
                idle.append((List[i][1], List[i + 1][0]))

            if len(List)==0:
                return idle
            else:
                idle.append((List[-1][1], 1e10))
                return idle

        # 机器工作列表
        machine_work_time = [[job["startTime"],job["endTime"]] for job in machine_jobs]
        # 工人工作列表
        people_work_time = [[job["startTime"],job["endTime"]] for job in people_jobs]
        # 共同空闲时间
        idle_time = find_idle_time(machine_work_time,people_work_time)

        # 可插入，则直接选择最早可用时间
        if idle_time != []:
            for (idleStart,idleEnd) in idle_time:
                if is_free(idleStart, idleEnd, processingTime):
                    return max(idleStart,start_ctr)
        elif machine_work_time==[] and people_work_time==[]:
            return 0

    # 对个体进行解码，分配工件至机器。返回每台机器上加工任务(考虑学习遗忘效应)
    def decode_lf(self, solution):
        os, ms, ws = solution
        o = self.jobs
        machine_operations = [[] for i in range(self.machinesNb)]  # [[机器1],[],[]..[机器n]]
        people_operations = [[] for i in range(self.machinesNb)]
        Job_process = [0] * self.jobsnNb  # 储存第几工件加工第几工序
        Job_before = [0] * self.jobsnNb   # 储存工件前一工序的完工时间
        Machine_before = [0] * self.machinesNb  # 储存设备前一工序的完工时间
        People_before = [0] * self.machinesNb   # 储存工人前一工序的完工时间
        # 记录员工在每台设备上加工的工件数，用于计算学习效应(行为工人，列为在每台设备上的累计加工工件数目)
        processed_job_count = [[1 for _ in range(self.machinesNb)] for _ in range(self.workerNb)]

        # 记录上一个机器加工的工件序号
        macjob_before_record = [[None] for _ in range(self.machinesNb)]

        ni = []  # 存储每个工件的工序数
        for job in self.jobs:
            ni.append(len(job))

        # 对基于工序的编码进行依次解码，并安排相应的加工机器
        for i in range(len(os)):
            job = os[i]
            opr = Job_process[job]
            index_machine = ms[sum(ni[:job]) + opr]  # 获取Oij的加工机器索引(索引重0开始)
            machine = o[job][opr][index_machine]['machine'] - 1     # （工件，工序，机器序号）加工机器(索引重1开始)
            people = ws[sum(ni[:job]) + opr]         # 获取Oij的执行工人(从0开始)
            # print(job,opr,machine,people)
            prcTime = o[job][opr][index_machine]['processingTime']  # 基本加工时间
            e_kw = self.lf.efficency[people][machine]      # 获取工人w操纵设备k的初始技能水平
            a_kw = self.lf.learning_rate[people][machine]  # 获取工人w操纵设备k的学习能力
            R_kw = processed_job_count[people][machine] # 当前工人w在设备k上的累计加工工件
            # (考虑学习效应的加工时间)实际加工时间，保留两位小数
            # processTime = round(prcTime*max(e_kw * (self.lf.M + (1 - self.lf.M) * R_kw ** a_kw), self.lf.theta),2)
            # (不考虑学习效应的加工时间)
            processTime = o[job][Job_process[job]][index_machine]['processingTime']  # 加工时间
            #print("考虑LF前:",prcTime,"\t考虑LF后",processTime)
            if macjob_before_record[machine] != [None]:
                jobbefore = macjob_before_record[machine]
                similarity = self.lf.delta[jobbefore][job]
                # print(people,machine,similarity)
                processed_job_count[people][machine] += similarity

            # 方法1：传统判断最早开工时间
            start_cstr = Job_before[job]  # 前工序的加工时间
            #start_machine = Machine_before[machine]
            #before_finishtime = People_before[people]
            #start = max(start_cstr, start_machine, before_finishtime)

            # 方法2：能动解码方式
            start = self.__find_first_available_place(start_cstr, prcTime, machine_operations[machine],people_operations[people])

            text = "{}-{}-{}-{}".format(job, opr, machine, people)  # 工件-工序（索引均为0）
            # （‘工件-工序’，加工时间，前序完工时间，最早开始时间, 完工时间,机器号，人员号）
            record = {"operation": text, "processingTime": processTime, "Cij-1": start_cstr, "startTime": start,\
                      "endTime": start + processTime, "job": job, "machine": machine, "people": people}
            machine_operations[machine].append(record)
            people_operations[people].append(record)

            # 更新工序
            Job_process[job] += 1
            # 加工时间
            Job_before[job] = start + processTime
            Machine_before[machine] = start + processTime
            People_before[people] = start + processTime
            # 更新机器加工工件记录
            macjob_before_record[machine] = job

        return machine_operations  # [[(),(),()],[],[]]

    '''--------------------局部搜索策略-------------------------'''

    # 基于关键路径的局部搜索方法
    def critcal_based_search(self,solution):
        keyPath,machineTime,peopleTime = criticalpath(self.parameters, self.decode_lf(solution))
        new_chrom = criticalpath_localsearch(keyOpe, solution, self.parameters, machineTime, peopleTime, m=3)
        #new_chrom = criticalpath_localsearch_lf(keyPath, solution, self.parameters, machineTime,peopleTime,self.lf.efficency)
        return new_chrom

    # Pathlinked 局部搜索策略,返回两个邻域解
    def pathlinked_based_search(self,nondominated, representive):
        '''
        :param nondominated: 非支配解，在Pareto前沿上的解
        :param representive: 代表解，选中的要执行局部搜索的解
        :return: 返回局部搜索途中的 2个领域解
        '''
        (os1, ms1, ws1) = nondominated
        (os2, ms2, ws2) = representive
        length = len(os1)
        idxset = [i for i in range(length)]
        random.shuffle(idxset)
        idx1 = idxset[:length // 3]
        idx2 = idxset[length // 3:2 * (length // 3)]

        off1 = copy.copy(representive)
        off2 = copy.copy(representive)

        # OS片段
        for i in range(length // 3):
            if os1[i] != os2[i]:
                loc = os2.index(os1[i])
                os2[i], os2[loc] = os2[loc], os2[i]
        off1[0] = os2

        for i in range(length // 3, 2 * (length // 3)):
            if os1[i] != os2[i]:
                loc = os2.index(os1[i])
                os2[i], os2[loc] = os2[loc], os2[i]
        off2[0] = os2

        # MS,WS片段
        for idx in idx1:
            ws2[idx] = ws1[idx]
            ms2[idx] = ms1[idx]
        off1[1] = ms2
        off1[2] = ws2

        for idx in idx2:
            ws2[idx] = ws1[idx]
            ms2[idx] = ms1[idx]
        off2[1] = ms2
        off2[2] = ws2

        return off1, off2


    '''--------------------甘特图绘制-------------------------'''

    # 绘制甘特图时使用
    def translate_decoded_to_gantt(self,machine_operations):
        data = {}
        # 用于统计完工时间
        makespan = []
        for idx, machine in enumerate(machine_operations):
            machine_name = "Machine-{}".format(idx + 1)
            operations = []
            for operation in machine:
                starttime = operation["startTime"]
                endtime = operation["startTime"] + operation["processingTime"]
                label = operation["operation"]
                operations.append([starttime, endtime, label])
                makespan.append(endtime)
            data[machine_name] = operations
        print("Makespan:",max(makespan))
        return data

    def get_name(self):
        return 'FJSPMKW'

if __name__ == "__main__":
    path = r'C:\Users\Hangyu\PycharmProjects\FJSPP\柔性作业车间调度\FJSP\FJSPBenchmark\Brandimarte_Data\Mk01.fjs'
    model = FJSPMKW(path)
    individual = (
    [8, 7, 0, 8, 9, 2, 0, 4, 6, 2, 1, 6, 4, 8, 5, 1, 5, 7, 4, 9, 7, 9, 6, 4, 6, 5, 4, 8, 2, 0, 1, 1, 5, 0, 7, 2, 5, 1,
     8, 5, 3, 9, 4, 3, 2, 0, 3, 6, 3, 8, 0, 7, 9, 3, 9],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0, 2, 0, 1, 0, 0, 0, 2, 1, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 1, 2, 2, 0,
     1, 2, 1, 0, 0, 0, 1, 1, 0, 2, 1, 0, 1, 0, 0, 1, 1],
    [1, 1, 0, 5, 1, 1, 3, 1, 3, 4, 2, 0, 0, 3, 0, 3, 3, 2, 0, 0, 1, 2, 4, 4, 5, 4, 4, 5, 3, 0, 2, 4, 0, 0, 5, 2, 4, 2,
     1, 0, 3, 2, 5, 1, 2, 5, 3, 5, 4, 5, 2, 0, 4, 0, 4])

    # makespan,maxload,totalload = model.get_objects(individual)
    # print('完工时间为:',makespan,maxload,totalload)

    #测试目标函数评价时间 (运行10万次,20.2s)
    solution = model.create_solution()
    time0 = time.time()
    for i in range(100000):
        model.evaluate(solution)
        if i %1000 == 0:
            print(i)
    print(time.time() - time0)


    # # 关键路径 (运行10万次,22s)
    # start = time.time()
    # for k in range(10):
    #     keyPath = criticalpath(model.parameters, model.decode(individual))
    # print(time.time() - start)
    # print(keyPath)

    # gantt_data = model.translate_decoded_to_gantt(model.decode(individual))
    # print(gantt_data)
    # title = "Dual resource-constrained Flexible Job Shop Solution"  # 甘特图title
    # gantt.draw_chart(gantt_data, title, 15)
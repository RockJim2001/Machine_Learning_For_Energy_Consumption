import time
import random
import copy

#from jmetal.problem import FJSPLF
from jmetal.util.schedule_util import gantt
from jmetal.util.schedule_util import idletime

def criticalpath(parameters, decoded):
    '''计算关键路径'''
    # [('6-0', 14.4, 0, 0, 14.4, 6), ('7-1', 18.6, 23.9, 23.9, 42.5, 7)]
    # 计算关键路径
    # 1 求得最大时间所在的工序编号
    maxIndex = ""  # 用来储存最大完工时间的索引
    keyPath = []   # 储存在关键路径上的工序序号
    critical_ope = []  # 储存关键路径上的工序信息
    Cmax = -1      # 计算完工时间

    n = parameters['jobs']
    m = parameters['VmNum']
    startTime = [[0]*len(n[i]) for i in range(len(n))]     # 每道工序的开工时间
    machineTime = [[0]*len(decoded[j]) for j in range(m)]  # 每个机器各操作的开工时间
    peoplecount0 = [0] * m    #(机器数和工人数一致)            # 每个人员操作的开工时间

    for i in range(m):
        for j in range(len(decoded[i])):
            info = decoded[i][j]
            peoplecount0[info["people"]] += 1  # 统计每人的工作负荷
    peopleTime = [[0] * peoplecount0[j] for j in range(m)]  # 每个员工各操作的开工时间

    peoplecount1 = [0] * m
    # 对m台机器循环：
    for i in range(m):
        #（‘工件-工序’，加工时间，前序完工时间，最早开始时间, 完工时间,工件号）
        # machine_operations[machine].append(record)
        # record = {"operation": text, "processingTime": prcTime, "Cij-1": start_cstr, "startTime": start,\
                          #  "endTime": start + prcTime, "job": job, "machine": machine, "people": people}
        for j in range(len(decoded[i])):
            info = decoded[i][j] # 每个工件的信息
            o_start = info["startTime"]  # 开工时间
            o_end = info["endTime"]  # 完工时间
            machineTime[i][j] = (o_start, o_end, info["operation"])   # (开工时间，完工时间，“工件-工序”)
            # 获取工件和工序编号
            job = int(info["operation"].split("-",2)[0])
            op = int(info["operation"].split("-",2)[1])
            startTime[job][op] = (o_start, o_end, info["operation"])   # (开工时间,完工时间，工序信息）
            peopleTime[info["people"]][peoplecount1[info["people"]]] = (o_start, o_end, info["operation"]) # (开工时间,完工时间，工序信息）
            peoplecount1[info["people"]] += 1
            # 查找每个工件最后工序的完工
            if o_end >= Cmax:
                maxIndex = info["operation"]  # 返回最后完工的工序信息
                Cmax = o_end

    # 进行排序
    for i in range(m):
        peopleTime[i] = sorted(peopleTime[i],key=lambda x: x[0])
        machineTime[i] = sorted(machineTime[i],key=lambda x: x[0])
    # <<<1.2将这个工序加入关键路径链表中
    keyPath.append(maxIndex)

    index_i = int(maxIndex.split("-", 3)[0])  # 工件号
    index_j = int(maxIndex.split("-", 3)[1])  # 工序号
    first_k = int(maxIndex.split("-", 3)[2])  # 机器号
    first_w = int(maxIndex.split("-", 3)[3])  # 工人号
    _time = int(startTime[index_i][index_j][1]) - int(startTime[index_i][index_j][0]) # 加工时间
    c_PJ = int(startTime[index_i][index_j-1][1])
    try:
        c_PM = int(machineTime[first_k][-2][1])
        c_PW = int(peopleTime[first_w][-2][1])
    except:
        # print("报错了一次")
        # print("machineTime",machineTime[first_k])
        # print("peopleTime",peopleTime[first_w])
        c_PM = int(machineTime[first_k][len(machineTime[first_k])-2][1])
        c_PW = int(peopleTime[first_w][len(peopleTime[first_w])-2][1])

    # 可用人员设备集合
    machineset = []
    timeset = []
    for item in n[index_i][index_j]:
        machineset.append(item["machine"])
        timeset.append(item['processingTime'])
    peopleset = [i for i in range(m)]
    firstOpe = Key_operation(index_i, index_j, first_k, first_w, _time,c_PJ, 10000,\
                             c_PM, 10000, c_PW, 10000, machineset, peopleset, timeset)
    critical_ope.append(firstOpe)


    # step2,开始递推寻找，直到首工序
    while startTime[index_i][index_j][0] != 0:  # 直到找开工时间为0的工序才停止
        preJobIndex = -1  # 初始化 前一工件的索引
        preMacIndex = -1  # 初始化 前一机器的索引
        prePeoIndex = -1  # 初始化 前一机器的索引

        # 寻找preJob(operation), preMac(operation)
        # preJob(operation)： 同一工件上，工序o的前一工序
        # preMac(operation):  同一机器上，工序o前面紧挨的工序
        # prePeo(opeartion):  前一个工人加工的工序

        nowtime = startTime[index_i][index_j][0]

        # 情况1（找preJob(operation)） (非同台机器上)存在前序工序,且结束时间等于当前工序的开始时间,否则 preJobIndex = -1
        if (index_j != -1 and startTime[index_i][index_j-1][1] == nowtime):
            now_ope = index_j-1
            # 找到该工件前工序索引
            machine, people = int(startTime[index_i][now_ope][2].split("-",3)[2]),int(startTime[index_i][now_ope][2].split("-",3)[3])

            preJobIndex = str(index_i) + "-" + str(now_ope) + "-" + str(machine) + "-" + str(people)

            # <<< 定义关键路径上的工序类
            _time = startTime[index_i][now_ope][1] - startTime[index_i][now_ope][0] # Oi(j-1)加工时间
            # 同一个工件上
            c_PJ = 0 if now_ope==0 else startTime[index_i][now_ope-1][1]  # 前工序完工时间OK
            s_SJ = 10000 if now_ope+1 == len(n[index_i])-1 else startTime[index_i][now_ope + 1][0]  # 后工序开始时间

            # 同一个设备上
            for i in range(len(machineTime[machine])):
                if machineTime[machine][i][2] == preJobIndex:
                    c_PM = 0 if i==0 else machineTime[machine][i-1][1]  # 前设备完工时间
                    s_SM = 10000 if i == len(decoded[machine])-1 else machineTime[machine][i + 1][0] # 后设备开始时间
                    #break

            # 同一个员工上
            for i in range(len(peopleTime[people])):
                if peopleTime[people][i][2] == preJobIndex:
                    c_PW =  0 if i==0 else peopleTime[people][i-1][1]  # 前工人完工时间
                    s_SW = 10000 if i == len(peopleTime[people])-1 else peopleTime[people][i + 1][0] # 后工人开始时间
                    #break

            # 可用人员设备集合
            machineset = []
            timeset = []
            for item in n[index_i][now_ope]:
                machineset.append(item["machine"])
                timeset.append(item['processingTime'])
            peopleset = [i for i in range(m)]
            JobOpe = Key_operation(index_i,now_ope,machine,people,_time,c_PJ,s_SJ,c_PM,s_SM,c_PW,s_SW,machineset,peopleset,timeset)
            # >>>

        # 情况2（找preMac(o)）（同台机器上）找到加工该工件上道工序的 机器序号，返回preMacIndex,否则返回-1
        machine = int(startTime[index_i][index_j][2].split("-",3)[2])  # 加工当前工序的机器
        # 已知机器编号machine，找到在这台机器上完工时间=开工时间的工件（是第几工序）

        # 找到与工件在同台机器上加工的前继工件(前继完工时间=该工序开工时间)编号，满足条件说明两工序直接相连
        for i in range(len(machineTime[machine])):
            if (index_j != -1 and machineTime[machine][i][1] == nowtime):
                macOpe_i = int(machineTime[machine][i][2].split("-", 3)[0])
                macOpe_j = int(machineTime[machine][i][2].split("-", 3)[1])
                macOpe_k = int(machineTime[machine][i][2].split("-", 3)[2])
                macOpe_w = int(machineTime[machine][i][2].split("-", 3)[3])
                preMacIndex = str(macOpe_i) + "-" + str(macOpe_j) + "-" + str(macOpe_k) + "-" + str(macOpe_w)
                # <<< 定义关键路径上的工序类
                _time = startTime[macOpe_i][macOpe_j][1] - startTime[macOpe_i][macOpe_j][0]
                # 同一个工件上
                c_PJ = 0 if macOpe_j == 0 else startTime[macOpe_i][macOpe_j-1][1]  # 前工序完工时间OK
                s_SJ = 10000 if macOpe_j == len(n[macOpe_i]) - 1 else startTime[macOpe_i][macOpe_j+1][0]  # 后工序开始时间
                # 同一个机器上
                c_PM = 0 if i==0 else machineTime[machine][i-1][1]
                s_SM = 10000 if i == len(decoded[machine]) - 1 else machineTime[machine][i + 1][0]  # 后设备开始时间
                # 同一个员工
                for j in range(len(peopleTime[macOpe_w])):
                    if peopleTime[macOpe_w][j][2] == preMacIndex:
                        c_PW = 0 if j == 0 else peopleTime[macOpe_w][j - 1][1]  # 前工人完工时间
                        s_SW = 10000 if j == len(peopleTime[macOpe_w]) - 1 else peopleTime[macOpe_w][j + 1][0]  # 后工人开始时间
                # 可用人员设备集合
                machineset = []
                timeset = []
                for item in n[macOpe_i][macOpe_j]:
                    machineset.append(item["machine"])
                    timeset.append(item['processingTime'])
                peopleset = [i for i in range(m)]
                MacOpe = Key_operation(macOpe_i,macOpe_j,macOpe_k,macOpe_w,_time,c_PJ,s_SJ,c_PM,s_SM,c_PW,s_SW,machineset,peopleset,timeset)
                # >>>
                break

        # 情况3 找到操作员工，返回prePeoIndex,否则返回-1
        people = int(startTime[index_i][index_j][2].split("-",3)[3])  # 加工当前工序的机器
        # 已知机器编号machine，找到在这台机器上完工时间=开工时间的工件（是第几工序）
        # 找到与工件在同台机器上加工的前继工件(前继完工时间=该工序开工时间)编号，满足条件说明两工序直接相连
        for i in range(len(peopleTime[people])):
            if (index_j != -1 and peopleTime[people][i][1] == nowtime):
                peoOpe_i = int(peopleTime[people][i][2].split("-", 3)[0])
                peoOpe_j = int(peopleTime[people][i][2].split("-", 3)[1])
                peoOpe_k = int(peopleTime[people][i][2].split("-", 3)[2])
                peoOpe_w = int(peopleTime[people][i][2].split("-", 3)[3])
                prePeoIndex = str(peoOpe_i) + "-" + str(peoOpe_j) + "-" + str(peoOpe_k) + "-" + str(peoOpe_w)
                # 定义类
                _time = startTime[peoOpe_i][peoOpe_j][1] - startTime[peoOpe_i][peoOpe_j][0]
                # 同一个工件上
                c_PJ = 0 if peoOpe_j == 0 else startTime[peoOpe_i][peoOpe_j - 1][1]  # 前工序完工时间OK
                s_SJ = 10000 if peoOpe_j == len(n[peoOpe_i]) - 1 else startTime[peoOpe_i][peoOpe_j + 1][0]  # 后工序开始时间
                # 同一个机器上
                for j in range(len(machineTime[peoOpe_k])):
                    if machineTime[peoOpe_k][j][2] == prePeoIndex:
                        c_PM = 0 if j == 0 else machineTime[peoOpe_k][j - 1][1]
                        s_SM = 10000 if j == len(decoded[peoOpe_k]) - 1 else machineTime[peoOpe_k][j + 1][0]  # 后设备开始时间
                # 同一个工人上
                c_PW = 0 if i==0 else peopleTime[peoOpe_w][i-1][1]
                s_SW = 10000 if i==len(peopleTime[peoOpe_w])-1 else peopleTime[peoOpe_w][i+1][0]
                # 可用人员设备集合
                machineset = []
                timeset = []
                for item in n[peoOpe_i][peoOpe_j]:
                    machineset.append(item["machine"])
                    timeset.append(item['processingTime'])
                peopleset = [i for i in range(m)]
                PeoOpe = Key_operation(peoOpe_i,peoOpe_j,peoOpe_k,peoOpe_w,_time,c_PJ,s_SJ,c_PM,s_SM,c_PW,s_SW,machineset,peopleset,timeset)
                break


        # 情况4 （情况1-3有多个满足）
        # 先安排同一工件、再安排同一机器、再安排同一加工人员

        # 同时存在时，取工件上的
        if (preJobIndex != -1 and preMacIndex != -1 and prePeoIndex != -1):
            # if Jm[index_i][index_j] != Jm[macOpe_i][macOpe_j]:
            keyPath.append(preJobIndex)
            critical_ope.append(JobOpe)
            index_j -= 1  # 工序减一
        # 机器，工件 ——> 工件
        elif preJobIndex != -1 and preMacIndex != -1 :
            keyPath.append(preJobIndex)
            critical_ope.append(JobOpe)
            index_j -= 1  # 工序减一
        # 人员，工件 ——> 工件
        elif preJobIndex != -1 and prePeoIndex != -1 :
            keyPath.append(preJobIndex)
            critical_ope.append(JobOpe)
            index_j -= 1  # 工序减一
        # 人员，机器 ——> 机器
        elif preMacIndex != -1 and prePeoIndex != -1:
            keyPath.append(preMacIndex)
            critical_ope.append(MacOpe)
            index_i = macOpe_i
            index_j = macOpe_j
        # 只存在符合要求的(同一机器上)工件前序工序
        elif preJobIndex != -1:
            keyPath.append(preJobIndex)
            critical_ope.append(JobOpe)
            index_j -= 1
        # 只存在符合要求的机器前序工序
        elif preMacIndex != -1:
            keyPath.append(preMacIndex)
            critical_ope.append(MacOpe)
            index_i = macOpe_i
            index_j = macOpe_j
        # 人
        elif prePeoIndex != -1:
            keyPath.append(prePeoIndex)
            critical_ope.append(PeoOpe)
            index_i = peoOpe_i
            index_j = peoOpe_j
    # keyPath的结构
    # ['8-0-5-0', '9-0-5-5', '9-1-2-0', '0-0-2-1', '5-2-2-2', '5-3-1-0', '3-1-1-0', '4-2-1-4', '2-2-1-4', '7-2-1-2',
    #  '0-3-5-2', '6-3-4-2', '9-3-5-2', '2-3-5-3', '9-4-3-3', '4-4-3-3', '8-4-5-3', '8-5-3-1', '7-4-3-4', '0-5-3-3', '9-5-3-0']
    keyPath.reverse()
    critical_ope.reverse()
    #print(critical_ope[-1].info())
    return critical_ope,machineTime,peopleTime  # 关键路径

def criticalpath_localsearch(keyOpe,chromosome, parameters,machineTime,peopleTime,m=3):
    '''基于关键路径的局部搜索策略'''
    # m : 对m个基因位进行局部搜索
    new_solution = copy.deepcopy(chromosome)  # 邻域解
    ni = [0] * parameters["jobsnum"]
    for i in range(parameters["jobsnum"]):
        ni[i] = len(parameters['jobs'][i])
    #keyOpe = Criticalpath(parameters, decoded)
    # 因为第一个关键工序不能再提前，所以无需考虑
    for operation in keyOpe[1:]:
        job = operation.jobidx  # 工件序号
        ope = operation.opeidx  # 工序序号
        loc = sum(ni[:job])+ ope  # 对应Oij的工序在ms和ws上的位置
        mac = operation.machine  # 机器序号
        peo = operation.people  # 员工序号


        machineset = copy.copy(operation.machineset)
        machineset.pop(machineset.index(mac+1))  # machineset元素从1开始
        random.shuffle(machineset)

        peopleset = copy.copy(operation.peopleset)
        peopleset.pop(peopleset.index(peo))      # peopleset元素从0开始
        random.shuffle(peopleset)

        # 如果有一个为空集，则对下一个工序进行搜索
        if machineset==[] or peopleset==[]:
            continue

        # 若不为空，则更换加工机器和操作人员
        #Ms = machineset if len(machineset)<=m else machineset[:m]  # 可用设备集合
        #Ps = peopleset if len(peopleset)<=m else peopleset[:m]  # 可用人员集合
        Ms = machineset
        Ps = peopleset
        for i in range(len(Ms)):
            for j in range(len(Ps)):
                alter_machine = Ms[i]
                alter_people = Ps[j]
                # 找到设备和人员的可用时间集合
                mac_work = [item[:2] for item in machineTime[alter_machine-1]]  # alter_machine 索引为1
                peo_work = [item[:2] for item in peopleTime[alter_people]]
                idle = idletime.find_idle_time(mac_work,peo_work)  # [(1, 3), (22, 23), (30, 33), (47, 56), (57, 68), (70, 87)]
                # 机器序号
                processingTime = parameters['jobs'][job][ope][i]['processingTime']
                #actual_processingTime = operation.processingTime*
                for (begin,end) in idle:
                    # 判断是否满足插入准则
                    flag = max(begin,operation.c_PJij) + processingTime <= min(end,operation.s_SJij)
                    if flag:  # 可以执行局部搜索操作
                        # print("+++++++",flag)
                        # print(new_solution[0])
                        # print(job,ope,mac,peo)
                        # print("机器：",alter_machine, Ms)  #  机器空闲时间
                        # print("人员：",alter_people, Ps)  #  员工空闲时间
                        # #print('空闲时间：',idle) # 机器和员工共同空闲时间
                        # print(k,j)
                        # print("最早开始时间:{} \t 加工时间:{}\t最迟结束时间:{}".format(max(begin, operation.c_PJij), processingTime, min(end, operation.s_SJij)))
                        # print('替换后', alter_machine-1,alter_people)
                        new_solution[1][loc] = operation.machineset.index(alter_machine)
                        new_solution[2][loc] = operation.peopleset.index(alter_people)
                        # print("局部搜索前", job, ope, mac, peo)
                        # print("局部搜索后",job, ope, alter_machine-1,alter_people)
                        # 搜索到更好的解，则返回更好的解
                        #todo 应该循环到不能找到更好的解为止
                        return new_solution
    # 没有搜索到更好的解，则返回原始解
    return chromosome

def criticalpath_localsearch_lf(keyOpe,chromosome,parameters,machineTime,peopleTime,efficency,m=3):
    '''基于关键路径的局部搜索策略'''
    # m : 对m个基因位进行局部搜索
    new_solution = copy.deepcopy(chromosome)  # 邻域解
    ni = [0] * parameters["jobsnum"]
    for i in range(parameters["jobsnum"]):
        ni[i] = len(parameters['jobs'][i])
    #keyOpe = Criticalpath(parameters, decoded)
    # 因为第一个关键工序不能再提前，所以无需考虑
    for operation in keyOpe[1:]:
        job = operation.jobidx  # 工件序号
        ope = operation.opeidx  # 工序序号
        loc = sum(ni[:job])+ ope  # 对应Oij的工序在ms和ws上的位置
        mac = operation.machine  # 机器序号
        peo = operation.people  # 员工序号
        # 最多对3*3=9个基因位进行邻域搜索
        machineset = copy.copy(operation.machineset)
        machineset.pop(machineset.index(mac+1))
        random.shuffle(machineset)

        operationset = copy.copy(operation.peopleset)
        operationset.pop(operationset.index(peo))
        random.shuffle(operationset)
        # 如果有一个为空集，则对下一个工序进行搜索
        if machineset==[] or operationset==[]:
            continue
        # 若不为空，则更换加工机器和操作人员
        Ms = machineset if len(machineset)<=m else machineset[:m]  # 可用设备集合 索引为1
        Os = operationset if len(operationset)<=m else operationset[:m]  # 可用人员集合
        for alter_machine in Ms:
            for alter_people in Os:
                # 找到设备和人员的可用时间集合
                mac_work = [item[:2] for item in machineTime[alter_machine-1]]  # alter_machine 索引为1
                peo_work = [item[:2] for item in peopleTime[alter_people]]
                idle = idletime.find_idle_time(mac_work,peo_work)
                e_kw = efficency[alter_people][alter_machine-1]  # 获取工人w操纵设备k的初始技能水平
                actual_processingTime = operation.processingTime*e_kw
                for (begin,end) in idle:
                    # 判断是否满足插入准则
                    flag = max(begin,operation.c_PJij) + actual_processingTime <= min(end,operation.s_SJij)
                    if flag:  # 可以执行局部搜索操作
                        # print("+++++++",flag)
                        # print("机器：",alter_machine, mac_work)  #  机器空闲时间
                        # print("人员：",alter_people, peo_work)  #  员工空闲时间
                        # print('空闲时间：',idle) # 机器和员工共同空闲时间
                        # print("最早开始时间:{} \t 加工时间:{}\t最迟结束时间:{}".format(max(begin, operation.c_PJij), operation.processingTime, min(end, operation.s_SJij)))
                        new_solution[1][loc] = operation.machineset.index(alter_machine)
                        new_solution[2][loc] = operation.peopleset.index(alter_people)
                        # print("局部搜索前", job, ope, mac, peo)
                        # print("局部搜索后",job, ope, alter_machine-1,alter_people)
                        # 搜索到更好的解，则返回更好的解
                        return new_solution
    # 没有搜索到更好的解，则返回原始解
    return chromosome

def alternative_machine_worker(keyOpe,chromosome,parameters):
    '''更换设备和工人，完工时间小则保存，否则不变'''
    n = parameters["jobs"]
    ni = [0] * parameters["jobsnum"]
    for i in range(parameters["jobsnum"]):
        ni[i] = len(parameters['jobs'][i])
    # 随机选取一个关键工序，更换其加工设备和人员
    idx = random.randint(0,len(keyOpe)-1)
    operation = keyOpe[idx]
    # 获取工序的序号,替换的设备和人员标号
    job,ope = operation.jobidx, operation.opeidx
    mac = random.choice(operation.machineset)
    peo = random.choice(operation.peopleset)

    if len(operation.machineset) != 1:
        for i in range(len(n[job][ope])):
            item = n[job][ope][i]
            if item["machine"] == mac:
                chromosome[1][sum(ni[:job]) + ope] = i
                break
    chromosome[2][sum(ni[:job])+ope] = peo
    return chromosome

def movement_machine_worker(keyOpe,chromosome,parameters):
    '''更换设备和工人，完工时间小则保存，否则不变'''
    n = parameters["jobs"]
    ni = [0] * parameters["jobsnum"]
    for i in range(parameters["jobsnum"]):
        ni[i] = len(parameters['jobs'][i])

    mac_workload = [0]*parameters['VmNum']
    peo_workload = [0]*parameters['VmNum']
    # 记录设备和工人的使用情况
    for ope in keyOpe:
        mac_workload[ope.machine] += 1
        peo_workload[ope.people] += 1
    # 关键路径上用的最多的设备和人
    maxmac = mac_workload.index(max(mac_workload))
    maxpeo = peo_workload.index(max(peo_workload))


    # 选取在使用最多的设备上加工的关键工序，更换为在可用设备集上用的最少的机器
    idx = random.randint(0, len(keyOpe) - 1)
    while keyOpe[idx].machine != maxmac:
        idx = random.randint(0, len(keyOpe) - 1)

    # 获取工序的序号,替换的设备和人员标号
    job, ope = keyOpe[idx].jobidx, keyOpe[idx].opeidx
    # 可用设备(人员)超过一个才可替换，否则不可替换
    temp_mac_set = keyOpe[idx].machineset
    temp_peo_set = keyOpe[idx].peopleset

    # 替换为最小负荷的设备
    map_mac_load = []
    if len(temp_mac_set) != 1:
        for mac in temp_mac_set:
            mac -= 1 # 索引从0开始
            map_mac_load.append([mac_workload[mac],mac])
        map_mac_load = sorted(map_mac_load, key=lambda x:x[0])
        # 确定更换的设备编号alter_machine及在可用设备集中的编号i
        alter_machine = map_mac_load[0][1]
        # 判断更换后会不会减小负荷，减小则替换，不减小则忽略
        if map_mac_load[0][0]+1 < map_mac_load[-1][0]-1:
            for i in range(len(n[job][ope])):
                item = n[job][ope][i]
                if item["machine"] == alter_machine:
                    chromosome[1][sum(ni[:job]) + ope] = i
                    break

    # 替换为最小负荷的人员
    map_peo_load = []
    if len(temp_peo_set) != 1:
        for peo in temp_peo_set:
            map_peo_load.append([peo_workload[peo], peo])
        map_peo_load = sorted(map_peo_load, key=lambda x: x[0])
        # 判断更换后人员会不会减小负荷，减小则替换，不减小则忽略
        if map_peo_load[0][0] + 1 < map_peo_load[-1][0] - 1:
            # 确定更换的设备编号alter_machine及在可用设备集中的编号i
            alter_people = map_peo_load[0][1]
            chromosome[2][sum(ni[:job]) + ope] = alter_people

    return chromosome


class Key_operation():
    def __init__(self,jobidx:int,opeidx:int,machine:int,people:int,time:int,c_PJij:int,s_SJij:int,c_PMij:int,\
                 s_SMij:int,c_PWij:int,s_SWij:int,machineset:list,peopleset:list,timeset:list):
        '''
        :param jobidx: 工件
        :param opeidx: 工序
        :param machine: 执行加工的机器
        :param people: 执行加工的人员
        :param time: 基本加工时间
        :param c_PJij: 前序工件完工时间
        :param s_SJij: 后续工件开工时间
        :param c_PMij: 同台设备上的前续工件完工时间
        :param s_SMij: 同台设备上的后续工件开工时间
        :param c_PWij: 同一员工操作的前续工件完工时间
        :param s_SWij: 同一员工操作的后续工件开工时间
        :param machineset: 可加工设备集合
        :param peopleset: 可加工人员集合
        :param timeset: 加工时间集合
        '''

        self.jobidx = jobidx   # 索引从0开始
        self.opeidx = opeidx   # 索引从0开始
        self.machine = machine # 索引从1开始
        self.people = people   # 索引从0开始
        self.processingTime = time
        self.c_PJij = c_PJij  # 前序工件完工时间
        self.s_SJij = s_SJij  # 后续工件开工时间
        self.c_PMij = c_PMij  # 同台设备上的前续工件
        self.s_SMij = s_SMij  # 同台设备上的后续工件
        self.c_PWij = c_PWij  # 同一员工操作的前续工件
        self.s_SWij = s_SWij  # 同一员工操作的后续工件
        # 对于关键路径上的工序（最早开始时间=最迟开始时间）= max{工件前序，设备上前序，人员上前序}的完工时间
        self.late_satrt_time = max(self.c_PJij, self.c_PMij, self.c_PWij)
        # 对于关键路径上的工序（最迟完工时间）= min{工件后序，设备上后序，人员上后序}的开工时间
        self.late_complete_time = min(self.s_SJij,self.s_SMij,self.s_SWij)
        self.machineset = machineset  # 可加工设备集合[] (索引从1开始)
        self.peopleset = peopleset    # 可加工人员集合[] (索引从0开始)
        self.timeset = timeset             # 加工时间集合[]
        self.efficient = []           # 加工效率集合

    def info(self):
        '''
        关键工序: 9 - 5 - 3 - 0
        前序完工时间： 64 	     前设备完工时间: 90 	  前工人完工时间: 89
        后序开工时间： 10000 	 后设备开工时间: 10000   后工人开工时间: 10000
        最早开工时间:  90       最迟完工时间: 10000 	  加工时间: 1
        可用机器集合： [1, 4]
        可用人员集合： [0, 1, 2, 3, 4, 5]
        加工时间： [3, 2]
        '''
        print("关键工序:",self.jobidx,"-",self.opeidx,"-",self.machine,"-",self.people)
        print("前序完工时间：",self.c_PJij,"\t","前设备完工时间:",self.c_PMij,"\t","前工人完工时间:",self.c_PWij)
        print("后序开工时间：", self.s_SJij, "\t", "后设备开工时间:", self.s_SMij, "\t", "后工人开工时间:", self.s_SWij)
        print("最早开工时间:", self.late_satrt_time,"\t","最迟完工时间:",self.late_complete_time,"\t","加工时间:",self.processingTime)
        print("可用机器集合：", self.machineset)
        print("可用人员集合：", self.peopleset)
        print("加工时间：", self.timeset)





if __name__ == "__main__":
    path = r'C:\Users\Hangyu\PycharmProjects\FJSPP\柔性作业车间调度\FJSP\FJSPBenchmark\Brandimarte_Data\Mk01.fjs'
    model = FJSPLF(path)
    individual = [[8, 9, 3, 5, 5, 9, 1, 8, 0, 6, 5, 2, 2, 4, 5, 7, 3, 8, 4, 7, 0, 1, 4, 6, 4, 6, 1, 2, 0, 7, 1, 3, 0, 6, 9, 9, 2, 5, 1, 8, 0, 3, 9, 4, 6, 8, 7, 5, 8, 3, 7, 4, 0, 9, 2],
                  [1, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 2, 0, 2, 0, 0, 0, 0, 1, 2, 0, 1, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 2, 1, 1, 0, 1, 0, 1, 1],
                  [1, 4, 4, 3, 4, 3, 1, 5, 2, 3, 4, 3, 2, 4, 3, 5, 2, 0, 5, 1, 1, 2, 3, 4, 3, 3, 0, 3, 2, 2, 0, 1, 0, 2, 1, 0, 2, 0, 3, 3, 2, 1, 4, 0, 0, 5, 5, 3, 1, 5, 0, 1, 2, 3, 0]]

    makespan, maxload, totalload = model.get_obj(individual)
    print('完工时间为:', makespan, maxload, totalload)

    start = time.time()
    for i in range(1):
        # 找到该方案的关键路径
        #keyPath,machineTime,peopleTime = criticalpath(model.parameters, model.decode_lf(individual))
        keyPath,machineTime,peopleTime = criticalpath(model.parameters, model.decode(individual))
        # 新解
        new_chrom = criticalpath_localsearch(keyPath, individual, model.parameters,machineTime, peopleTime)
        # for j in range(100000):
        #     movement_machine_worker(keyPath, individual, model.parameters)
        # print(time.time() - start)
        makespan, maxload, totalload = model.get_obj(new_chrom)
        print('完工时间为:', makespan, maxload, totalload)
    print((time.time() - start))
    # print(new_chrom==individual)

    # ——————————————————————————————————————————————————————————————————————
    # 未执行局部搜索前的Gantt图
    gantt_data = model.translate_decoded_to_gantt(model.decode(individual))
    #print(gantt_data)
    #title = "局部搜索前"  # 甘特图title
    #gantt.draw_chart(gantt_data, title, 15)
    # 执行局部搜索后的Gantt图
    gantt_data = model.translate_decoded_to_gantt(model.decode(new_chrom))
    #print(gantt_data)
    #title = "局部搜索后"  # 甘特图title
    #gantt.draw_chart(gantt_data, title, 15)


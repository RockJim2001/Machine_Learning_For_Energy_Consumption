
# 剩余加工时间最长的优先加工
def init_mwr(parameters, os, ms, ws):
    o = parameters['jobs']
    Job_process = [0] * parameters['jobsnum']  # 储存第几工件加工第几工序

    ni = []  # 存储每个工件的工序数
    for job in parameters['jobs']:
        ni.append(len(job))

    vecotr = [[] for i in range(len(parameters['jobs']))]

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
        vecotr[job].append(((job,opr),prcTime, index_machine, people))
        # 更新工件加工到第几工序
        Job_process[job] += 1

    os_vector = []
    ms_vector = [-1 for _ in range(total_opr)]
    ws_vector = [-1 for _ in range(total_opr)]

    for i in range(total_opr):
        block = None
        maxval = max(item[0][1] for item in vecotr)
        for j in range(parameters['jobsnum']):
            if vecotr[j][0][1] == maxval:
                block = vecotr[j][0]
                print(block)
                del vecotr[j][0]
                vecotr[j].append((0,0,0,0))
                break

        job_num = block[0][0]  # 工件编码
        opr_num = block[0][1]  # 工序编码
        mac_num = block[2]     #
        people_num = block[3]
        os_vector.append(job_num)
        ms_vector[sum(ni[:job_num])+opr_num] = mac_num
        ws_vector[sum(ni[:job_num])+opr_num] = people_num

    return (os_vector,ms_vector,ws_vector)
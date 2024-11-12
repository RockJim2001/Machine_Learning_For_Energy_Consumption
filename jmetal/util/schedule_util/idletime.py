
def find_idle_time(list1,list2):
    '''寻找list1和list2共同的空闲时间'''
    # 先按第一个数排序，再按第二个排序
    List = sorted(list1+list2, key=lambda x:(x[0],x[1]))
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
    #idle.append((List[-1][1], 1000000))
    return idle





if __name__ == "__main__":
    # a,b 为工作区间， idle返回的是空闲区间
    a = [(1,3),(5,7),(8,10),[9,12],[7.5,8.5],[19,20],[22,25],[18.5,25]]
    b = [(1,2),(3,6),(7,7.5),[35,37]]
    idle = find_idle_time(a, b)
    print(idle)


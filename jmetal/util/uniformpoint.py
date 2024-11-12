import itertools
import math
import numpy as np

'''
自己写的 好用
UniformPoint(N,M): 在M维目标空间上取近似N个均匀分布的点
return: 均匀点组成的列表
'''

def uniformpoint(N :int,M :int) -> list:
    '''N 希望得到的点数， M 目标空间维度（目标函数个数）'''
    H1 = 1
    while nchoosek(H1+M, M-1) <= N:
        H1 += 1
    W = np.array([*itertools.combinations([i for i in range(1,H1+M)],M-1)],dtype=np.float64) - np.tile([i for i in range(0,M-1)],\
        (nchoosek(H1+M-1, M-1),1)) - 1

    W = ( np.hstack((W, np.zeros((W.shape[0],1)) + H1)) - np.hstack((np.zeros((W.shape[0],1)), W)) ) / H1

    if H1 < M:
        H2 = 0
        while nchoosek(H1+M-1, M-1) + nchoosek(H2+M, M-1) <= N:
            H2 += 1
        if H2 > 0:
            W2 = np.array([*itertools.combinations([i for i in range(1,H2+M)],M-1)],dtype=np.float64) \
                 - np.tile([i for i in range(0,M-1)],(nchoosek(H2+M-1, M-1),1)) - 1
            W2 = (np.hstack((W2, np.zeros((W2.shape[0], 1)) + H2)) - np.hstack((np.zeros((W2.shape[0], 1)), W2))) / H2
            W = np.hstack(W,W2/2+1/(2*M))

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i][j] < 1e-6:
                W[i][j] = 1e-6
    # 写入文件

    np.savetxt(X=W, fname=r'C:/Users/Hangyu/PycharmProjects/JmetalPlus/resources/uniform_point/uniformPoint_{0}_{1}'.format(N, M))

    return W.tolist()


def nchoosek(n, m):
    res = math.factorial(n) // (math.factorial(n-m) * math.factorial(m))
    return res


if __name__ == "__main__":
    W = uniformpoint(10,3)
    print(W)
    print(len(W))
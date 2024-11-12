import random
import numpy as np

def efficency(m):
    '''初始技能水平'''
    matric = [[round(random.uniform(0.85,1.15),2) for _ in range(m)] for _ in range(m)]
    matric = np.array(matric)
    for i in range(m):
        for j in range(i + 1):
            matric[j][i] = matric[i][j]

    return matric.tolist()

def learning_rate(m):
    '''学习遗忘效应'''

    alpha = [[round(random.uniform(-0.15,-0.3), 2) for _ in range(m)] for _ in range(m)]
    alpha = np.array(alpha)
    for i in range(m):
        for j in range(i + 1):
            alpha[j][i] = alpha[i][j]

    beta = [[round(random.uniform(0.15, 0.25), 2) for _ in range(m)]for _ in range(m)]
    beta = np.array(beta)
    for i in range(m):
        for j in range(i + 1):
            beta[j][i] = beta[i][j]

    return alpha.tolist(), beta.tolist()

def similarity(n):
    '''工件相似度矩阵'''
    matric = [[round(random.uniform(0.4,1),1) for _ in range(n)] for _ in range(n)]
    matric = np.array(matric)
    for i in range(n):
        for j in range(i + 1):
            matric[j][i] = matric[i][j]
            if i == j:
                matric[j][i]=1
    return matric.tolist()

def write_to_file(path,text):
    with open(path,"w") as f:
        f.write(text)
        f.write("\n")




if __name__ == "__main__":
    n = 30
    m = 10
    E = efficency(m)
    alpha, beta = learning_rate(m)
    delta = similarity(n)

    print('Efficency: \n', E)
    print('learning_rate: \n', alpha)
    print('forgetting_rate: \n', beta)
    print('delta: \n', delta)
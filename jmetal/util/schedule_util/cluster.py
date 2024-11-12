from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import random

X = np.array([[random.random(), random.random(), random.random()] for i in range(100)])

def cluster(X,k):
    kmeans_model = KMeans(n_clusters=k, random_state=1).fit(X)
    labels = kmeans_model.labels_

    # figure = plt.figure()
    # ax = figure.add_subplot(111, projection = '3d')
    #
    # colorlist = ["red","orange","green"]
    #
    # for k in set(labels.tolist()):
    #     c1 = random.randint(0,255)
    #     c2 = random.randint(0,255)
    #     c3 = random.randint(0,255)
    #     ax.scatter(X[labels==k,0], X[labels==k,1], X[labels==k,2], s = 40, color = (c1/255, c2/255, c3/255), marker = '.')
    #
    # #设置坐标轴
    # ax.invert_xaxis()
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
     #显示图像
    # plt.show()
    return labels

if __name__ == "__main__":
    start = time.time()
    for i in range(10000):
        labels = cluster(X, 5)
    print(time.time() - start)
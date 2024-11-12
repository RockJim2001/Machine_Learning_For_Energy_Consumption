#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：boston_housing.py
@Author ：RockJim
@Date ：2023/12/8 11:16
@Description ：波士顿房价预估
@Version ：1.0
"""
import pandas as pd

'''导入数据'''
data = pd.read_excel('波士顿房价预测.xlsx', header=None,
                     index_col=None)  # 一共506组数据，每组数据13个特征，13个特征对应一个输出
y = data.loc[:, 13:13]  # 将标签数据存储在y中，表格最后一列为标签
x = data.loc[:, 0:12]  # 将特征数据存储在x中，表格前13列为特征,

from sklearn.preprocessing import StandardScaler

'''对每列数据执行标准化'''
scaler = StandardScaler()  # 实例化
X = scaler.fit_transform(x)  # 标准化特征
Y = scaler.fit_transform(y)  # 标准化标签

# x = scaler.inverse_transform(X) # 这行代码可以将数据恢复至标准化之前

import torch

'''划分数据集'''
X = torch.tensor(X, dtype=torch.float32)  # 将数据集转换成torch能识别的格式
Y = torch.tensor(Y, dtype=torch.float32)
torch_dataset = torch.utils.data.TensorDataset(X, Y)  # 组成torch专门的数据库
batch_size = 6  # 设置批次大小

# 划分训练集测试集与验证集
torch.manual_seed(seed=2021)  # 设置随机种子分关键，不然每次划分的数据集都不一样，不利于结果复现
train_validaion, test = torch.utils.data.random_split(
    torch_dataset,
    [450, 56],
)  # 先将数据集拆分为训练集+验证集（共450组），测试集（56组）
train, validation = torch.utils.data.random_split(
    train_validaion, [400, 50])  # 再将训练集+验证集拆分为训练集400，测试集50


class Net(torch.nn.Module):
    '''搭建神经网络'''

    def __init__(
            self, n_feature, n_output, n_neuron1, n_neuron2,
            n_layer):  # n_feature为特征数目，这个数字不能随便取,n_output为特征对应的输出数目，也不能随便取
        self.n_feature = n_feature
        self.n_output = n_output
        self.n_neuron1 = n_neuron1  # 待优化超参数
        self.n_neuron2 = n_neuron2  # 待优化超参数
        self.n_layer = n_layer  # 待优化超参数
        super(Net, self).__init__()
        self.input_layer = torch.nn.Linear(self.n_feature,
                                           self.n_neuron1)  # 输入层
        self.hidden1 = torch.nn.Linear(self.n_neuron1, self.n_neuron2)  # 1类隐藏层
        self.hidden2 = torch.nn.Linear(self.n_neuron2, self.n_neuron2)  # 2类隐藏
        self.predict = torch.nn.Linear(self.n_neuron2, self.n_output)  # 输出层

    def forward(self, x):
        '''定义前向传递过程'''
        out = self.input_layer(x)
        out = torch.relu(out)  # 使用relu函数非线性激活
        out = self.hidden1(out)
        out = torch.relu(out)
        for _ in range(self.n_layer):
            out = self.hidden2(out)
            out = torch.relu(out)
        out = self.predict(  # 回归问题最后一层不需要激活函数
            out
        )  # 除去n_feature与out_prediction不能随便取，隐藏层数与其他神经元数目均可以适当调整以得到最佳预测效果
        return out


def structure_initialization(parameters):
    '''实例化神经网络'''
    n_layer = parameters.get('n_layer', 2)  # 若n_layer缺省则取默认值2
    n_neuron1 = parameters.get('n_neuron1', 140)
    n_neuron2 = parameters.get('n_neuron2', 140)
    learning_rate = parameters.get('learning_rate', 0.0001)
    net = Net(n_feature=13,
              n_output=1,
              n_layer=n_layer,
              n_neuron1=n_neuron1,
              n_neuron2=n_neuron2)  # 这里直接确定了隐藏层数目以及神经元数目，实际操作中需要遍历
    optimizer = torch.optim.Adam(net.parameters(),
                                 learning_rate)  # 使用Adam算法更新参数
    criteon = torch.nn.MSELoss()  # 误差计算公式，回归问题采用均方误差
    return net, optimizer, criteon


def train_evaluate(parameterization):
    '''此函数返回模型误差作为贝叶斯优化依据'''
    net, optimizer, criteon = structure_initialization(parameterization)
    batch_size = parameterization.get('batch_sizes', 6)
    epochs = parameterization.get('epochs', 100)
    # 将训练集划分批次，每batch_size个数据一批
    train_data = torch.utils.data.DataLoader(train,
                                             batch_size=batch_size,
                                             shuffle=True)
    net.train()  # 启动训练模式
    for epoch in range(epochs):  # 整个数据集迭代次数
        for batch_idx, (data, target) in enumerate(train_data):
            logits = net.forward(data)  # 前向计算结果（预测结果）
            loss = criteon(logits, target)  # 计算损失
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 后向传递过程
            optimizer.step()  # 优化权重与偏差矩阵

    logit = []  # 这个是验证集，可以根据验证集的结果进行调参，这里根据验证集的结果选取最优的神经网络层数与神经元数目
    target = []
    net.eval()  # 启动测试模式
    for data, targets in validation:  # 输出验证集的平均误差
        logits = net.forward(data).detach().numpy()
        targets = targets.detach().numpy()
        target.append(targets[0])
        logit.append(logits[0])
    average_loss = criteon(torch.tensor(logit), torch.tensor(target))  # 计算损失
    return float(average_loss)


from ax.service.managed_loop import optimize
# 使用贝叶斯优化超参数,可以使用pip install ax-platform命令安装，贝叶斯优化具体介绍见https://ax.dev/docs/bayesopt.html


def bayesian_optimization():
    best_parameters, values, experiment, model = optimize(
        parameters=[{
            "name": "learning_rate",
            "type": "range",
            "bounds": [1e-6, 0.1],
            "log_scale": True
        }, {
            "name": "n_layer",
            "type": "range",
            "bounds": [0, 4]
        }, {
            "name": "n_neuron1",
            "type": "range",
            "bounds": [40, 300]
        }, {
            "name": "n_neuron2",
            "type": "range",
            "bounds": [40, 300]
        }, {
            "name": "batch_sizes",
            "type": "range",
            "bounds": [6, 100]
        }, {
            "name": "epochs",
            "type": "range",
            "bounds": [300, 500]
        }],
        evaluation_function=train_evaluate,
        objective_name='MSE LOSS',
        total_trials=200,  # 执行200次优化
        minimize=True)  # 往最小值方向优化（默认往最大值方向优化）
    return best_parameters


best = bayesian_optimization()  # 返回最优的结构
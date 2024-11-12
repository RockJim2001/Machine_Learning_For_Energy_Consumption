#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：config.py
@Author ：RockJim
@Date ：2023/7/24 23:48
@Description ：数据集配置文件
@Version ：1.0
"""
import os.path
import numpy as np

# ########数据配置#################
# 数据存储根目录
ROOT_PATH = r'E:\PythonProject\machine_learning\ML_project-Data_3\ML_project'
# base-数据.csv
BASE_DATASET_NAME = 'base-数据.csv'
# 波士顿房价
BOSTON_HOUSING_DATA = 'boston_housing.data'

# 800组的数据
BASE_DATASET_NAME_800 = 'AllCombinedResults-0800组.csv'
# 900组的数据
BASE_DATASET_NAME_900 = 'AllCombinedResults-0900组.csv'

# 3000组的数据
BASE_DATASET_NAME_3000 = 'AllCombinedResults-3000组.csv'
# 5000组的数据
BASE_DATASET_NAME_5000 = 'AllCombinedResults-5000组.csv'
# 模型1的数据
BASE_DATASET_NAME_MODEL_1 = 'table_3_10000_cols.csv'
# 模型2的数据
BASE_DATASET_NAME_MODEL_2 = '表2的10000组-发东东版本.csv'


# 模型3的300数据
BASE_DATASET_NAME_MODEL_3 = 'table_3_50000_300_cols.csv'
# 模型3的1000数据
BASE_DATASET_NAME_MODEL_4 = 'table_3_50000_1000_cols.csv'
# 模型3的4000数据
BASE_DATASET_NAME_MODEL_5 = 'table_3_50000_4000_cols.csv'
# 模型3的6000数据
BASE_DATASET_NAME_MODEL_6 = 'table_3_50000_6000_cols.csv'
# 模型3的9000数据
BASE_DATASET_NAME_MODEL_7 = 'table_3_50000_9000_cols.csv'
# 模型3的12000数据
BASE_DATASET_NAME_MODEL_8 = 'table_3_50000_12000_cols.csv'
# 模型3的21000数据
BASE_DATASET_NAME_MODEL_9 = 'table_3_50000_21000_cols.csv'
# 模型3的30000数据
BASE_DATASET_NAME_MODEL_10 = 'table_3_50000_30000_cols.csv'
# 模型3的40000数据
BASE_DATASET_NAME_MODEL_11 = 'table_3_50000_40000_cols.csv'
# 模型3的50000数据
BASE_DATASET_NAME_MODEL_12 = 'table_3_50000_50000_cols.csv'
# 模型3的50000数据，只研究目标1
BASE_DATASET_NAME_MODEL_13 = 'table_3_50000_cols_zong.csv'

# ###########日志配置#############
# 指定日志存储路径
DATASET_NAME = BASE_DATASET_NAME_MODEL_1  # 更换当前的数据集
folder = DATASET_NAME.split('.')[0]
log_dir = os.path.join(ROOT_PATH, 'logs', folder)
log_file_path = os.path.join(ROOT_PATH, 'logs', folder, 'test.log')

# ###########随机数种子设置#############

# 设置随机数种子
seed_value = 2023
np.random.seed(seed_value)

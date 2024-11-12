#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：split.py
@Author ：RockJim
@Date ：2024/2/25 17:50
@Description ：None
@Version ：1.0
"""
import os.path

import pandas as pd

# 读取 CSV 文件
df = pd.read_csv(r'C:\Users\25760\Desktop\表3_50000.csv')

list_data = [300, 1000, 4000, 6000, 9000, 12000, 21000, 30000, 40000, 50000]  # 数据的大小

for size in list_data:
    sampled_data = df.sample(n=size, random_state=42)
    output_file_path = f'table_3_50000_{size}_cols.csv'
    sampled_data.to_csv(os.path.join(r'C:\Users\25760\Desktop', output_file_path), index=False)

# 将处理后的数据保存到新的 CSV 文件中
# df.to_csv(r'C:\Users\25760\Desktop\table_4_10000_cols.csv', index=True)

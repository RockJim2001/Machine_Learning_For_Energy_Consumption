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
df = pd.read_csv(r'E:\PythonProject\machine_learning\ML_project-Data_4\ML_project\resource\008_18_floor_table_4\18层-AllCombinedResults-22000+28000组20240423.csv')

list_data = [300, 1000, 4000, 6000, 9000, 12000, 21000, 30000, 40000, 50000]  # 数据的大小

for size in list_data:
    sampled_data = df.sample(n=size, random_state=42)
    output_file_path = f'008_18_floor_table_4_50000_cols_{size}_cols.csv'
    sampled_data.to_csv(os.path.join(r'E:\PythonProject\machine_learning\ML_project-Data_4\ML_project\resource\008_18_floor_table_4', output_file_path), index=False)

# 将处理后的数据保存到新的 CSV 文件中
# df.to_csv(r'C:\Users\25760\Desktop\table_4_10000_cols.csv', index=True)

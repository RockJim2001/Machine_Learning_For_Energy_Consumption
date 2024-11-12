import pandas as pd
import numpy as np

path = r"E:\PythonProject\machine_learning\ML_project-Data_4\ML_project\resource\table_4_10000_cols_new.csv"
data = pd.read_csv(path).iloc[:, 2:32]
data = np.array(data)

print('变量上界：', np.max(data, axis=0).tolist())
print('变量下界：', np.min(data, axis=0).tolist())

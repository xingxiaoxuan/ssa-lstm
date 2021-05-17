#!/usr/bin/python3
# -*- coding: utf-8 -*-
# 缺失值：3634	3635	3636	3637	3638	3639	3656	4609
import xlrd
import xlwt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

src = 'F:/深度学习资料/XGBoost/SSA-LSTM/'


def get_data():
    workbook = xlrd.open_workbook(r'F:/深度学习资料/聚类/数据/CHP.xls')
    sheet1 = workbook.sheet_by_name('Data_CHP')
    data = np.mat(np.zeros((8784, 3)))
    for n in np.arange(1, 8785):
        data[n-1, :] = sheet1.row_values(n)[1:4]
    data = np.array(data)
    return data


data = get_data()
data = data.astype('float64')

# 标准化
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(data)

# 取冷热电数据
cool = data[:, 0]
steam = data[:, 1]
elec = data[:, 2]

# 缺失值 3634 3635 3636 3637 3638 3639 3656 4609
for i in range(8784):
    for j in range(3):
        if data[i, j] == 0:
            print("i, j: ", i, j)


data[3656, :] = data[3655, :] / 4 + data[3657, :] - data[3658, :] / 4
# print(data[3656, :])  # [1.35823633e+04 1.08959649e+05 3.98070054e+01]
data[4609, :] = data[4608, :] / 4 + data[4610, :] - data[4611, :] / 4
# print(data[4609, :])  # [5.87853885e+03 1.17679398e+05 2.78577566e+01]

f = xlwt.Workbook()
sheet1 = f.add_sheet(r'离散缺失值')
for j in range(3):
    for i in range(8784):
        sheet1.write(i, j, float(data[i, j]))
f.save(r'F:/深度学习资料/XGBoost/SSA-LSTM/Outlier_heading.xls')

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import xlrd
import xlwt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
# from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(tf.test.is_gpu_available())

src = 'F:/深度学习资料/XGBoost/SSA-LSTM/'


def get_data():
    workbook = xlrd.open_workbook(r'F:/深度学习资料/XGBoost/SSA-LSTM/数据/部分月份CHP.xls')
    sheet1 = workbook.sheet_by_name('1、4、7、10')
    data = np.mat(np.zeros((2952, 3)))
    for n in np.arange(0, 2952):
        data[n, :] = sheet1.row_values(n)[0:3]
    data = np.array(data)
    return data


data = get_data()
data = data.astype('float64')
print("data.shape: ", data.shape)
# 标准化
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(data)
# 取冷热电数据
cool = data[:, 0]
steam = data[:, 1]
elec = data[:, 2]

# path = "xxxx"  # 数据集路径
# series = np.loadtxt(path)
# series = series - np.mean(series)  # 中心化(非必须)
series = cool

# step1 嵌入
windowLen = 168  # 嵌入窗口长度
seriesLen = len(series)  # 序列长度 8784

K = seriesLen - windowLen + 1
X = np.zeros((windowLen, K))
for i in range(K):
    X[:, i] = series[i:i + windowLen]

# step2: svd分解， U和sigma已经按升序排序
U, sigma, VT = np.linalg.svd(X, full_matrices=False)
# print(sigma)

sigma_2 = np.power(sigma, 2)
# print(sigma_2)

sum = np.sum(sigma_2)
# print(sum)

contribution = sigma_2 / sum  # 计算贡献率
# print(contribution)

# 保存贡献率
f = xlwt.Workbook()
sheet1 = f.add_sheet(r'贡献率')
for i in range(len(contribution)):
    sheet1.write(i, 0, float(contribution[i]))
f.save(r'./结果/cool/contribution_cool.xls')

for i in range(VT.shape[0]):
    VT[i, :] *= sigma[i]
A = VT

# 重组
rec = np.zeros((windowLen, seriesLen))
for i in range(windowLen):
    for j in range(windowLen - 1):
        for m in range(j + 1):
            rec[i, j] += A[i, j - m] * U[m, i]
        rec[i, j] /= (j + 1)
    for j in range(windowLen - 1, seriesLen - windowLen + 1):
        for m in range(windowLen):
            rec[i, j] += A[i, j - m] * U[m, i]
        rec[i, j] /= windowLen
    for j in range(seriesLen - windowLen + 1, seriesLen):
        for m in range(j - seriesLen + windowLen, windowLen):
            rec[i, j] += A[i, j - m] * U[m, i]
        rec[i, j] /= (seriesLen - j)

print("rec.shape: ", rec.shape)
f = xlwt.Workbook()
sheet1 = f.add_sheet(r'cool')
for i in range(20):
    for j in range(seriesLen):
        sheet1.write(j, i, float(rec[i, j]))
f.save(r'./结果/cool/前20特征分量.xls')

rrr = np.sum(rec[0: 20, :], axis=0)  # 选择重构的部分，这里选了大于0.01%的序列

f = xlwt.Workbook()
sheet1 = f.add_sheet(r'cool')
for i in range(seriesLen):
    sheet1.write(i, 0, float(rrr[i]))
f.save(r'./结果/cool/重组时间序列.xls')

plt.figure()
# for i in range(12):
#     ax = plt.subplot(3, 4, i + 1)
#     ax.plot(rec[i, :])

for i in range(20):
    ax = plt.subplot(4, 5, i + 1)
    ax.plot(rec[i, :])

plt.figure(2)
plt.plot(series, 'r', label='Original')
plt.plot(rrr, 'b', label='Reconstructed')
plt.title('cool')
plt.legend(['Original', 'Reconstructed'])  # 对应曲线的标签

plt.show()

RMSE = sqrt(mean_squared_error(series, rrr))  # 987.4258576473045
MAPE = mean_absolute_error(series, rrr)  # 720.2506467125254
print("RMSE: ", RMSE)
print("MAPE: ", MAPE)

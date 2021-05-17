import numpy as np
import xlrd
import xlwt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr


def get_data():
    workbook = xlrd.open_workbook(r'./数据/index.xls')
    sheet1 = workbook.sheet_by_name('1、4、7、10')
    data = np.mat(np.zeros((2952, 10)))
    for n in np.arange(1, 2953):
        data[n-1, :] = sheet1.row_values(n)[1:11]
    data = np.array(data)
    return data


index = get_data()
index = index.astype('float64')
print("data.shape: ", index.shape)  # (2952, 10)
# 月  日  小时  冷负荷  热负荷  电负荷  WBT(F)  DBT(F)  RH(%)CS3  RH(%)IAC-GT10

monthx = np.sin((2 * 3.14 / 12) * index[:, 0])
monthy = np.cos((2 * 3.14 / 12) * index[:, 0])
Dayx = np.sin((2 * 3.14 / 31) * index[:, 1])
Dayy = np.cos((2 * 3.14 / 31) * index[:, 1])
Hourx = np.sin((2 * 3.14 / 24) * index[:, 2])
Houry = np.cos((2 * 3.14 / 24) * index[:, 2])

# 标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(index[:, 3:10])
print(scaled.shape)

input = np.zeros((2952, 13))
input[:, 0] = monthx
input[:, 1] = monthy
input[:, 2] = Dayx
input[:, 3] = Dayy
input[:, 4] = Hourx
input[:, 5] = Houry
for i in range(6, 13):
    input[:, i] = scaled[:, i-6]

# f = xlwt.Workbook()
# sheet1 = f.add_sheet(r'index')
# for i in range(13):
#     for j in range(2952):
#         sheet1.write(j, i, float(input[j, i]))
# f.save(r'./数据/input_index.xls')


def get_cool():
    workbook = xlrd.open_workbook(r'./结果/cool/前20特征分量.xls')
    sheet1 = workbook.sheet_by_name('cool')
    data = np.mat(np.zeros((2952, 20)))
    for n in np.arange(0, 2952):
        data[n, :] = sheet1.row_values(n)[0:20]
    data = np.array(data)
    return data


cool_eign = get_cool()
cool_eign = cool_eign.astype('float64')
# print(cool_eign.shape)  # (2952, 20)


def get_elec():
    workbook = xlrd.open_workbook(r'./结果/elec/前21分量.xls')
    sheet1 = workbook.sheet_by_name('elec')
    data = np.mat(np.zeros((2952, 22)))
    for n in np.arange(0, 2952):
        data[n, :] = sheet1.row_values(n)[0:22]
    data = np.array(data)
    return data


elec_eign = get_elec()
elec_eign = elec_eign.astype('float64')
# print(elec_eign.shape)  # (2952, 22)


def get_steam():
    workbook = xlrd.open_workbook(r'./结果/steam/前24特征分量.xls')
    sheet1 = workbook.sheet_by_name('steam')
    data = np.mat(np.zeros((2952, 24)))
    for n in np.arange(0, 2952):
        data[n, :] = sheet1.row_values(n)[0:24]
    data = np.array(data)
    return data


steam_eign = get_steam()
steam_eign = steam_eign.astype('float64')
# print(steam_eign.shape)  # (2952, 24)

f = xlwt.Workbook()
sheet1 = f.add_sheet('cool')

result = np.mat(np.zeros((13, 20)))
# 进行关联度分析
for i in range(13):
    for j in range(20):
        a = pearsonr(input[:, i], cool_eign[:, j])[0]
        sheet1.write(i, j, a)
        result[i, j] = a
print(result)
f.save(r'./结果/关联度分析/cool.xls')

f = xlwt.Workbook()
sheet1 = f.add_sheet('elec')

result = np.mat(np.zeros((13, 22)))
# 进行关联度分析
for i in range(13):
    for j in range(22):
        a = pearsonr(input[:, i], elec_eign[:, j])[0]
        sheet1.write(i, j, a)
        result[i, j] = a
print(result)
f.save(r'./结果/关联度分析/elec.xls')

f = xlwt.Workbook()
sheet1 = f.add_sheet('steam')

result = np.mat(np.zeros((13, 24)))
# 进行关联度分析
for i in range(13):
    for j in range(24):
        a = pearsonr(input[:, i], steam_eign[:, j])[0]
        sheet1.write(i, j, a)
        result[i, j] = a
print(result)
f.save(r'./结果/关联度分析/steam.xls')

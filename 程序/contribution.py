import xlrd
import numpy as np
import matplotlib.pyplot as plt


def get_data():
    workbook = xlrd.open_workbook(r'F:/深度学习资料/XGBoost/SSA-LSTM/结果/贡献率汇总.xls')
    sheet1 = workbook.sheet_by_name('Sheet1')
    data = np.mat(np.zeros((168, 3)))
    for n in np.arange(0, 168):
        data[n, :] = sheet1.row_values(n)[0:3]
    data = np.array(data)
    return data


data = get_data()
data = data.astype('float64')
print(data.shape)
log_data = np.log10(data)

for i in range(3):
    plt.figure()
    plt.plot(range(168), log_data[:, i])
    plt.plot(20, )
plt.show()

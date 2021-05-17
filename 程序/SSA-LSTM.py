import numpy as np
import xlrd
import xlwt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Activation
from keras import optimizers
from matplotlib import pyplot
from math import sqrt
from keras import Input, Model
from keras.utils import plot_model


def get_origin_data():
    workbook = xlrd.open_workbook(r'F:/深度学习资料/XGBoost/SSA-LSTM/数据/部分月份CHP.xls')
    sheet1 = workbook.sheet_by_name('1、4、7、10')
    data = np.mat(np.zeros((2952, 3)))
    for n in np.arange(0, 2952):
        data[n, :] = sheet1.row_values(n)[0:3]
    data = np.array(data)
    return data


def get_data():
    workbook = xlrd.open_workbook(r'F:/深度学习资料/XGBoost/SSA-LSTM/数据/降噪后数据.xls')
    sheet1 = workbook.sheet_by_name('Sheet1')
    data = np.mat(np.zeros((2952, 3)))
    for n in np.arange(0, 2952):
        data[n, :] = sheet1.row_values(n)[0:3]
    data = np.array(data)
    return data


data = get_data()
origin_data = get_origin_data()
data = data.astype('float64')
origin_data = origin_data.astype('float64')

# 标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data)
origin_scaled = scaler.fit_transform(origin_data)

# 取冷热电数据
cool = scaled[:, 0]
origin_cool = origin_scaled[:, 0]

steam = scaled[:, 1]
origin_steam = origin_scaled[:, 1]

elec = scaled[:, 2]
origin_elec = origin_scaled[:, 2]

# 生成训练集
train_cool = np.zeros((len(cool) - 23, 24))
for i in range(len(cool) - 23):
    for x in range(24):
        train_cool[i][x] = steam[i + x]  # 0 cool 1 steam 2 elec

print(train_cool.shape)  # (2929, 24)

test_cool = origin_steam[23:]  # 0 cool 1 steam 2 elec
train_cool = train_cool.reshape((train_cool.shape[0], 1, train_cool.shape[1]))

x_train_cool = train_cool[: -23, :, :]
x_test_cool = train_cool[-23:, :, :]

y_train_cool = test_cool[:-23]
y_test_cool = test_cool[-23:]

print(x_train_cool.shape, x_test_cool.shape, y_train_cool.shape, y_test_cool.shape)
# (2906, 1, 24) (23, 1, 24) (2906,) (23,)


def build_model():
    # model = Sequential()
    # model.add(LSTM(128, input_shape=(x_train_cool.shape[1], x_train_cool.shape[2]), return_sequences=True))
    # model.add(Dropout(0.3))
    # model.add(LSTM(64, return_sequences=False))
    # model.add(Dropout(0.3))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    input_tensor = Input(shape=(x_train_cool.shape[1], x_train_cool.shape[2]))
    x = LSTM(48, return_sequences=True)(input_tensor)
    x = LSTM(24, return_sequences=True)(x)
    x = LSTM(12, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    output_tensor = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, output_tensor)
    model.summary()
    # plot_model(model, to_file='model.png')
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mae', optimizer=adam, metrics=['accuracy'])
    # fit network
    history = model.fit(x_train_cool, y_train_cool, epochs=300, batch_size=20,
                        validation_data=(x_test_cool, y_test_cool), shuffle=False)
    model.save('./elec_SSA-LSTM3')
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    return model


model = build_model()
# model = load_model('F:/深度学习资料/XGBoost/SSA-LSTM/model/steam/SSA-LSTM')

# make a prediction
yhat = model.predict(x_test_cool)

# invert scaling for forecast
x_test_cool = x_test_cool.reshape((x_test_cool.shape[0], x_test_cool.shape[2]))
# inv_yhat = np.concatenate((yhat, x_test_cool[:-23, 1:]), axis=1)

for i in range(23):
    origin_data[-23+i, 1] = yhat[i]  # 0 cool 1 steam 2 elec

inv_yhat = scaler.inverse_transform(origin_data)
inv_yhat = inv_yhat[-23:, 1]  # 0 cool 1 elec 2 steam
print("inv_yhat: ", inv_yhat)

# invert scaling for actual
for i in range(23):
    origin_data[-23+i, 1] = y_test_cool[i]  # 0 cool 1 steam 2 elec

inv_y = scaler.inverse_transform(origin_data)
inv_y = inv_y[-23:, 1]  # 0 cool 1 steam 2 elec
print("inv_y: ", inv_y)

# 做ROC曲线
pyplot.figure()
pyplot.plot(range(len(inv_yhat)), inv_yhat, 'b', label="predict")
pyplot.plot(range(len(inv_y)), inv_y, 'r', label="test")
pyplot.legend(['predict', 'test'])  # 显示图中的标签
pyplot.show()

# calculate RMSE
RMSE = sqrt(mean_squared_error(inv_y, inv_yhat))
MAPE = mean_absolute_error(inv_y, inv_yhat)
print('Test RMSE: %.3f ，Test MAPE: %.3f' % (RMSE, MAPE))

acc = (1 - np.mean(np.abs(inv_y - inv_yhat) / inv_y)) * 100
print("精准度：", acc, "%")

f = xlwt.Workbook()
sheet1 = f.add_sheet('steam')

result = np.mat(np.zeros((13, 24)))
# 进行关联度分析
for i in range(23):
    sheet1.write(i, 0, inv_y[i])
    sheet1.write(i, 1, inv_yhat[i])

f.save(r'F:/深度学习资料/XGBoost/SSA-LSTM/结果/elec/SSA-LSTM预测3.xls')

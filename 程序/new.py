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


def input_data():
    workbook = xlrd.open_workbook(r'F:/深度学习资料/XGBoost/SSA-LSTM/结果/cool/cool输入特征.xls')
    sheet1 = workbook.sheet_by_name('Sheet1')
    data = np.mat(np.zeros((2952, 11)))
    for n in np.arange(1, 2953):
        data[n-1, :] = sheet1.row_values(n)[0:11]
    data = np.array(data)
    return data


input_data = input_data()
input_data = input_data.astype('float64')
print(input_data.shape)

time_stamp = 24
train = input_data[0: 2904 + time_stamp]
valid = input_data[2904 - time_stamp:]
print(train.shape, valid.shape)

# 标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(train)

x_train, y_train = [], []
for i in range(time_stamp, len(train)):
    x_train.append(scaled[i - time_stamp:i])
    y_train.append(scaled[i, 10])
x_train, y_train = np.array(x_train), np.array(y_train)

scaled = scaler.transform(valid)
x_valid, y_valid = [], []
for i in range(time_stamp, len(valid)):
    x_valid.append(scaled[i - time_stamp: i])
    y_valid.append(scaled[i, 10])
x_valid, y_valid = np.array(x_valid), np.array(y_valid)

print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)
# (2904, 24, 11) (2904,) (48, 24, 11) (48,)


def build_model():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_dim=x_train.shape[-1], input_length=x_train.shape[1]))
    # model.add(LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # input_tensor = Input(shape=(x_train.shape[1], x_train.shape[2]))
    # x = LSTM(48, return_sequences=True)(input_tensor)
    # x = LSTM(24, return_sequences=True)(x)
    # x = LSTM(12, return_sequences=False)(x)
    # x = Dropout(0.2)(x)
    # output_tensor = Dense(1, activation='sigmoid')(x)
    # model = Model(input_tensor, output_tensor)
    model.summary()
    # plot_model(model, to_file='model.png')
    adam = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mae', optimizer=adam, metrics=['accuracy'])
    # fit network
    model.fit(x_train, y_train, epochs=50, batch_size=20, verbose=1)
    model.save('./cool_new1')
    # plot history
    return model


model = build_model()
# model = load_model('cool_new1')

# make a prediction
yhat = model.predict(x_valid)
scaler.fit_transform(input_data[:, 10].reshape(-1, 1))
yhat = scaler.inverse_transform(yhat)
y_valid = scaler.inverse_transform(y_valid.reshape(-1, 1))

# 做ROC曲线
pyplot.figure()
pyplot.plot(range(len(yhat)), yhat, 'b', label="predict")
pyplot.plot(range(len(y_valid)), y_valid, 'r', label="test")
pyplot.legend(['predict', 'test'])  # 显示图中的标签
pyplot.show()

# calculate RMSE
RMSE = sqrt(mean_squared_error(y_valid, yhat))
MAPE = mean_absolute_error(y_valid, yhat)
print('Test RMSE: %.3f ，Test MAPE: %.3f' % (RMSE, MAPE))

acc = (1 - np.mean(np.abs(y_valid - yhat) / y_valid)) * 100
print("精准度：", acc, "%")

# f = xlwt.Workbook()
# sheet1 = f.add_sheet('cool')
#
# for i in range(23):
#     sheet1.write(i, 0, y_valid[i])
#     sheet1.write(i, 1, yhat[i])
#
# f.save(r'F:/深度学习资料/XGBoost/SSA-LSTM/结果/cool/new0.xls')

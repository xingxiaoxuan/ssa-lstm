import numpy as np
import xlrd
import xlwt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras import optimizers
from matplotlib import pyplot
from math import sqrt
from keras import Input, Model


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
# 标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data)
# 取冷热电数据
cool = scaled[:, 0]
steam = scaled[:, 1]
elec = scaled[:, 2]

train_cool = np.zeros((len(cool) - 23, 24))
for i in range(len(cool) - 23):
    for x in range(24):
        train_cool[i][x] = steam[i + x]
print(train_cool.shape)

test_cool = steam[23:]
train_cool = train_cool.reshape((train_cool.shape[0], 1, train_cool.shape[1]))

x_train_cool = train_cool[: -23, :, :]
x_test_cool = train_cool[-23:, :, :]

y_train_cool = test_cool[:-23]
y_test_cool = test_cool[-23:]

print(x_train_cool.shape, x_test_cool.shape, y_train_cool.shape, y_test_cool.shape)


def build_model():
    # model = Sequential()
    # model.add(LSTM(128, input_shape=(x_train_cool.shape[1], x_train_cool.shape[2]), return_sequences=True))
    # model.add(Dropout(0.3))
    # model.add(LSTM(128, return_sequences=False))
    # model.add(Dropout(0.3))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    input_tensor = Input(shape=(x_train_cool.shape[1], x_train_cool.shape[2]))
    # x = LSTM(128, return_sequences=True)(input_tensor)
    # x = Dropout(0.3)(x)
    # x = LSTM(64, return_sequences=False)(x)
    # x = Dropout(0.3)(x)
    x = LSTM(48, return_sequences=True)(input_tensor)
    # x = Dropout(0.3)(x)
    x = LSTM(24, return_sequences=True)(x)
    x = LSTM(12, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    output_tensor = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, output_tensor)
    model.summary()
    adam = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mae', optimizer=adam, metrics=['accuracy'])
    # fit network
    history = model.fit(x_train_cool, y_train_cool, epochs=300, batch_size=72,
                        validation_data=(x_test_cool, y_test_cool), shuffle=False)
    model.save('./steam_LSTM1')
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    return model


model = build_model()
# model = load_model('./steam-LSTM')
# make a prediction
yhat = model.predict(x_test_cool)

# invert scaling for forecast
x_test_cool = x_test_cool.reshape((x_test_cool.shape[0], x_test_cool.shape[2]))
# inv_yhat = np.concatenate((yhat, x_test_cool[:-23, 1:]), axis=1)

for i in range(23):
    data[-23+i, 1] = yhat[i]

inv_yhat = scaler.inverse_transform(data)
inv_yhat = inv_yhat[-23:, 1]
print("inv_yhat: ", inv_yhat)

# invert scaling for actual
for i in range(23):
    data[-23+i, 1] = y_test_cool[i]

inv_y = scaler.inverse_transform(data)
inv_y = inv_y[-23:, 1]
print("inv_y: ", inv_y)

# 做ROC曲线
pyplot.figure()
pyplot.plot(range(len(inv_yhat)), inv_yhat, 'b', label="predict")
pyplot.plot(range(len(inv_y)), inv_y, 'r', label="test")
pyplot.legend(['predict', 'test'])  # 显示图中的标签
pyplot.show()

# calculate RMSE、MAPE
RMSE = sqrt(mean_squared_error(inv_y, inv_yhat))
MAPE = mean_absolute_error(inv_y, inv_yhat)
print('Test RMSE: %.3f ，Test MAPE: %.3f' % (RMSE, MAPE))

acc = (1 - np.mean(np.abs(inv_y - inv_yhat) / inv_y)) * 100
print("精准度：", acc, "%")  # 97.73722075679345 %

f = xlwt.Workbook()
sheet1 = f.add_sheet('steam')

# 进行关联度分析
for i in range(23):
    sheet1.write(i, 0, inv_y[i])
    sheet1.write(i, 1, inv_yhat[i])

f.save(r'F:/深度学习资料/XGBoost/SSA-LSTM/结果/steam/LSTM预测1.xls')

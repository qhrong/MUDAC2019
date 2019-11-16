import os
import warnings
import numpy as np
import pandas as pd
import talos
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

data = pd.read_csv('Train_March_X10_Y0_v1.csv', index_col=0)
data.set_index('date',inplace=True)
wavelet = pd.read_csv('wavelet.csv', )
wavelet.drop('Unnamed: 0', axis=1, inplace=True)
wavelet = wavelet.set_index(data.index)
data['price'] = wavelet
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data)
max0 = data.price.max()
min0 = data.price.min()
aa = (np.array(data.price) - min0)/ np.array([(max0-min0)]*len(data.price))

# split into train and test sets
values = scaled
n_train = 492 # 360*24
train = values[:n_train, :]
test = values[n_train:, :]

# split into input and outputs
train_X, train_y = train[:, 1:68], train[:, 0]
test_X, test_y = test[:, 1:68], test[:, 0]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


def lstm_model(train_X, train_y, test_X, test_y, params):
    # design network
    model = Sequential()
    model.add(LSTM(params['unit1'], input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(Activation(params['activation']))
    model.add(LSTM(params['unit2']))
    model.add(Dense(1))
    # model.add(Activation("linear"))
    model.compile(loss=params['loss'], optimizer=params['optimizer'])

    # fit network
    history = model.fit(train_X, train_y, epochs=params['epochs'], batch_size=params['batch_size'],
                        validation_data=(test_X, test_y), verbose=2, shuffle=False)
    return history, model

p = {'activation': ['relu', 'elu'],
   'loss': ['mse', 'mae'],
   'optimizer': ['Adam', 'Nadam'],
   'batch_size': [10], #range(0,500,step=10)
   'epochs': [500], #range(0,800,step=50)
   'unit1': [50], #range(80,130,step=10)
   'unit2': [50]} #range(40,50,60)

talos.Scan(train_X, train_y, x_val=test_X, y_val=test_y, model=lstm_model, params=p)


from __future__ import print_function
import os

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import RMSprop
import pandas as pd
import numpy as np

batch_size = 128
num_classes = 10
epochs = 3000
term_days = 500
csv_filenames = ['^N225.csv', '^DJI.csv', 'TM.csv', 'INTC.csv', 'GOOGL.csv', 'GS.csv', 'AMD.csv', 'COST.csv']
stock_count = len(csv_filenames)
# the data, split between train and test sets

# Load Datasets
csv_datas = []
for filename in csv_filenames:
    # csv_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'csv', filename))
    csv_data = pd.read_csv(os.path.join(os.getcwd(), 'datasets', 'csv', filename))
    data = pd.DataFrame(csv_data['Close']).T
    data.columns = csv_data['Date']
    csv_datas.append(data)

load_datas = np.asarray(pd.concat(csv_datas).dropna(axis=1))
datasets = []

# Reshape datasets
for i in range(len(load_datas[0]) - term_days):
    datasets.append(load_datas[:,i:i+term_days+1])

datasets = np.array(datasets)
percentage_datasets = datasets / datasets[:,:,term_days - 1].reshape(len(datasets), len(datasets[0]),1)
x_datasets = percentage_datasets[:,:,:term_days]
y_datasets = percentage_datasets[:,:,term_days]

train_data_length = int(len(x_datasets) / 1.3)
x_train = x_datasets[:train_data_length]
y_train = y_datasets[:train_data_length]

x_test = x_datasets[train_data_length:]
y_test = y_datasets[train_data_length:]

print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')
print(y_train[0])
print('stockcount', stock_count)

model = Sequential()
model.add(Dense(2000, activation='sigmoid', input_shape=(stock_count,term_days)))
model.add(Dropout(0.2))
model.add(Dense(500, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(stock_count, activation='linear'))

model.summary()
default_loss = np.abs(y_test - 1).sum()
print(default_loss)

model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

test_length = 50
result = model.predict(x_test[:test_length])
valid = 0
failed = 0
for i in range(test_length):
    print('Test data:  ', y_test[i])
    print('Result data:', result[i])
    for l in range(len(result[0])):
        if y_test[i][l] < 1 and result[i][l] < 1 or\
            y_test[i][l] >= 1 and result[i][l] >= 1:
            valid += 1
        else:
            failed += 1

print(valid)
print(failed)

from __future__ import print_function
import os

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import pandas as pd
import numpy as np

batch_size = 128
epochs = 30
term_days = 1500

# the data, split between train and test sets
csv_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'csv', '^N225.csv'))
chronologcal_data = np.asarray(csv_data['Close'].dropna())
datasets = []

for i in range(len(chronologcal_data) - term_days):
    datasets.append(chronologcal_data[i:i+term_days+1])

datasets = np.array(datasets)
percentage_datasets = datasets / datasets[:,term_days - 1].reshape(len(datasets), 1) * 1
x_datasets = percentage_datasets[:,0:term_days]
y_datasets = percentage_datasets[:,term_days]
print(percentage_datasets[0])
print(x_datasets[0])
print(y_datasets[0])

train_data_length = int(len(percentage_datasets) / 1.6)
x_train = x_datasets[:train_data_length]
y_train = y_datasets[:train_data_length]

x_test = x_datasets[train_data_length:]
y_test = y_datasets[train_data_length:]

print(x_train)
print(y_train)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Dense(500, activation='sigmoid', input_shape=(term_days,)))
# model.add(LSTM(self.n_hidden, batch_input_shape = (None, self.maxlen, self.n_in),
#              kernel_initializer = glorot_uniform(seed=20170719), 
#              recurrent_initializer = orthogonal(gain=1.0, seed=20170719)))
model.add(Dropout(0.2))
model.add(Dense(300, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

model.summary()
default_loss = y_test.sum() - (1 * len(y_test))
print(default_loss)

model.compile(loss='mse',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0.1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

result = model.predict(x_test[:5], verbose=0)
print(result)
print(y_test[:5])
for i in range(5):
    print(y_test[i], result[i])
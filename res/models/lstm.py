import os
import sys
import json
import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, CuDNNLSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

EPOCHS = 10
BATCH_SIZE = 128

print("loading data...")
name = sys.argv[1]
data = json.load(open(".{}/data.json".format(name), 'r'))
X = np.array(data["X"], dtype="float64")
Y = np.array(data["Y"], dtype="int32")

print("X: {}; Y: {}; {} buy; {} sell".format(X.shape, Y.shape, len([i for i in Y if np.argmax(i) == 0]), len([i for i in Y if np.argmax(i) == 1])))

np.random.seed(648060351)

save_file = "res/models/{}.hdf5".format(name) # TODO Get this from a Commandline Argument
checkpoint = ModelCheckpoint(save_file, monitor='val_acc', save_best_only=True, verbose=0, mode='max')

model = Sequential()
model.add(CuDNNLSTM(100, return_sequences=True, input_shape=X[0].shape))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(100))
model.add(Dropout(0.2))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.33, callbacks=[checkpoint])

'''
# conv_lstm_2.pb - 60.38 w/l; 954 closed; 395.20 pips; 1 open; -1.30 pips; Buy; [7.22/-9.96] avg pips;
model = Sequential()
model.add(CuDNNLSTM(100, return_sequences=True, input_shape=X[0].shape))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(100))
model.add(Dropout(0.2))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.33, callbacks=[checkpoint])
'''

'''
# conv_lstm.pb - 61.56 w/l; 731 closed; 389.70 pips; 1 open; -5.40 pips; Sell; [7.44/-10.52] avg pips;
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=X[0].shape))
model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(100))
model.add(Dropout(0.2))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.33, callbacks=[checkpoint])
'''
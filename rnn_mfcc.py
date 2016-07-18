from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import numpy as np
import mfcPreprocessor as mfcpp
# for reproducibility
np.random.seed(1337) 

# Directory where the mfc files are in
input = "../SPK_DB/mfcs"
# Ratio of tests vs input. Training set is (1 - this) of the input.
ratioOfTestsInInput = 0.1
# Cull samples which have <= this ratio of data points as non-zero values
percentageThreshold = 0.7
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
filter_len = 3

# number of samples before weight update
batch_size = 128
# number of possible classes. In this case, just 2 (TODO: should be 3 after adding noise)
nb_classes = 2
# how many iterations to run
nb_epoch = 12

((X_train, y_train), (X_test, y_test)) = mfcpp.run(input, testRatio = ratioOfTestsInInput, percentageThreshold = percentageThreshold)

print X_train.shape, y_train.shape, X_test.shape, y_test.shape
print X_train.dtype, y_train.dtype, X_test.dtype, y_test.dtype

# X_train:	input for the training set
# X_test:	input for the test set
# y_train:	result for the training set
# y_test:	result for the test set

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, filter_len, 1,
                        border_mode='valid',
                        input_shape=(1, X_train.shape[2], 1)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, filter_len, 1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,1)))


model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

start = time.clock()
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Time taken:', time.clock() - start)
print('Test score:', score[0])
print('Test accuracy:', score[1])
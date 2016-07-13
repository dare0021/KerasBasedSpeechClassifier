from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils

import numpy as np
import preprocessor as inp

# Cull samples which have <= this ratio of data points as non-zero values
percentageThreshold = 0.7
# in ms
frameSize = 30
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 2

batch_size = 128
# number of possible classes. In this case, just 2 (TODO: should be 3 after adding noise)
nb_classes = 2
nb_epoch = 12

((X_train, y_train), (X_test, y_test)) = inp.run(frameSize, percentageThreshold)

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

model.add(Convolution1D(nb_filters, nb_conv*8, 
                        border_mode='valid',
                        input_shape=(1, X_train.shape[2])))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(nb_pool)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

start = time.clock()
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=0, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print(time.clock() - start)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# 1) import wav file. DONE
# 2) carve up wav file in to (overlapping?) fragements by a certain timestep.
# 		This involves figuring out how a single frame lasts (use framerate?)
# 		Kinnuen & Li suggests 20~30ms is short and yields good results over longer (100s of ms) analyses
# 		Dynamic Time Warping is an alternative
# 		Velocity and acceleration (of the feature value) are possible additional feature data
# 3) design DNN to use said overlapping fragments with a true/false value for each timestep.
# 		note this is not speech recognition; this is speaker recognition
# 		there is no need for memory, and this is a classification issue
# 		i.e. more of a MNIST job than a LSTM job
# 		Kinnunen & Li suggests negative samples be provided and classified to "no one" to prevent false positives
# 		They also claim GMM can't handle high dimensional data, but Rouvier showed DNN is fine with 60
# 			Jain claims the feature vector should have less than 1/10 the number of speakers as its dimension. That's 0.2 features for our case.
# Other: MFC continues to dominate despite being from the 80's, and various modern attempts to create a better feature vector
# 			PLP is one of the more successful feature filters used along side MFC, despite being from 1990. Ceptra and spectral are more recent (both 2001)
# 			Different filters can be used to create different feature vectors of the same sample, tripling the samples available and increasing accuracy
# """
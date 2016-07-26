from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import numpy as np
import time
import mfcPreprocessor as mfcpp

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

X_set = None
Y_set = None

def shuffleTwoArrs(x, y):
    rng_state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(rng_state)
    np.random.shuffle(y)

# input: Directory(ies) where the mfc files are in
def prepareDataSet(input, unpredictableSeed = False, featureVectorSize = 13):
	global X_set
	global Y_set

	# for reproducibility
	if not unpredictableSeed:
		np.random.seed(1337)

	X_set, Y_set = mfcpp.run(input, percentageThreshold = percentageThreshold, featureVectorSize = featureVectorSize)
	
	print "Total size:", X_set.size

# X_train:	input for the training set
# X_test:	input for the test set
# y_train:	result for the training set
# y_test:	result for the test set
def getSubset(dropout):
	global X_set
	global Y_set

	shuffleTwoArrs(X_set, Y_set)

	X_set_i = []
	Y_set_i = []
	for i in range(X_set.size):
		if np.random.uniform(0,1) > dropout:
			# store
			X_set_i.append(X_set[i])
			Y_set_i.append(Y_set[i])
		# else ignore
	
	trainListSize = X_set.shape[0] // (1 / (1 - ratioOfTestsInInput))
	(X_train, X_test) = np.split(X_set_i, [trainListSize])
	(y_train, y_test) = np.split(Y_set_i, [trainListSize])

	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)

	return X_train, Y_train, X_test, Y_test

# Call prepareDataSet() first
# inputDrop is how much of the input to drop as a ratio [0,1]
def run(inputDrop = 0):
	X_train, Y_train, X_test, Y_test = getSubset(inputDrop)

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
	# Verbose 0: No output while processing
	# Verbose 1: Output each batch
	# Verbose 2: Output each epoch
	model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
	          verbose=2, validation_data=(X_test, Y_test))
	score = model.evaluate(X_test, Y_test, verbose=0)
	timeTaken = time.clock() - start
	print('Time taken:', timeTaken)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
	return (score[0], score[1], timeTaken)
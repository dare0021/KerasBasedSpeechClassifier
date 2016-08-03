from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

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

# input: Directory(ies) where the mfc files are in
def prepareDataSet(input, unpredictableSeed = False, featureVectorSize = 13, explicitTestSet = None):
	# for reproducibility
	if not unpredictableSeed:
		np.random.seed(1337)

	mfcpp.run(input, percentageThreshold = percentageThreshold, featureVectorSize = featureVectorSize, explicitTestSet = None)

# Evaluation function for collating the files' various time steps' predictions
# accThresh:	Files with accuracy above or equal this are counted as correct
def evaluate(model, accThresh = 0.5):
	testSpeakers = mfcpp.testSpeakers
	accSum = 0
	i = 0
	for s in testSpeakers:
		featVects = mfcpp.fileDict[s]
		truthVal = mfcpp.truthVals[s]
		for f in fileFeatVects:
			i += 1
			score = model.evaluate(f, np.full((len(f)), truthVal, dtype='int8'), verbose=0)
			if score[1] > accThresh:
				accSum += 1
	return ((float)(accSum)) / i
	# get feature vectors from a directory
	# recurse over the directory one file at a time
		# get the file's feature vectors
		# model.evaluate() over the vectors
		# parse the resultant accuracy (score[1]) as a correct or incorrect outcome
	# return the ratio of correct outcomes

# Call prepareDataSet() first
# inputDrop is how much of the input to drop as a ratio [0,1]
def run(inputDrop = 0, returnCustomEvalAccuracy = True):
	X_train, Y_train, X_test, Y_test = mfcpp.getSubset(nb_classes, inputDrop, ratioOfTestsInInput)

	print "X_train", X_train.shape
	print "Y_train", Y_train.shape
	print "X_test", X_test.shape
	print "Y_test", Y_test.shape

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
	if ratioOfTestsInInput > 0:
		score = model.evaluate(X_test, Y_test, verbose=0)
	timeTaken = time.clock() - start

	print('Time taken:', timeTaken)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
	print('Evaluator accuracy:', acc)
	if returnCustomEvalAccuracy:
		return (-1, evaluate(model), timeTaken)
	return (score[0], score[1], timeTaken)
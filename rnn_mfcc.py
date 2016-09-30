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
# number of possible classes
nb_classes = 3
# how many iterations to run
nb_epoch = 10
# how to bundle the MFCC vectors
windowSize = 50
# Files with accuracy above this are counted as correct
# 	Manually set due to the otherGroup messing with it
evaluateAccuracy = 0.5

saveWeightsTo = "weights"
saveModelsTo = "model"

# =======================================================
# Internal logic variables
# NOT SETTINGS
maxAccuracy = 0

# input: Directory(ies) where the mfc files are in
def prepareDataSet(input, unpredictableSeed, featureVectorSize, explicitTestSet):
	# for reproducibility
	if not unpredictableSeed:
		np.random.seed(1337)

	mfcpp.run(input, percentageThreshold, featureVectorSize, explicitTestSet, windowSize)

# Evaluation function for collating the files' various time steps' predictions
def evaluate(model, accThresh):
	testSpeakers = mfcpp.testSpeakers
	accSum = 0
	i = 0
	for s in testSpeakers:
		fileFeatVects = mfcpp.fileDict[s]
		truthVal = mfcpp.truthVals[s]
		for f in fileFeatVects:
			x = np.array(f)
			i += 1
			score = model.evaluate(x.reshape(x.shape[0], 1, x.shape[1], x.shape[2]), np_utils.to_categorical(np.full((len(f)), truthVal, dtype='int8'), nb_classes), verbose=0)
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
# decayLR:	The learning rate to use for time-based LR scheduling. 0 means no decay.
def run(inputDrop, flags):
	global maxAccuracy

	X_train, Y_train, X_test, Y_test = mfcpp.getSubset(nb_classes, inputDrop, ratioOfTestsInInput)

	print "X_train", X_train.shape
	print "Y_train", Y_train.shape
	print "X_test", X_test.shape
	print "Y_test", Y_test.shape

	model = Sequential()

	model.add(Convolution2D(nb_filters, filter_len, filter_len,
	                        border_mode='valid',
	                        input_shape=(1, windowSize, X_train.shape[3]),
	                        activation='relu'))
	model.add(Convolution2D(nb_filters, filter_len, filter_len, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes, activation='softmax'))

	model.compile(loss='categorical_crossentropy',
	              optimizer='adadelta',
	              metrics=['accuracy'])

	start = time.clock()
	# Verbose 0: No output while processing
	# Verbose 1: Output each batch
	# Verbose 2: Output each epoch
	model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
	          verbose=2, validation_data=(X_test, Y_test))
	score = [0, 0]
	if ratioOfTestsInInput > 0:
		score = model.evaluate(X_test, Y_test, verbose=0)
	timeTaken = time.clock() - start

	print('Time taken:', timeTaken)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	s = score[0]
	acc = score[1]
	if 'returnCustomEvalAccuracy' in flags:
		s = -1
		acc = evaluate(model, evaluateAccuracy)
		print('Evaluator accuracy:', acc)
	if 'saveMaxVer' in flags and maxAccuracy < acc:
		maxAccuracy = acc
		import os
		model.save_weights(saveWeightsTo + "_" + str(acc) + ".h5", overwrite = True)
		jsonData = model.to_json()
		with open (saveModelsTo + "_" + str(acc) + ".json", 'w') as f:
			f.write(jsonData)

	return (s, acc, timeTaken)
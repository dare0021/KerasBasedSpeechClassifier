from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
# There may come a time when we have to put these in to their respective functions
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

from keras.layers import Embedding
from keras.layers import LSTM

import numpy as np
import time
import mfcPreprocessor as mfcpp
import speakerInfo as sinfo

# Ratio of tests vs input. Training set is (1 - this) of the input.
ratioOfTestsInInput = 0.1

# number of samples before weight update
batch_size = 128
# how many iterations to run
nb_epoch = 50
# how to bundle the MFCC vectors
windowSize = 100
# Files with accuracy above this are counted as correct
# 	Manually set due to the otherGroup messing with it
evaluateAccuracy = 0.5

saveWeightsTo = "weights"
saveModelsTo = "model"

# =======================================================
# Internal logic variables
# NOT SETTINGS
maxAccuracy = 0
onlySaveBestOnes = False;

# dictionaries are passed by value
def windowDict(dic, featureVectorSize):
	out = dict()
	for k in dic.iterkeys():
		raw = dic[k]
		files = []
		for file in raw:
			frames = []
			for frame in file:
				frames.append(frame)
			numcells = len(frames) // windowSize
			if len(frames) % windowSize > 0:
				numcells += 1
			flat = np.array(frames).flatten()
			flat = np.append(flat, np.zeros(shape=(numcells*windowSize*featureVectorSize - len(flat))))
			files.append(np.array(flat).reshape(numcells, windowSize, featureVectorSize))
		out[k] = files
	return out

# Loads compacted data set
# Use mfcPreprocessor.pickleDataSet() to create pickles
def loadPickledDataSet(pickleName, featureVectorSize, samplingMode="random"):
	import cPickle as pickle
	with open("pickles/"+pickleName, 'rb') as f:
		mfcpp.fileDict, mfcpp.otherData, mfcpp.truthVals = pickle.load(f)
	print "LENS", len(mfcpp.fileDict), len(mfcpp.otherData), len(mfcpp.truthVals)
	mfcpp.fileDict = windowDict(mfcpp.fileDict, featureVectorSize)
	mfcpp.otherData = windowDict(mfcpp.otherData, featureVectorSize)
	mfcpp.samplingMode = samplingMode

# Loads data as is
# input: Directory(ies) where the mfc files are in
# use dropout to speed up training for whatever reason
def prepareDataSet(input, unpredictableSeed, featureVectorSize, dropout=0.0, samplingMode="random"):
	# for reproducibility
	if not unpredictableSeed:
		np.random.seed(1337)
	mfcpp.run(input, featureVectorSize, dropout)
	mfcpp.fileDict = windowDict(mfcpp.fileDict, featureVectorSize)
	mfcpp.otherData = windowDict(mfcpp.otherData, featureVectorSize)
	mfcpp.samplingMode = samplingMode

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
			score = model.evaluate(x, np_utils.to_categorical(np.full((len(f)), truthVal, dtype='int8'), sinfo.getNbClasses()), verbose=0)
			if score[1] > accThresh:
				accSum += 1
	return ((float)(accSum)) / i
	# get feature vectors from a directory
	# recurse over the directory one file at a time
		# get the file's feature vectors
		# model.evaluate() over the vectors
		# parse the resultant accuracy (score[1]) as a correct or incorrect outcome
	# return the ratio of correct outcomes

def formatOutput(model, score, startTime):
	global maxAccuracy

	timeTaken = time.clock() - startTime
	print('Time taken:', timeTaken)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	s = score[0]
	acc = score[1]
	if 'returnCustomEvalAccuracy' in flags:
		s = -1
		acc = evaluate(model, evaluateAccuracy)
		print('Evaluator accuracy:', acc)
	if 'saveMaxVer' in flags and ((not onlySaveBestOnes) or maxAccuracy < acc):
		if maxAccuracy < acc:
			maxAccuracy = acc
		import os
		model.save_weights(saveWeightsTo + "_" + str(acc) + ".h5", overwrite = True)
		jsonData = model.to_json()
		with open (saveModelsTo + "_" + str(acc) + ".json", 'w') as f:
			f.write(jsonData)

	return (s, acc, timeTaken)

# Call prepareDataSet() or loadPickledDataSet() first
# inputDrop is how much of the input to drop as a ratio [0,1]
# decayLR:	The learning rate to use for time-based LR scheduling. 0 means no decay.

def runCNN1D(inputDrop, flags):
	# weights
	nb_filters = 32
	# kernel size
	filter_len = 3
	# pool size
	nb_pool = 2

	X_train, Y_train, X_test, Y_test = mfcpp.getSubset(inputDrop, ratioOfTestsInInput)

	print "X_train", X_train.shape
	print "Y_train", Y_train.shape
	print "X_test", X_test.shape
	print "Y_test", Y_test.shape

	model = Sequential()

	model.add(Conv1D(nb_filters, filter_len,
	                 input_shape=(windowSize, X_train.shape[-1]),
	                 activation='relu'))
	model.add(Dropout(0.2))
	model.add(Conv1D(nb_filters, filter_len, activation='relu'))
	model.add(Dropout(0.2))
	model.add(MaxPooling1D(2))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(sinfo.getNbClasses(), activation='softmax'))

	model.compile(loss='categorical_crossentropy',
	              optimizer='adadelta',
	              metrics=['accuracy'])

	start = time.clock()
	# Verbose 0: No output while processing
	# Verbose 1: Output each batch
	# Verbose 2: Output each epoch
	model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
	          verbose=0, validation_data=(X_test, Y_test))
	score = [0, 0]
	if ratioOfTestsInInput > 0:
		score = model.evaluate(X_test, Y_test, verbose=0)
	
	return formatOutput(score, time.clock() - start)

def runCNN2D(inputDrop, flags):
	# weights
	nb_filters = 32
	# kernel size
	filter_len = 3
	# pool size
	nb_pool = 2

	X_train, Y_train, X_test, Y_test = mfcpp.getSubset(inputDrop, ratioOfTestsInInput)
	featureVectorSize = X_train.shape[-1]
	X_train = np.reshape(X_train, (-1, windowSize, X_train.shape[-1], 1))
	X_test = np.reshape(X_test, (-1, windowSize, X_test.shape[-1], 1))

	print "X_train", X_train.shape
	print "Y_train", Y_train.shape
	print "X_test", X_test.shape
	print "Y_test", Y_test.shape

	model = Sequential()

	model.add(Conv2D(nb_filters, (filter_len, filter_len),
					 # treat input as a 2D greyscale image
	                 input_shape=(windowSize, featureVectorSize, 1),
	                 activation='relu'))
	model.add(Dropout(0.2))
	model.add(Conv2D(nb_filters, (filter_len, filter_len), activation='relu'))
	model.add(Dropout(0.2))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(sinfo.getNbClasses(), activation='softmax'))

	model.compile(loss='categorical_crossentropy',
	              optimizer='adadelta',
	              metrics=['accuracy'])

	start = time.clock()
	# Verbose 0: No output while processing
	# Verbose 1: Output each batch
	# Verbose 2: Output each epoch
	model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
	          verbose=2, validation_data=(X_test, Y_test))
	score = [0, 0]
	if ratioOfTestsInInput > 0:
		score = model.evaluate(X_test, Y_test, verbose=0)
	
	return formatOutput(score, start)

run = runCNN2D
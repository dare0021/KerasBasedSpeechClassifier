import numpy as np
import unpackMFC as unmfc
import speakerInfo as sinfo
from os import listdir
from os.path import isfile, join
from keras.utils import np_utils

# speakerID -> 3D array of feature vectors grouped by file
fileDict = dict()
truthVals = dict()

shuffle = False
explicit_X_test = []
explicit_Y_test = []

testSpeakers = []

def strToArr(input):
	if type(input) is str:
		return [input]
	return input

def loadTestSetAuto(rootPath, featureVectorSize = 13):
	global explicit_X_test
	global explicit_Y_test

	rootPath = strToArr(rootPath)
	for p in rootPath:
		filesThisPath = [p + "/" + f for f in listdir(p) if isfile(join(p, f)) and f.endswith(".mfc")]
		for path in filesThisPath:
			data = unmfc.run(path, featureVectorSize = featureVectorSize)
			explicit_X_test.append(data)
			explicit_Y_test.extend(np.full((len(data)), sinfo.getTruthValue(path), dtype='int8'))

# rootPath is the string or an array of strings of paths of directories to use
# percentageThreshold is when to drop a feature vector by how much of it is zero
# not implemented in this version
# <1% drops for 0.7 
# >20% for 0.6
# Both of above for clean samples
# explicitTestSet[0] contains paths, while explicitTestSet[1] contains truth values
def run(rootPath, percentageThreshold = 0.7, featureVectorSize = 13, explicitTestSet = None):
	global fileDict
	global truthVals
	global explicit_X_test
	global explicit_Y_test

	rootPath = strToArr(rootPath)
	if explicitTestSet != None:
		explicitTestSet = strToArr(explicitTestSet[0])
		explicitTestSet = unmfc.runForAll(explicitTestSet, featureVectorSize)
		i = 0
		for f in explicitTestSet:
			explicit_X_test.extend(f)
			explicit_Y_test.extend(np.full((len(f)), explicitTestSet[1][i], dtype='int8'))
			i += 1

	print "Importing files..."
	fileCount = 0
	for p in rootPath:
		filesThisPath = [p + "/" + f for f in listdir(p) if isfile(join(p, f)) and f.endswith(".mfc")]
		fileCount += len(filesThisPath)
		print fileCount, "files found so far."
		for path in filesThisPath:
			sid = sinfo.getSpeakerID(path)
			data = unmfc.run(path, featureVectorSize = featureVectorSize)
			if sid in fileDict:
				fileDict[sid].append(data)
			else:
				fileDict[sid] = [data.tolist()]
				truthVals[sid] = sinfo.getTruthValue(path)

	print fileCount, " files found from"
	print len(fileDict), " speakers."

# x: ANN input MFCC data
# y: ANN output validation data
def collateData(speakerList):
	global fileDict
	global truthVals

	x = []
	y = []
	for s in speakerList:
		data = fileDict[s]
		for f in data:
			x.extend(f)
			y.extend(np.full((len(f)), truthVals[s], dtype='int8'))
	return x, y

def shuffleTwoArrs(x, y):
    rng_state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(rng_state)
    np.random.shuffle(y)

# X_train:	input for the training set
# X_test:	input for the test set
# y_train:	result for the training set
# y_test:	result for the test set
# dropout:	ratio of the superset to disregard
# dropout is done here to make sure the np.array generated is not bigger than it has to be
# since using Keras dropout would mean feeding everything in to the ANN
def getSubset(nb_classes, dropout, ratioOfTestsInInput):
	global fileDict
	global truthVals
	global shuffle
	global explicit_X_test
	global explicit_Y_test
	global testSpeakers

	if (len(explicit_Y_test) > 0) and (ratioOfTestsInInput > 0):
		print "ERR: can't have both explicit test set and randomly sampled tests from the training sample"
		import sys
		sys.exit()

	# generate chosen speakers
		# randomly drop speakers
		# randomly choose test set speakers

	# create X & Y sets
		# X set should be [None, 1, freq, 1]
		# TODO: use a 2D set of mfcc data instead of single time steps
		# Y set should be [None, 1] and match the X set in cardinality

	if len(explicit_Y_test) <= 0:
		speakerList = [s for s in fileDict if np.random.uniform(0,1) > dropout]
		np.random.shuffle(speakerList)

		trainListSize = len(speakerList) // (1 / (1 - ratioOfTestsInInput))
		speakersTrain, speakersTest = np.split(speakerList, [trainListSize])
		testSpeakers = speakersTest

		X_train, Y_train = collateData(speakersTrain)
		X_test, Y_test = collateData(speakersTest)

		if shuffle is True:
			shuffleTwoArrs(X_train, Y_train)
			shuffleTwoArrs(X_test, Y_test)

		Y_train = np_utils.to_categorical(Y_train, nb_classes)
		Y_test = np_utils.to_categorical(Y_test, nb_classes)

		X_train = np.array(X_train, dtype='float32')
		X_test = np.array(X_test, dtype='float32')

		X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], 1)
		X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], 1)

		print "Training set:", speakersTrain
		print "Testing  set:", speakersTest

		return X_train, Y_train, X_test, Y_test
	else:
		speakerList = [s for s in fileDict if np.random.uniform(0,1) > dropout]
		np.random.shuffle(speakerList)

		X_train, Y_train = collateData(speakerList)

		if shuffle is True:
			shuffleTwoArrs(X_train, Y_train)

		Y_train = np_utils.to_categorical(Y_train, nb_classes)
		Y_test = np_utils.to_categorical(explicit_Y_test, nb_classes)

		X_train = np.array(X_train, dtype='float32')
		X_test = np.array(explicit_X_test, dtype='float32')

		X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], 1)
		X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], 1)

		print "Training set:", speakersTrain
		print "Testing  set:", speakersTest
		
		return X_train, Y_train, X_test, Y_test


# for unit test
# run("../SPK_DB/mfc13")
# run("../SPK_DB/mfc60", featureVectorSize=60)
# run(["../SPK_DB/mfc13", "../SPK_DB/mfc60"])
# run("/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/VCTK-Corpus/mfc13")
# print fileDict
# print truthVals
# loadTestSetAuto("../SPK_DB/mfc13")
# loadTestSetAuto(["../SPK_DB/mfc13","/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/VCTK-Corpus/mfc13"])
# print np.array(explicit_X_test).shape
# print np.array(explicit_Y_test).shape
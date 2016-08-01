import numpy as np
import unpackMFC as unmfc
import speakerInfo as sinfo
from os import listdir
from os.path import isfile, join
from keras.utils import np_utils

fileDict = dict()
truthVals = dict()

shuffle = False

# rootPath is the string or an array of strings of paths of directories to use
# <1% drops for 0.7 
# >20% for 0.6
# Both of above for clean samples
def run(rootPath, percentageThreshold = 0.7, featureVectorSize = 13):
	global fileDict
	global truthVals

	rootPaths = []
	if type(rootPath) is str:
		rootPaths = [rootPath]
	else: 
		# Assume rootPath is an array
		rootPaths = rootPath

	print "Importing files..."
	fileCount = 0
	for p in rootPaths:
		filesThisPath = [p + "/" + f for f in listdir(p) if isfile(join(p, f)) and f.endswith(".mfc")]
		fileCount += len(filesThisPath)
		print fileCount, "files found so far."
		for path in filesThisPath:
			sid = sinfo.getSpeakerID(path)
			data = unmfc.run(path, featureVectorSize = featureVectorSize)
			if sid in fileDict:
				fileDict[sid].extend(data)
			else:
				fileDict[sid] = data.tolist()
				truthVals[sid] = sinfo.getTruthValue(f)

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
		x.extend(data)
		# ERR: y is too short. x & y size mismatch
		# difference is too large to be a boundary case. 303642 x vs 79399 y
		y.extend(np.full((len(data)), truthVals[s], dtype='int8'))
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

	# generate chosen speakers
		# randomly drop speakers
		# randomly choose test set speakers

	# create X & Y sets
		# X set should be [None, 1, freq, 1]
		# TODO: use a 2D set of mfcc data instead of single time steps
		# Y set should be [None, 1] and match the X set in cardinality

	speakerList = [s for s in fileDict if np.random.uniform(0,1) > dropout]
	trainListSize = len(speakerList) // (1 / (1 - ratioOfTestsInInput))
	np.random.shuffle(speakerList)
	speakersTrain, speakersTest = np.split(speakerList, [trainListSize])
	print "Training set:", speakersTrain
	print "Testing  set:", speakersTest

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

	return X_train, Y_train, X_test, Y_test


# for unit test
# out = run("../SPK_DB/mfc13")
# out = run("../SPK_DB/mfc60", featureVectorSize=60)
# out = run(["../SPK_DB/mfc13", "../SPK_DB/mfc60"])
# out = run("/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/VCTK-Corpus/mfc13")
# print out[0][0].shape
# print out[0][1].shape
# print out[1][0].shape
# print out[1][1].shape

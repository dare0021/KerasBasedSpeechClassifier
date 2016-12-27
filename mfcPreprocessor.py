import numpy as np
import unpackMFC as unmfc
import speakerInfo as sinfo
from os import listdir
from os.path import isfile, join
from keras.utils import np_utils

# speakerID -> 3D array of feature vectors grouped by file
fileDict = dict()
# Things that are not from speakers
# -1: Silences
otherData = dict()
otherDataKeys = []
truthVals = dict()

explicit_X_test = []
explicit_Y_test = []

testSpeakers = []

sidKeyType = sinfo.getSIDKeyType()
if "int" == sidKeyType:
	otherDataKeys = [-1]
elif "string" == sidKeyType:
	otherDataKeys = ['-1']
else:
	print "ERR: unknown speakerInfo.getSIDKeyType()", sidKeyType
	assert False

def strToArr(input):
	if type(input) is str:
		return [input]
	return input

def loadTestSetAuto(rootPath, featureVectorSize):
	global explicit_X_test
	global explicit_Y_test

	rootPath = strToArr(rootPath)
	for p in rootPath:
		filesThisPath = [p + "/" + f for f in listdir(p) if isfile(join(p, f)) and f.endswith(".mfc")]
		for path in filesThisPath:
			data = unmfc.run(path, featureVectorSize)
			explicit_X_test.append(data)
			explicit_Y_test.extend(np.full((len(data)), sinfo.getTruthValue(path), dtype='int8'))

def clean():
	global fileDict
	global otherData
	global otherDataKeys
	global truthVals
	global explicit_X_test
	global explicit_Y_test
	global testSpeakers
	fileDict = dict()
	otherData = dict()
	otherDataKeys = []
	truthVals = dict()
	explicit_X_test = []
	explicit_Y_test = []
	testSpeakers = []

# rootPath is the string or an array of strings of paths of directories to use
# Does not peek in to subdirectories
# percentageThreshold is when to drop a feature vector by how much of it is zero
# not implemented in this version
# <1% drops for 0.7 
# >20% for 0.6
# Both of above for clean samples
# explicitTestSet[0] contains paths, while explicitTestSet[1] contains truth values
def run(rootPath, percentageThreshold, featureVectorSize, explicitTestSet, windowSize):
	global fileDict
	global otherData
	global truthVals
	global explicit_X_test
	global explicit_Y_test

	rootPath = strToArr(rootPath)
	if explicitTestSet != None:
		explicitTestSet = strToArr(explicitTestSet[0])
		explicitTestSet = unmfc.runForAll(explicitTestSet, featureVectorSize, windowSize)
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
			data = unmfc.returnWindowed(path, featureVectorSize, windowSize)
			if sid in fileDict:
				fileDict[sid].append(data)
			elif sid in otherData:
				otherData[sid].append(data)
			elif sid in otherDataKeys:
				otherData[sid] = [data.tolist()]
				truthVals[sid] = sinfo.getTruthValue(path)
			else:
				fileDict[sid] = [data.tolist()]
				truthVals[sid] = sinfo.getTruthValue(path)

	print fileCount, " files found from"
	print len(fileDict), " speakers."

# x: ANN input MFCC data
# y: ANN output validation data
def collateData(speakerList):
	global fileDict
	global otherData
	global truthVals

	x = []
	y = []
	for s in speakerList:
		if s in fileDict:
			data = fileDict[s]
		elif s in otherData:
			data = otherData[s]
		else:
			print "ERR: unknown speaker", s
			print fileDict.keys()
			print otherData.keys()
			assert False
		for f in data:
			x.extend(f)
			y.extend(np.full((len(f)), truthVals[s], dtype='int8'))
	return x, y

# X_train:	input for the training set
# X_test:	input for the test set
# y_train:	result for the training set
# y_test:	result for the test set
# dropout:	ratio of the superset to disregard
# dropout is done here to make sure the np.array generated is not bigger than it has to be
# since using Keras dropout would mean feeding everything in to the ANN
def getSubset(dropout, ratioOfTestsInInput):
	global fileDict
	global otherData
	global truthVals
	global explicit_X_test
	global explicit_Y_test
	global testSpeakers

	if (len(explicit_Y_test) > 0) and (ratioOfTestsInInput > 0):
		print "ERR: can't have both explicit test set and randomly sampled tests from the training sample"
		assert False

	# generate chosen speakers
		# randomly drop speakers
		# randomly choose test set speakers

	# create X & Y sets
		# X set should be [None, 1, freq, 1]
		# Y set should be [None, 1] and match the X set in cardinality

	# Auto
	if len(explicit_Y_test) <= 0:
		print 'auto getSubset()'
		speakerList = [s for s in fileDict if np.random.uniform(0,1) > dropout]
		np.random.shuffle(speakerList)

		if ratioOfTestsInInput == 1:
			trainListSize = 0
		else:
			trainListSize = len(speakerList) // (1 / (1 - ratioOfTestsInInput))
		speakersTrain, speakersTest = np.split(speakerList, [trainListSize])
		testSpeakers = speakersTest

		otherGroup = []
		otherGroup.extend(otherData)
		speakersTrain = np.append(speakersTrain, otherGroup)
		X_train, Y_train = collateData(speakersTrain)
		X_test, Y_test = collateData(speakersTest)

		Y_train = np_utils.to_categorical(Y_train, sinfo.getNbClasses())
		Y_test = np_utils.to_categorical(Y_test, sinfo.getNbClasses())

		X_train = np.array(X_train, dtype='float32')
		X_test = np.array(X_test, dtype='float32')
	# Manual explicit test group
	else:
		print 'explicit manual test set'
		speakerList = [s for s in fileDict if np.random.uniform(0,1) > dropout]
		np.random.shuffle(speakerList)

		otherGroup = []
		otherGroup.extend(otherData)
		speakersTrain = np.append(speakersTrain, otherGroup)
		X_train, Y_train = collateData(speakerList)

		Y_train = np_utils.to_categorical(Y_train, sinfo.getNbClasses())
		Y_test = np_utils.to_categorical(explicit_Y_test, sinfo.getNbClasses())

		X_train = np.array(X_train, dtype='float32')
		X_test = np.array(explicit_X_test, dtype='float32')

	if X_train.shape[0] > 0:
		X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
		# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])	#LSTM
	else:
		print 'WARN: X_train.shape[0] <= 0', X_train.shape
	if X_test.shape[0] > 0:
		X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
		# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])	#LSTM
	else:
		print 'WARN: X_test.shape[0] <= 0', X_test.shape

	print "Training set:", speakersTrain
	print "Testing  set:", speakersTest
	
	return X_train, Y_train, X_test, Y_test


# for unit test
# run("../SPK_DB/mfc13")
# for x in fileDict[1]:
# 	print np.array(x).shape
# run("../SPK_DB/mfc60", featureVectorSize=60)
# run(["../SPK_DB/mfc13", "../SPK_DB/mfc60"])
# run("/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/VCTK-Corpus/mfc13")
# print fileDict
# print truthVals
# loadTestSetAuto("../SPK_DB/mfc13")
# loadTestSetAuto(["../SPK_DB/mfc13","/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/VCTK-Corpus/mfc13"])
# print np.array(explicit_X_test).shape
# print np.array(explicit_Y_test).shape
# print getSubset(3, 0, .1)
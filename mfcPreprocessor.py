import numpy as np
import unpackMFC as unmfc
import speakerInfo as sinfo
import random
from os import listdir
from os.path import isfile, join
from keras.utils import np_utils

# speakerID -> 3D array of feature vectors Zz
# fileDict[speakerID] = [file#][frame#][featVect]
fileDict = dict()
# Things that are not from speakers
# -1: Silences
otherData = dict()
otherDataKeys = []
truthVals = dict()

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

def clean():
	global fileDict
	global otherData
	global otherDataKeys
	global truthVals
	global testSpeakers
	fileDict = dict()
	otherData = dict()
	otherDataKeys = []
	truthVals = dict()
	testSpeakers = []


def pickleDataSet(input, featureVectorSize, pickleName):
	import cPickle as pickle
	run(input, featureVectorSize, 0)
	with open("pickles/"+pickleName, 'w') as f:
		pickle.dump((fileDict, otherData, truthVals), f, pickle.HIGHEST_PROTOCOL)	
	print "LENS", len(fileDict), len(otherData), len(truthVals)

# rootPath is the string or an array of strings of paths of directories to use
# Does not peek in to subdirectories
# <1% drops for 0.7 
# >20% for 0.6
# dropout in the function argument is for dropping files
def run(rootPath, featureVectorSize, dropout):
	global fileDict
	global otherData
	global truthVals

	rootPath = strToArr(rootPath)

	print "Importing files..."
	fileCount = 0
	for p in rootPath:
		filesThisPath = [p + "/" + f for f in listdir(p) if isfile(join(p, f)) and f.endswith(".mfc")]
		if dropout > 0.0:
			filesThisPath = random.sample(filesThisPath, int(len(filesThisPath)*(1-dropout)))
		fileCount += len(filesThisPath)
		print fileCount, "files found so far."
		for path in filesThisPath:
			sid = sinfo.getSpeakerID(path)
			data = unmfc.run(path, featureVectorSize)
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
	global testSpeakers

	# generate chosen speakers
		# randomly drop speakers
		# randomly choose test set speakers

	# create X & Y sets
		# X set should be [None, 1, freq, 1]
		# Y set should be [None, 1] and match the X set in cardinality

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
# pickleDataSet(["/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/oscaar_matlab"], 13, "test.pkl")
# pickleDataSet(["/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/oscaar_matlab", "/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/CSLU-7to10auda_matlab", "/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/VCTK-Corpus/mfc13NoSilences2e5"], 13, "oscar_clsu7_10_vctk2e5.pkl")
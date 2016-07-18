import numpy as np
import unpackMFC as unmfc
import sys
from os import listdir
from os.path import isfile, join

# data points <= threshold are counted as zero
# Samples with >= threshold ratio of its data points being zero are ignored
def removeZeroSamples(x, y, datapointThreshold = 0, percentageThreshold = 1):
	toRemove = np.array([])
	for i in range(x.shape[0]):
		zeros = 0
		total = 0
		for j in x[i]: # ERR: x is float64 and not an Iterable
			total += 1
			if j > datapointThreshold:
				zeros += 1
		if ((((float)(zeros)) / total) >= percentageThreshold):
			toRemove = np.append(toRemove, i)
	return (np.delete(x, toRemove, axis=0),	np.delete(y, toRemove))

def shuffleTwoArrs(x, y):
    rng_state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(rng_state)
    np.random.shuffle(y)

# returns whether the file path is for a male or female speaker
def getTruthValue(path):
	path = path[path.rfind("/")+1:]
	path = path[path.find("_")+1:]
	if path[0] == 'M':
		return 1
	elif path[0] == 'F':
		return 0
	else:
		print "mfcPreprocessor.getTruthValue() failed with input: ", path
		sys.exit()
		return -1

def run(rootPath, testRatio = 0.1, percentageThreshold = 0.7):
	fileList = [rootPath + "/" + f for f in listdir(rootPath) if isfile(join(rootPath, f)) and f.endswith(".mfc")]
	X_train_i = unmfc.run(fileList[0])
	y_train_i = np.full((X_train_i.shape[0]), getTruthValue(fileList[0]), dtype='int8')
	for f in fileList[1:]:
		data = unmfc.run(f)
		X_train_i = np.append(X_train_i, data, axis=0)
		y_train_i = np.append(y_train_i, np.full((data.shape[0]), getTruthValue(f), dtype='int8'))

	(X_train_i, y_train_i) = removeZeroSamples(X_train_i, y_train_i, percentageThreshold = percentageThreshold)

	shuffleTwoArrs(X_train_i, y_train_i)

	X_train_i = X_train_i.reshape(X_train_i.shape[0], 1, X_train_i.shape[1], 1)

	trainListSize = X_train_i.shape[0] // (1 / (1 - testRatio))
	(X_train_i, X_test_i) = np.split(X_train_i, [trainListSize])
	(y_train_i, y_test_i) = np.split(y_train_i, [trainListSize])
	
	return ((X_train_i, y_train_i), (X_test_i, y_test_i))

# for unit test
# out = run("../SPK_DB/mfcs")
# print out[0][0].shape
# print out[0][1].shape
# print out[1][0].shape
# print out[1][1].shape
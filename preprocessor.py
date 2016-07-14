import numpy as np
import wavOpener as wo
import overlappingSamples as oss
from os import listdir
from os.path import isfile, join

reducedInput = False


# not the python int max, int max for the wav file
# note we're ignoring that INT_MIN is (INT_MAX*-1)-1
INT_MAX = 2147483647

root = "../SPK_DB"
X_train_path = ["001_M3", "002_M4", "003_M3", "004_M3", "012_F2", "013_F3", "014_F2", "015_F3"]
y_train_data = [1, 1, 1, 1, 0, 0, 0, 0]
X_test_path = ["005_M4", "011_F3"]
y_test_data = [1, 0]

def generateFileList(rootPath):
	iterNum = 0
	X_train_i = np.array([])
	X_test_i = np.array([])
	y_train_i = np.array([])
	y_test_i = np.array([])

	if reducedInput:
		print "WARN: not using all available input"
		X_train_path = ["004_M3", "012_F2"]
		y_train_data = [1, 0]
	else:
		global X_train_path, y_train_data

	for p in X_train_path:
		a = root + "/" + p + "/C" + p + "_INDE"
		b = root + "/" + p + "/C" + p + "_SENT"
		filesThisFolder = np.append([a + "/" + f for f in listdir(a) if isfile(join(a, f)) and f.endswith(".wav")],
									[b + "/" + f for f in listdir(b) if isfile(join(b, f)) and f.endswith(".wav")])
		y_train_i = np.append(y_train_i, np.full((filesThisFolder.shape[0]), y_train_data[iterNum], dtype='int8'))
		X_train_i = np.append(X_train_i, filesThisFolder)
		iterNum += 1
	iterNum = 0
	for p in X_test_path:
		a = root + "/" + p + "/C" + p + "_INDE"
		b = root + "/" + p + "/C" + p + "_SENT"
		filesThisFolder = np.append([a + "/" + f for f in listdir(a) if isfile(join(a, f)) and f.endswith(".wav")],
									[b + "/" + f for f in listdir(b) if isfile(join(b, f)) and f.endswith(".wav")])
		y_test_i = np.append(y_test_i, np.full((filesThisFolder.shape[0]), y_test_data[iterNum], dtype='int8'))
		X_test_i = np.append(X_test_i, filesThisFolder)
		iterNum += 1
	return ((X_train_i, y_train_i), (X_test_i, y_test_i))

def readFile(input, framesPerItem):
	if not wo.open(input):
		print "Failed to open ", input
		sys.exit()
	bytarr = wo.read(nearZeroThreshold = 0.00).astype(float) / INT_MAX
	frames = np.size(bytarr) // framesPerItem
	if frames < (float)(np.size(bytarr)) / framesPerItem:
		frames += 1
		bytarr = np.append(bytarr, np.zeros(shape=(frames * framesPerItem - np.size(bytarr))))
	return bytarr.reshape((frames, framesPerItem))

def getFramerate(input):
	if not wo.open(input):
		print "Failed to open ", input
		sys.exit()
	return wo.getFramerate()

def fpschk(files):
	fps = -1
	for i in files:
		if fps == -1:
			fps = getFramerate(i)
		elif fps != getFramerate(i):
			sys.exit()
	print "FPS OK"
	print "FPS: ", fps
	return fps

# data points <= threshold are counted as zero
# Samples with >= threshold ratio of its data points being zero are ignored
def removeZeroSamples(x, y, datapointThreshold = 0, percentageThreshold = 1):
	toRemove = np.array([])
	for i in range(x.shape[0]):
		zeros = 0
		total = 0
		for j in x[i]:
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

def run(frameSize, percentageThreshold):
	((X_train_i, y_train_i), (X_test_i, y_test_i)) = generateFileList(root)
	fps = fpschk(np.append(X_train_i, X_test_i))
	framesPerItem = fps * frameSize / 1000

	fileCount = X_train_i.shape[0] + X_test_i.shape[0]

	print "importing files... Total count: " + str(fileCount)
	X_train = oss.run(readFile(X_train_i[0], framesPerItem))
	y_train = np.full((X_train.shape[0]), y_train_i[0], dtype='int8')
	inum = 1
	processedCount = 1
	for i in X_train_i[1:]:
		iter = oss.run(readFile(i, framesPerItem))
		X_train = np.append(X_train, iter, axis=0)
		y_train = np.append(y_train, np.full((iter.shape[0]), y_train_i[inum], dtype='int8'))
		inum += 1
		processedCount += 1
		if processedCount % 100 == 0:
			print "importing files: " + str(processedCount) + " / " + str(fileCount)

	X_test = oss.run(readFile(X_test_i[0], framesPerItem))
	y_test = np.full((X_test.shape[0]), y_test_i[0], dtype='int8')
	inum = 1
	for i in X_test_i[1:]:
		iter = oss.run(readFile(i, framesPerItem))
		X_test = np.append(X_test, iter, axis=0)
		y_test = np.append(y_test, np.full((iter.shape[0]), y_test_i[inum], dtype='int8'))
		inum += 1
		processedCount += 1
		if processedCount % 100 == 0:
			print "importing files: " + str(processedCount) + " / " + str(fileCount)

	(X_train, y_train) = removeZeroSamples(X_train, y_train, percentageThreshold = percentageThreshold)
	(X_test, y_test) = removeZeroSamples(X_test, y_test, percentageThreshold = percentageThreshold)

	shuffleTwoArrs(X_train, y_train)
	shuffleTwoArrs(X_test, y_test)

	X_train = X_train.reshape(X_train.shape[0], 1, framesPerItem, 1)
	X_test = X_test.reshape(X_test.shape[0], 1, framesPerItem, 1)

	return ((X_train, y_train), (X_test, y_test))

# for unit test
# out = run(30, .7)
# print out[0][0].shape
# print out[0][1].shape
# print out[1][0].shape
# print out[1][1].shape
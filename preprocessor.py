import numpy as np
import wavOpener as wo
import overlappingSamples as oss


# not the python int max, int max for the wav file
# note we're ignoring that INT_MIN is (INT_MAX*-1)-1
INT_MAX = 2147483647

X_train_path = ["wavs/f2-1.wav", "wavs/f2-3.wav", "wavs/f2-5.wav", 
				"wavs/m3-4.wav", "wavs/m3-8.wav", "wavs/m3-10.wav"]
y_train_data = [0, 0, 0, 1, 1, 1]
X_test_path = ["wavs/m3-12.wav", "wavs/f2-7.wav"]
y_test_data = [1, 0]


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

def fpschk():
	fps = -1
	for i in X_train_path:
		if fps == -1:
			fps = getFramerate(i)
		elif fps != getFramerate(i):
			sys.exit()
	for i in X_test_path:
		if fps != getFramerate(i):
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
	fps = fpschk()
	framesPerItem = fps * frameSize / 1000

	X_train = oss.thirdsWithAThirdUnique(readFile(X_train_path[0], framesPerItem))
	y_train = np.full((X_train.shape[0]), y_train_data[0], dtype='int8')
	inum = 1
	for i in X_train_path[1:]:
		iter = oss.thirdsWithAThirdUnique(readFile(i, framesPerItem))
		X_train = np.append(X_train, iter, axis=0)
		y_train = np.append(y_train, np.full((iter.shape[0]), y_train_data[inum], dtype='int8'))
		inum += 1

	X_test = oss.thirdsWithAThirdUnique(readFile(X_test_path[0], framesPerItem))
	y_test = np.full((X_test.shape[0]), y_test_data[0], dtype='int8')
	inum = 1
	for i in X_test_path[1:]:
		iter = oss.thirdsWithAThirdUnique(readFile(i, framesPerItem))
		X_test = np.append(X_test, iter, axis=0)
		y_test = np.append(y_test, np.full((iter.shape[0]), y_test_data[inum], dtype='int8'))
		inum += 1

	(X_train, y_train) = removeZeroSamples(X_train, y_train, percentageThreshold = percentageThreshold)
	(X_test, y_test) = removeZeroSamples(X_test, y_test, percentageThreshold = percentageThreshold)

	shuffleTwoArrs(X_train, y_train)
	shuffleTwoArrs(X_test, y_test)

	X_train = X_train.reshape(X_train.shape[0], 1, framesPerItem)
	X_test = X_test.reshape(X_test.shape[0], 1, framesPerItem)

	return ((X_train, y_train), (X_test, y_test))
import numpy as np
import struct

# CMU Sphinx 4 mfc file opener
# takes file path as input
# Sphinx uses feature vectors of length 13 by default
def run(input, featureVectorSize):
	file = open(input, 'r')
	size = struct.unpack('>i', ''.join(file.read(4)))[0]
	if ((float)(size)) / featureVectorSize - (float)(size // featureVectorSize) != 0:
		print "ERR: unpackMFC.run().featureVectorSize is inconsistent with the feature count read from the file given."
		print "File given: ", input
		print "Feature count read: ", size
		print "featureVectorSize: ", featureVectorSize
		assert False
	out = np.zeros(shape=(size/featureVectorSize, featureVectorSize))
	for i in range(size):
		out[i//featureVectorSize][i%featureVectorSize] = struct.unpack('>f', ''.join(file.read(4)))[0]
	return out

# Returns windowed result with 0 padding
# e.g. for frames 1,2,3,4,5: [[1,2,3], [4,5,0]]
# windowSize is in frames
# a frame is 10ms
# recommended value: 1~3 sec
def returnWindowed(input, featureVectorSize, windowSize):
	raw = run(input, featureVectorSize)
	numcells = len(raw) // windowSize
	if len(raw) % windowSize > 0:
		numcells += 1
	raw = raw.flatten()
	raw = np.append(raw, np.zeros(shape=(numcells*windowSize*featureVectorSize - len(raw))))
	return raw.reshape(numcells, windowSize, featureVectorSize)

def runForAll(input, featureVectorSize, windowSize):
	out = []
	if windowSize > 1:
		for i in input:
			out.append(returnWindowed(i, featureVectorSize, windowSize))
	else:
		for i in input:
			out.append(run(i, featureVectorSize))
	return out

# print returnWindowed("../SPK_DB/mfc13OnlySilences2e5/C002_M4_INDE_025.wav.mfc", 13, 100).shape

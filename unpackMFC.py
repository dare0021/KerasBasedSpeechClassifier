import numpy as np
import struct
import overlappingSamples as ols

# CMU Sphinx 4 mfc file opener
# takes file path as input
# Sphinx uses feature vectors of length 13 by default
def run(input, featureVectorSize):
	sign = 1;
	file = open(input, 'r')
	size = struct.unpack('>i', ''.join(file.read(4)))[0]
	if ((float)(size)) / featureVectorSize - (float)(size // featureVectorSize) != 0:
		print "ERR: unpackMFC.run().featureVectorSize is inconsistent with the feature count read from the file given."
		print "File given: ", input
		print "Feature count read: ", size
		print "featureVectorSize: ", featureVectorSize
		import sys
		sys.exit()
	out = np.zeros(shape=(size/featureVectorSize, featureVectorSize))
	for i in range(size):
		out[i//featureVectorSize][i%featureVectorSize] = struct.unpack('>f', ''.join(file.read(4)))[0]
	return out

# Returns windowed result
# e.g. for frames 1,2,3,4,5: [[1,2,3], [2,3,4], [3,4,5]]
def returnWindowed(input, featureVectorSize):
	raw = run(input, featureVectorSize)
	return ols.run(raw)

def runForAll(input, featureVectorSize):
	out = []
	for i in input:
		out.append(run(i, featureVectorSize))
	return out

# print returnWindowed("../SPK_DB/mfc13OnlySilences2e5/C002_M4_INDE_025.wav.mfc", 13).shape
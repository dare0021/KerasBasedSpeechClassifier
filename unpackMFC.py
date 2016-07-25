import numpy as np
import struct

# CMU Sphinx 4 mfc file opener
# takes file path as input
# Sphinx uses feature vectors of length 13 by default
def run(input, featureVectorSize):
	sign = 1;
	file = open(input, 'r')
	size = struct.unpack('>i', ''.join(file.read(4)))[0]
	if size / featureVectorSize - (float)(size // featureVectorSize) != 0:
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
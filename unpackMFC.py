import numpy as np
import struct

# takes file path as input
def run(input):
	sign = 1;
	file = open(input, 'r')
	size = struct.unpack('>i', ''.join(file.read(4)))[0]
	out = np.zeros(shape=(size/13, 13))
	for i in range(size):
		out[i//13][i%13] = struct.unpack('>f', ''.join(file.read(4)))[0]
	return out
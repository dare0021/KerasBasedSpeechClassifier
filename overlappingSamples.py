import numpy as np

# Example:
# in: 0-30 30-60 ...
# out: 0-30 20-50 40-70 ...
# Last sample might be padded up to its last theird but never more than that
# if the samples does not cleanly divide to thirds, the dividend is (int) of the float. i.e. drops the remainders.
def thirdsWithAThirdUnique(sequence):
	l3 = sequence.shape[1] // 3
	lf = sequence.shape[1]
	lt = (int)(((float)(sequence.shape[0]) - 1/3) * 3/2)
	out = np.zeros(shape=(lt,lf))
	out[0] = sequence[0]
	cursor = lf - l3
	sequence = sequence.flatten()
	for i in range(1, lt):
		nextCursor = cursor + lf - l3
		if nextCursor >= np.size(sequence):
			out[i] = np.append(sequence[cursor:], np.zeros(shape=(lf - np.size(sequence[cursor:]))))
			break
		out[i] = sequence[cursor : cursor + lf]
		cursor = nextCursor
	return out
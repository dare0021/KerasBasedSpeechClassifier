import numpy as np
import wave
import struct

waveFile = "No open file"
path = ""
# 0 is the least verbose
printLevel = 0

def open(input, verbose=0):
	global waveFile, path
	path = input
	printLevel = verbose
	waveFile = wave.open(input, 'r')

	if waveFile.getcompname() != 'not compressed':
		waveFile.close()
		print "ERR: file uses compression"
		print "NOT IMPLEMENTED"
		return False
	if printLevel > 0:
		print "Opening: ",path
		# all data in little endian (>)
		# for stereo, use int (i)
		# for mono, short (h)
		# to think about: returning stereo data as an array of length 2?
		print "Channels: ", waveFile.getnchannels()
	return True

def close():
	waveFile.close()

def isZero(i, threshold):
	if abs(i) > threshold:
		return i
	return 0

def read(nearZeroThreshold = 0):
	if printLevel > 0:
		print "Reading: ",path
	out = "nothing yet"
	length = waveFile.getnframes()
	leadingZeros = 0
	for i in range(0, length):
		waveData = waveFile.readframes(1)
		data = isZero(struct.unpack("<h", waveData)[0], nearZeroThreshold)
		if data != 0:
			out = np.zeros(shape=(length - leadingZeros))
			out[0] = data
			break
		else:
			leadingZeros += 1

	if out == "nothing yet":
		out = np.zeros(shape=(length))

	for i in range(1, length - leadingZeros):
	    waveData = waveFile.readframes(1)
	    data = isZero(struct.unpack("<h", waveData)[0], nearZeroThreshold)
	    out[i] = data

	trailingZeros = 0
	for i in range(length - leadingZeros - 1, 0, -1):
		if out[i] != 0:
			break
		else:
			trailingZeros += 1

	return out[0 : length - leadingZeros - trailingZeros].astype(int)

def getFramerate():
	return waveFile.getframerate()
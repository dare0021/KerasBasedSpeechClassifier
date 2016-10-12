import numpy as numpy
import matplotlib.pyplot as plt

input = "/home/jkih/projects/KerasBasedSpeechClassifier/saveData_Dropout/10mA_0.761203902762.txt"

data = []

def readData(intput):
	global data
	data = []
	f = open(input, 'r')
	print "file from", f.readline()
	s = f.readline()
	while ("[ " in s):
		s = s[2:len(s)-4]
		s = s.split(", ")
		thisarr = []
		for num in s:
			thisarr.append(float(num))
		data.append(thisarr)

		s = f.readline()

readData(input)
print data
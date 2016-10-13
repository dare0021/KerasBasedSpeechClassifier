import numpy as np
import matplotlib.pyplot as plt

cj60ba = "/home/jkih/projects/KerasBasedSpeechClassifier/saveData_Dropout/CJ60A_dither.txt"
cj60bc = "/home/jkih/projects/KerasBasedSpeechClassifier/saveData_Dropout/CJ60C_dither.txt"

input = cj60ba
target = 0

data = []
dataMax = []

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

def getMax(nb_classes):
	global dataMax
	dataMax = []
	for arr in data:
		dataMax.append(np.argmax(arr[:nb_classes]))

def getAccuracy(silentTreatment, target):
	global dataMax
	sum = 0.0
	total = len(data)
	if silentTreatment == "ignore silence":
		getMax(2)
		for i in dataMax:
			if i == target:
				sum += 1
	elif silentTreatment == "silence drop":
		getMax(3)
		total = 0
		for i in dataMax:
			if i == target:
				sum += 1
				total += 1
			elif i != 2:
				total += 1
	elif silentTreatment == "strictly no silence":
		getMax(3)
		for i in dataMax:
			if i == target:
				sum += 1
	else:
		sum = -1
		print "Unknown silentTreatment in outputVisualizer.getAccuracy()"
	print sum, '/', total, '=',sum / total, '\t',silentTreatment
	return sum / total

# savePath can be png or pdf
def getRawGraph(nb_classes, savePath = "", verbose = True):
	global data
	xaxis = np.arange(0, len(data))
	plt.figure(1)
	plt.subplot(111)	
	npcopy = np.array(data)
	for i in range(0, nb_classes):
		plt.plot(xaxis, npcopy[:,i], label='C'+str(i))
	plt.legend(loc='center right')
	if not verbose:
		plt.ioff()
	if savePath != "":
		plt.savefit(savePath, bbox_inches='tight')
	else:
		plt.show()

readData(input)
getAccuracy("ignore silence", target)
getAccuracy("silence drop", target)
getAccuracy("strictly no silence", target)
getRawGraph(3)
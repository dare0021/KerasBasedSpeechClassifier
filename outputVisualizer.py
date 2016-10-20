#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

cj60ba = "/home/jkih/projects/KerasBasedSpeechClassifier/inputData/CJ60A_dither.txt"
cj60bc = "/home/jkih/projects/KerasBasedSpeechClassifier/inputData/CJ60C_dither.txt"

input = cj60bc
target = 1
# 1 frame is 10ms
windowSize = 100

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

# stride set to 1
def getSlidingWindowModes(windowSize):
	global dataMax
	assert len(dataMax) > 0
	assert windowSize <= len(dataMax)
	modes = []
	nb_windows = len(dataMax) - windowSize + 1
	datalet = dataMax[:windowSize]
	modes.append(stats.mode(datalet)[0][0])
	for i in range(1, nb_windows):
		datalet.pop(0)
		datalet.append(dataMax[i])
		modes.append(stats.mode(datalet)[0][0])
	print "modes", modes
	return modes

# stride set to 1
def getSlidingWindowAverages(windowSize):
	global data
	assert len(data) > 0
	assert windowSize <= len(data)
	averages = []
	nb_windows = len(data) - windowSize + 1
	datalet = data[:windowSize]
	averages.append(np.mean(datalet, axis=0))
	for i in range(1, nb_windows):
		datalet.pop(0)
		datalet.append(data[i])
		averages.append(np.mean(datalet, axis=0))
	print "averages", averages
	return averages

def getSlidingWindowModeAccuracy(windowSize, target):
	modes = getSlidingWindowModes(windowSize)
	sum = 0.0
	for mode in modes:
		if mode == target:
			sum += 1
	print "SlidingWindow Mode Accuracy", sum, '/', len(modes),'=',sum/len(modes)
	return sum / len(modes)

def getSlidingWindowAverageAccuracy(windowSize, target):
	averages = getSlidingWindowAverages(windowSize)
	sum = 0.0
	for iter in averages:
		if np.argmax(iter) == target:
			sum += 1
	print "SlidingWindow Average Accuracy", sum, '/', len(averages), '=', sum/len(averages)
	return sum / len(averages)

# gets the delta confidence between the target class and the max confidence class
def getConfidenceDifferential(target):
	global data
	assert len(data) > 0
	retval = []
	for iter in data:
		retval.append(iter[target] - np.max(iter))
	print 'ConfidenceΔ', retval
	print 'ConfidenceΔσ', np.std(retval)
	return retval

def getAccuracy(silentTreatment, target):
	global data
	assert len(data) > 0
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
		print "Unknown silentTreatment in outputVisualizer.getAccuracy()"
		assert False
	print sum, '/', total, '=',sum / total, '\t',silentTreatment
	return sum / total

# savePath can be png or pdf
def getRawGraph(nb_classes, savePath = "", verbose = True):
	global data
	assert len(data) > 0
	xaxis = np.arange(0, len(data))
	plt.figure(1)
	plt.subplot(111)	
	npcopy = np.array(data)
	for i in range(0, nb_classes):
		plt.plot(xaxis, npcopy[:,i], label='C'+str(i))
	showGraph(plt, savePath, verbose)

def getCompressedGraph(nb_classes, compressionRatio, savePath = "", verbose = True):
	global data
	assert len(data) > 0
	xaxis = np.arange(0, len(data)//compressionRatio+1)
	plt.figure(1)
	plt.subplot(111)
	yval = []
	for i in range(0, len(data)//compressionRatio+1):
		sum = np.array([0.0,0,0])
		for arr in data[i:i + compressionRatio]:
			sum += np.array(arr)
		yval.append(sum / compressionRatio)
	yval = np.array(yval)
	for i in range(0, nb_classes):
		plt.plot(xaxis, yval[:,i], label='C'+str(i))
	showGraph(plt, savePath, verbose)

# Uses fuzzy logic to prevent a filled rect of a graph
def getFuzzyGraph(nb_classes, fuzziness, savePath = "", verbose = True):
	global data
	assert len(data) > 0
	xaxis = np.arange(0, len(data))
	plt.figure(1)
	plt.subplot(111)
	import fuzzyHelper as fuzzball
	fuzzball.init(nb_classes, fuzziness)
	yvals = []
	for arr in range(0, len(data)):
		yvals.append(fuzzball.push(np.argmax(arr)))
	plt.plot(xaxis, yvals)
	showGraph(plt, savePath, verbose)

def showGraph(plot, savePath, verbose):
	plot.legend(loc='center right')
	plot.gca().set_ylim([0,1])
	if not verbose:
		plot.ioff()
	if savePath != "":
		plot.savefit(savePath, bbox_inches='tight')
	else:
		plot.show()

def getStats(verbose = True):
	assert len(data) > 0
	npcopy = np.array(data)
	stddev = np.std(npcopy, axis = 0)
	mean = np.mean(npcopy, axis = 0)
	median = np.median(npcopy, axis = 0)
	if verbose:
		print "stddev", stddev
		print "mean", mean
		print "median", median
	return stddev, mean, median

readData(input)
getAccuracy("ignore silence", target)
getAccuracy("silence drop", target)
getAccuracy("strictly no silence", target)
getStats()
getSlidingWindowModeAccuracy(windowSize, target)
getSlidingWindowAverageAccuracy(windowSize, target)
getConfidenceDifferential(target)
# getRawGraph(3)
# getFuzzyGraph(3, 0)
getCompressedGraph(3,3)
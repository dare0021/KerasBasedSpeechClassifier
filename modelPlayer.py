# Takes a bunch of MFCC files from a single class, then puts them through
# a saved Keras model / weight combo

from keras.models import model_from_json
import numpy as np
import os, time

import mfcPreprocessor as mfcpp
import speakerInfo as si
si.overrideForModelPlayer()

modelFile = "saveData_0.1Drop/model_0.860061407652.json"
weightsFile = "saveData_0.1Drop/weights_0.860061407652.h5"
# Does not check subdirectoreis
input = "inputData/MF/F"
output = "inputData/clean/outC.txt"

# The ground truth value
targetClass = 1

unpredictableSeed = True
featureVectorSize = 13
# should be the same one used during training
windowSize = 100
# drop rate for the whole data set
inputDropWhole = 0
# drop rate for each time the model is run
inputDropIter = 0

# ====================================
# Internal logic variables
sumCorrect = 0
sumTotal = 0

def evaluate(vect):
	if vect[0] > vect[1]:
		return 0
	return 1

# dictionaries are passed by value
def windowDict(dic, featureVectorSize):
	out = dict()
	for k in dic.iterkeys():
		raw = dic[k]
		files = []
		for file in raw:
			frames = []
			for frame in file:
				frames.append(frame)
			numcells = len(frames) // windowSize
			if len(frames) % windowSize > 0:
				numcells += 1
			flat = np.array(frames).flatten()
			flat = np.append(flat, np.zeros(shape=(numcells*windowSize*featureVectorSize - len(flat))))
			files.append(np.array(flat).reshape(numcells, windowSize, featureVectorSize))
		out[k] = files
	return out

def generateOutput(model, parentDir):
	mfcpp.run(parentDir, featureVectorSize, inputDropWhole)
	mfcpp.fileDict = windowDict(mfcpp.fileDict, featureVectorSize)
	mfcpp.otherData = windowDict(mfcpp.otherData, featureVectorSize)

	X_train, Y_train, X_test, Y_test = mfcpp.getSubset(inputDropIter, 1)

	print "X_train", X_train.shape
	print "Y_train", Y_train.shape
	print "X_test", X_test.shape
	print "Y_test", Y_test.shape
	
	return model.predict_proba(X_test)

def saveGeneratedData(data, path):
	global sumCorrect, sumTotal

	i = 1
	while os.path.isfile(path):
		path = path[:len(path)-len(str(i))] + str(i)
		i += 1
	stringData = time.asctime() + "\n"
	listTargetProb = []
	for vect in data:
		cls = evaluate(vect)
		if cls == targetClass:
			sumCorrect += 1
		sumTotal += 1
		stringData += "[ "
		for prob in vect:
			stringData += str(prob) + ", "
		stringData = stringData[:len(stringData)-2]
		stringData += "] " + str(cls) + "\n"
	acc = ((float)(sumCorrect)) / sumTotal
	stringData += "Accuracy: " + str(acc) + " (" + str(sumCorrect) + " / " + str(sumTotal) + ")\n"
	with open(path, 'w') as f:
		f.write(stringData)
	print stringData
	return acc

def clean():
	global sumCorrect
	global sumTotal

	sumCorrect = 0
	sumTotal = 0
	mfcpp.clean()

def run(input, output):
	if not unpredictableSeed:
		np.random.seed(1337)
	with open(modelFile, 'r') as f:
		model = f.read()
	model = model_from_json(model)
	model.load_weights(weightsFile)
	model.compile(loss='categorical_crossentropy',
	              optimizer='adadelta')
	genData = generateOutput(model, input)
	return saveGeneratedData(genData, output)

def multiRun(input, output, weightsFolder):
	global modelFile
	global weightsFile

	max = 0.0
	sum = 0.0
	maxFile = ""
	filesThisPath = [weightsFolder + "/" + f for f in os.listdir(weightsFolder) if os.path.isfile(os.path.join(weightsFolder, f)) and f.endswith(".h5")]
	for path in filesThisPath:
		clean()
		weightsFile = path
		modelFile = path[:path.rfind("weights_")] + "model_"
		prob = path[path.rfind("_")+1:]
		prob = prob[:len(prob)-3]
		modelFile += prob + ".json"
		print "run", modelFile
		acc = run(input, output + "/" + prob + ".txt")
		sum += acc
		if acc > max:
			max = acc
			maxFile = weightsFile

	print "Max Accuracy:", max
	print "Using:", maxFile
	print "Avg:", sum / len(filesThisPath)
	with open(output + "/_max.txt", 'w') as f:
		f.write(str(max) + "\n" + maxFile)

# run(input, output)
# change the targetClass variable 0 - adult, 1 - child
multiRun("inputData/sukwoo/C", "saves/2017jan.Feasibility/C", "saves/2017jan.Feasibility/weights")
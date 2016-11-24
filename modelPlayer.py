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
input = "inputData/clean/M"
output = "inputData/clean/outC.txt"

# The ground truth value
targetClass = 0

unpredictableSeed = True
percentageThreshold = 0.7
featureVectorSize = 13
explicitTestSet = None
windowSize = 300
inputDrop = 0

# ====================================
# Internal logic variables
sumCorrect = 0
sumTotal = 0

def evaluate(vect):
	if vect[0] > vect[1]:
		return 0
	return 1

def generateOutput(model, parentDir):
	mfcpp.run(parentDir, percentageThreshold, featureVectorSize, explicitTestSet, windowSize)
	X_train, Y_train, X_test, Y_test = mfcpp.getSubset(inputDrop, 1)
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
multiRun(input, "saveData_MF300_0.1/M", "saveData_MF300_0.1/weights")
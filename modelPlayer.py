from keras.models import model_from_json

modelFile = ""
weightsFile = ""
input = ""
output = ""

def generateOutput(model, parentDir):
	outputData = []
	for pname in listdir(parentDir):
		if path.isdir(parentDir + pname):
			outputThisIter = generateOutput(parentDir + pname + "/")
			for vect in outputThisIter:
				outputData.append(vect)
		elif path.isfile(parentDir + pname) and pname.endswith(".mfc"):
			outputData.append(model.predict_proba(parentDir + pname))
	return outputData

def saveGeneratedData(data, path):
	i = 1
	while path.isfile(path):
		path = path[:len(path)-len(str(i))] + str(i)
		i += 1
	stringData = time.asctime() + "\n"
	for vect in data:
		stringData += "[ "
		for prob in vect:
			stringData += str(prob) + ", "
		stringData = stringData[:len(stringData)-2]
		stringData += "]\n"
	with open(path, 'w') as f:
		f.write(stringData)
	print stringData

def run(input, output):
	with open(modelFile, 'r') as f:
		model = f.read()
	model = model_from_json(model)
	model.load_weights(weightsFile)
	model.compile(loss='categorical_crossentropy',
	              optimizer='adadelta',
	              metrics=['accuracy'])
	genData = generateOutput(model, input)
	saveGeneratedData(genData, ouput)
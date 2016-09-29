from keras.models import model_from_json
import mfcPreprocessor as mfcpp

modelFile = ""
weightsFile = ""
# Does not check subdirectoreis
input = ""
output = ""

unpredictableSeed = True
percentageThreshold = 0.7
featureVectorSize = 13
explicitTestSet = None
windowSize = 50
inputDrop = 0
nb_classes = 3

def generateOutput(model, parentDir):
	mfcpp.run(parentDir, percentageThreshold, featureVectorSize, explicitTestSet, windowSize)
	X_train, Y_train, X_test, Y_test = mfcpp.getSubset(nb_classes, inputDrop, 1)
	return model.predict_proba(X_test)

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
	if not unpredictableSeed:
		np.random.seed(1337)
	with open(modelFile, 'r') as f:
		model = f.read()
	model = model_from_json(model)
	model.load_weights(weightsFile)
	model.compile(loss='categorical_crossentropy',
	              optimizer='adadelta',
	              metrics=['accuracy'])
	genData = generateOutput(model, input)
	saveGeneratedData(genData, ouput)

run(input, output)
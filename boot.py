import time
import rnn_mfcc
import rnn_raw

# Number of iterations of traning / testing sets to run
numIterations = 20

# If false will use seed "1337" instead of current time
useFreshRngSeeds = True

inputDrop = 0.0

# possible flags:
# returnCustomEvalAccuracy
# 	Use rnn_mfcc.evaluate() instead of Keras built in accuracy metric
# saveMaxVer
# 	Save the best models & weights
flags = [
# 'returnCustomEvalAccuracy', 
'saveMaxVer']

start = time.clock()

avgTime = -1
avgScore = -1
avgAcc = -1
# Uncomment the section(s) to use
# Positioning determines whether new ones are used each iteration
# If input dropout is 0, there is no reason to reload the data set

# rnn_mfcc.prepareDataSet(["../SPK_DB/mfc13NoSilences2e5", "../SPK_DB/mfc13OnlySilences2e5"], unpredictableSeed = useFreshRngSeeds, featureVectorSize = 13)
# rnn_mfcc.prerpareDataSet(["../SPK_DB/mfc13NoSilences2e5", "../SPK_DB/mfc13OnlySilences2e5", "/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/VCTK-Corpus/mfc13NoSilences2e5", "/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/VCTK-Corpus/mfc13OnlySilences2e5"], unpredictableSeed = useFreshRngSeeds, featureVectorSize = 13)
# rnn_mfcc.prepareDataSet(["/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/VCTK-Corpus/mfc13noisyNoSilences8.4e6", "/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/VCTK-Corpus/mfc13noisyOnlySilences8.4e6", "/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/CSLU-Corpus/mfc13Set1NoSilences8.4e6", "/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/CSLU-Corpus/mfc13Set1OnlySilences8.4e6"], unpredictableSeed = useFreshRngSeeds, featureVectorSize = 13)
# rnn_mfcc.prepareDataSet(["/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/VCTK-Corpus/wav16normalizedNoisy", "/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/CSLU-Corpus/1234normalized"], unpredictableSeed = useFreshRngSeeds, featureVectorSize = 13)
# rnn_mfcc.prepareDataSet(["/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/VCTK-Corpus/mfc13NoSilences2e5", "/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/CSLU-7to10auda_matlab"], unpredictableSeed = useFreshRngSeeds, featureVectorSize = 13)
# rnn_mfcc.prepareDataSet(["/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/oscaar_matlab", "/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/CSLU-7to10auda_matlab", "/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/VCTK-Corpus/mfc13NoSilences2e5"], unpredictableSeed = useFreshRngSeeds, featureVectorSize = 13, dropout = 0.9)
rnn_mfcc.prepareDataSet(["/home/jkih/projects/KerasBasedSpeechClassifier/inputData/SKW"], unpredictableSeed = useFreshRngSeeds, featureVectorSize = 13, samplingMode = "oneAtATime")

# rnn_mfcc.loadPickledDataSet("oscar_clsu7_10_vctk2e5.pkl", 13)
# rnn_mfcc.loadPickledDataSet("test.pkl", 13)

for i in range(numIterations):
	print "___________________________"
	print "Starting iteration: ", i+1, " / ", numIterations
	print "___________________________"
	# All .mfc files should be present in the directory given using the parameter "input"
	score = rnn_mfcc.run(inputDrop, flags)
	# input is hard coded in to preprocessor.py
	# score = rnn_raw.run(unpredictableSeed = useFreshRngSeeds)
	if avgTime < 0:
		avgTime = score[2]
		avgScore = score[0]
		avgAcc = score[1]
	else:
		avgTime += score[2]
		avgScore += score[0]
		avgAcc += score[1]

print "___________________________"
print "Total iime taken:\t", time.clock() - start
print "Average time taken:\t", avgTime/numIterations
print "Average score:\t", avgScore/numIterations
print "Average accuracy:\t", avgAcc/numIterations
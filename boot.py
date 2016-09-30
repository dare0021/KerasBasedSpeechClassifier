import time
import rnn_mfcc
import rnn_raw

# Number of iterations of traning / testing sets to run
numIterations = 50

# If false will use seed "1337" instead of current time
useFreshRngSeeds = True

# possible flags:
# returnCustomEvalAccuracy
# 	Use rnn_mfcc.evaluate() instead of Keras built in accuracy metric
# saveMaxVer
# 	Save the best models & weights
flags = ['returnCustomEvalAccuracy', 'saveMaxVer']

start = time.clock()

avgTime = -1
avgScore = -1
avgAcc = -1
# Uncomment the section(s) to use
ets = None
# ets = [[], \
# 		[]]
rnn_mfcc.prepareDataSet(["../SPK_DB/mfc13NoSilences2e5", "../SPK_DB/mfc13OnlySilences2e5"], unpredictableSeed = useFreshRngSeeds, featureVectorSize = 13, explicitTestSet = ets)
# rnn_mfcc.prepareDataSet(["../SPK_DB/mfc13NoSilences2e5", "../SPK_DB/mfc13OnlySilences2e5", "/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/VCTK-Corpus/mfc13NoSilences2e5", "/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/VCTK-Corpus/mfc13OnlySilences2e5"], unpredictableSeed = useFreshRngSeeds, featureVectorSize = 13, explicitTestSet = ets)
# rnn_mfcc.prepareDataSet(["/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/VCTK-Corpus/mfc13noisyNoSilences8.4e6", "/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/VCTK-Corpus/mfc13noisyOnlySilences8.4e6", "/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/CSLU-Corpus/mfc13Set1NoSilences8.4e6", "/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/CSLU-Corpus/mfc13Set1OnlySilences8.4e6"], unpredictableSeed = useFreshRngSeeds, featureVectorSize = 13, explicitTestSet = ets)

for i in range(numIterations):
	print "___________________________"
	print "Starting iteration: ", i+1, " / ", numIterations
	print "___________________________"
	# All .mfc files should be present in the directory given using the parameter "input"
	score = rnn_mfcc.run(0, flags)
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
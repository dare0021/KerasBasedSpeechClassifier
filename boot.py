import time
import rnn_mfcc
import rnn_raw

# Number of iterations of traning / testing sets to run
numIterations = 100

# If false will use seed "1337" instead of current time
useFreshRngSeeds = True

start = time.clock()

avgTime = -1
avgScore = -1
avgAcc = -1
# Uncomment the section to use
for i in range(numIterations):
	print "___________________________"
	print "Starting iteration: ", i+1, " / ", numIterations
	print "___________________________"
	# All .mfc files should be present in the directory given using the parameter "input"
	score = rnn_mfcc.run(unpredictableSeed = useFreshRngSeeds)
	# input is hard coded in to preprocessor.py
	# score = rnn_raw.run(unpredictableSeed = useFreshRngSeeds)
	if avgScore < 0:
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

# TODO: test with 60 features instead of the current default 13
# 		Also, fiddle around with the ANN settings.
# Contains information about the files being used
# Separated out since 1) it's long, and 2) the code should change according to what files are used

import re
import sys

# automatically assigned unique speaker
uid = 999

def getUID():
	global uid
	uid += 1
	return uid

def regexMatch(pattern, string):
	return bool(re.compile(pattern).match(string))

# returns whether the file path is for a male or female speaker
# 0: Adult
# 1: Child
# 2: Silence
def getTruthValue(path):
	if 'OnlySilences' in path:
		return 2
	path = path[path.rfind("/")+1:]
	# regex match increases running time by 3%
	if regexMatch("C\d{3}_[MF]\d_(INDE|SENT)_\d{3}\.wav\.mfc", path):
		return 0
	elif regexMatch("p\d{3}_\d{3}\.wav\.mfc", path):
		return 0
	elif "CSLU" in directory:
		return 1
	print "mfcPreprocessor.getTruthValue() failed with input: ", path
	sys.exit()
	return -1

# -1: Silences
def getSpeakerID(path):
	if 'OnlySilences' in path:
		return -1
	fileName = path[path.rfind("/")+1:]
	directory = path[:path.rfind("/")]
	# regex match increases running time by 3%
	if regexMatch("C\d{3}_[MF]\d_(INDE|SENT)_\d{3}\.wav\.mfc", fileName):
		return int(fileName[1:4])
	elif regexMatch("p\d{3}_\d{3}\.wav\.mfc", fileName):
		return int(fileName[1:4])
	elif "CSLU" in directory:
		return getUID()
	print "mfcPreprocessor.getSpeakerID() failed with input: ", path
	sys.exit()
	return -1
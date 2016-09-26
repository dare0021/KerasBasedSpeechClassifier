# Contains information about the files being used
# Separated out since 1) it's long, and 2) the code should change according to what files are used

import re
import sys

# automatically assigned unique speaker
# uid = 999

# def getUID():
# 	global uid
# 	uid += 1
# 	return uid

def regexMatch(pattern, string):
	return bool(re.compile(pattern).match(string))

# returns whether the file path is for a male or female speaker
# 0: Adult
# 1: Child
# 2: Silence
def getTruthValue(path):
	if 'OnlySilences' in path:
		return 2
	fileName = path[path.rfind("/")+1:]
	directory = path[:path.rfind("/")]
	# regex match increases running time by 3%
	if regexMatch("C\d{3}_[MF]\d_(INDE|SENT)_\d{3}\.wav\.mfc", fileName):
		return 0
	elif regexMatch("p\d{3}_\d{3}\.wav\.mfc", fileName):
		return 0
	elif "CSLU" in directory:
		return 1
	print "infoAdultChild.getTruthValue() failed with input:", directory
	sys.exit()
	return -1

# -1: Silences
def getSpeakerID(path):
	if 'OnlySilences' in path:
		return '-1'
	fileName = path[path.rfind("/")+1:]
	directory = path[:path.rfind("/")]
	# regex match increases running time by 3%
	if regexMatch("C\d{3}_[MF]\d_(INDE|SENT)_\d{3}\.wav\.mfc", fileName):
		return fileName[1:4]
	elif regexMatch("p\d{3}_\d{3}\.wav\.mfc", fileName):
		return fileName[1:4]
	elif "CSLU" in directory:
		return fileName[2:5]
	print "infoAdultChild.getSpeakerID() failed with input:", path
	sys.exit()
	return -1

def getSIDKeyType():
	return "string"

# print getSpeakerID("/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/CSLU-Corpus/mfc13Set1NoSilences8.4e6/ks000010.wav.mfc")
# print getSpeakerID("ks000010.wav.mfc")

# print getSpeakerID("/media/jkih/4A98B4D598B4C12D/Users/jkih/Desktop/CSLU-Corpus/mfc13k123/5/ks012/ks012xx0.wav.mfc") #012
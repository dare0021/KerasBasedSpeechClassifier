# Contains information about the files being used
# Separated out since 1) it's long, and 2) the code should change according to what files are used

import re

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
	s,type,suffix = fileName.split(' ')
	if type == 'c':
		return 1
	if type == 'm':
		return 0
	print "infoSKW.getTruthValue() failed with input:", path
	assert False
	return -1

# -1: Silences
def getSpeakerID(path):
	if 'OnlySilences' in path:
		return '-1'
	fileName = path[path.rfind("/")+1:]
	s,type,suffix = fileName.split(' ')
	return s

def getSIDKeyType():
	return "string"

def getNbClasses():
	return 2	# 3 -> 2
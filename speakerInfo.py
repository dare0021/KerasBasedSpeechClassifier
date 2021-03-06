# Interface for file info
# So I don't have to alter other files to change the classifier data

import re
import sys

# import infoMaleFemale as moduleToUse
# import infoAdultChild as moduleToUse
import infoSKW as moduleToUse

# returns whether the file path is for a male or female speaker
# 0: Male
# 1: Female
# 2: Silence
# DType is int8, [-128, 127]
def getTruthValue(path):
	return moduleToUse.getTruthValue(path)

# -1: Silences
def getSpeakerID(path):
	return moduleToUse.getSpeakerID(path)

# nparray only supports one type per array
def getSIDKeyType():
	return moduleToUse.getSIDKeyType()

def getNbClasses():
	return moduleToUse.getNbClasses()

def overrideForModelPlayer():
	global moduleToUse
	import infoUID as moduleToUse

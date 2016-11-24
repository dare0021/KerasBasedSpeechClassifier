# Effectively a dummy file		
		
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
	return 0		
		
#1: Silences		
def getSpeakerID(path):		
	if 'OnlySilences' in path:		
		return '-1'		
	return str(uid)		
		
def getSIDKeyType():		
	return "string"		
		
def getNbClasses():		
	return 3 

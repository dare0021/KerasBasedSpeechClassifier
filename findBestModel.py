import os
import numpy as np
parentFolder = "/home/jkih/projects/KerasBasedSpeechClassifier/saves/"
projectFolder = parentFolder + "2017jan.Feasibility/25epochs/"

# Used to find the model with the best results for all samples across given directories
# Checks via the file name. Same file name == same model
# Checks the accuracy by finding the last line, then checking for "Accuracy: "%float%" ("
# Does not actually check for "Accuracy: ", instead just starts at the 10th char until " ("
# If the last char is \n, the last line will be "" instead of the previous line

inputFolders = [projectFolder+"junhong.c1a1sc1.A", projectFolder+"junhong.c1a1sc1.C"]

def run(inputFolders):
	accuracyArr = []
	for folderPath in inputFolders:
		print "folder", folderPath
		iter = dict()
		filesThisPath = [folderPath + "/" + f for f in os.listdir(folderPath) if os.path.isfile(os.path.join(folderPath, f)) and f.endswith(".txt") and not f[0] == '_']
		for filePath in filesThisPath:
			print "open", filePath
			with open(filePath, 'r') as file:
				line = file.readlines()[-1]
				acc = float(line[10:line.find(' (')])
				fileName = filePath[filePath.rfind("/")+1:]
				iter[fileName] = acc
		accuracyArr.append(iter)

	maxval = 0
	maxarr = []
	maxkey = ""
	for key in accuracyArr[0].keys():
		iterList = []
		# attempt to build an accuracy list
		# skip if model is not present for all samples
		allPresent = True
		for dic in accuracyArr:
			if not (key in dic.keys()):
				allPresent = False
				break
			else:
				iterList.append(dic[key])
		if not allPresent:
			continue
		result = np.array(iterList).mean()
		if result > maxval:
			maxval = result
			maxkey = key
			maxarr = iterList
	print maxkey, maxval, iterList

run(inputFolders)
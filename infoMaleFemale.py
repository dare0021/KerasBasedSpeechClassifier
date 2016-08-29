# Contains information about the files being used
# Separated out since 1) it's long, and 2) the code should change according to what files are used

import re
import sys

def regexMatch(pattern, string):
	return bool(re.compile(pattern).match(string))

# returns whether the file path is for a male or female speaker
# 0: Male
# 1: Female
# 2: Silence
def getTruthValue(path):
	if 'OnlySilences' in path:
		return 2
	path = path[path.rfind("/")+1:]
	# regex match increases running time by 3%
	if regexMatch("C\d{3}_[MF]\d_(INDE|SENT)_\d{3}\.wav\.mfc", path):
		path = path[path.find("_")+1:]
		if path[0] == 'M':
			return 1
		elif path[0] == 'F':
			return 0
	elif regexMatch("p\d{3}_\d{3}\.wav\.mfc", path):
		comp = int(path[1:4])
		# Array based approach takes 300 sec / 5000 files (same)
		# outVect = [0,1,1,0,0,0,0,1,0,0,2,0,1,0,0,0,1,2,1,0,1,1,1,0,0,0,1,1,0,1,1,1,0,1,1,1,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,1,1,2,1,0,0,1,1,1,1,0,2,2,2,1,0,0,0,2,0,1,0,0,0,1,0,1,0,0,0,0,2,0,1,0,0,0,1,1,0,0,2,2,2,2,0,2,2,1,2,2,0,0,2,2,0,1,0,0,2,2,0,0,0,2,0,2,1,2,1,2,2,2,0,2,2,2,2,2,2,2,2,1,0,0,1,1,2,2,2,2,2,2,2,2,2,1,2,1]
		# Ctrl CV brute force 300 sec / 5000 files
		if comp == 225:
			return 0
		if comp == 226:
			return 1
		if comp == 227:
			return 1
		if comp == 228:
			return 0
		if comp == 229:
			return 0
		if comp == 230:
			return 0
		if comp == 231:
			return 0
		if comp == 232:
			return 1
		if comp == 233:
			return 0
		if comp == 234:
			return 0
		if comp == 236:
			return 0
		if comp == 237:
			return 1
		if comp == 238:
			return 0
		if comp == 239:
			return 0
		if comp == 240:
			return 0
		if comp == 241:
			return 1
		if comp == 243:
			return 1
		if comp == 244:
			return 0
		if comp == 245:
			return 1
		if comp == 246:
			return 1
		if comp == 247:
			return 1
		if comp == 248:
			return 0
		if comp == 249:
			return 0
		if comp == 250:
			return 0
		if comp == 251:
			return 1
		if comp == 252:
			return 1
		if comp == 253:
			return 0
		if comp == 254:
			return 1
		if comp == 255:
			return 1
		if comp == 256:
			return 1
		if comp == 257:
			return 0
		if comp == 258:
			return 1
		if comp == 259:
			return 1
		if comp == 260:
			return 1
		if comp == 261:
			return 0
		if comp == 262:
			return 0
		if comp == 263:
			return 1
		if comp == 264:
			return 0
		if comp == 265:
			return 0
		if comp == 266:
			return 0
		if comp == 267:
			return 0
		if comp == 268:
			return 0
		if comp == 269:
			return 0
		if comp == 270:
			return 1
		if comp == 271:
			return 1
		if comp == 272:
			return 1
		if comp == 273:
			return 1
		if comp == 274:
			return 1
		if comp == 275:
			return 1
		if comp == 276:
			return 0
		if comp == 277:
			return 0
		if comp == 278:
			return 1
		if comp == 279:
			return 1
		if comp == 281:
			return 1
		if comp == 282:
			return 0
		if comp == 283:
			return 0
		if comp == 284:
			return 1
		if comp == 285:
			return 1
		if comp == 286:
			return 1
		if comp == 287:
			return 1
		if comp == 288:
			return 0
		if comp == 292:
			return 1
		if comp == 293:
			return 0
		if comp == 294:
			return 0
		if comp == 295:
			return 0
		if comp == 297:
			return 0
		if comp == 298:
			return 1
		if comp == 299:
			return 0
		if comp == 300:
			return 0
		if comp == 301:
			return 0
		if comp == 302:
			return 1
		if comp == 303:
			return 0
		if comp == 304:
			return 1
		if comp == 305:
			return 0
		if comp == 306:
			return 0
		if comp == 307:
			return 0
		if comp == 308:
			return 0
		if comp == 310:
			return 0
		if comp == 311:
			return 1
		if comp == 312:
			return 0
		if comp == 313:
			return 0
		if comp == 314:
			return 0
		if comp == 315:
			return 1
		if comp == 316:
			return 1
		if comp == 317:
			return 0
		if comp == 318:
			return 0
		if comp == 323:
			return 0
		if comp == 326:
			return 1
		if comp == 329:
			return 0
		if comp == 330:
			return 0
		if comp == 333:
			return 0
		if comp == 334:
			return 1
		if comp == 335:
			return 0
		if comp == 336:
			return 0
		if comp == 339:
			return 0
		if comp == 340:
			return 0
		if comp == 341:
			return 0
		if comp == 343:
			return 0
		if comp == 345:
			return 1
		if comp == 347:
			return 1
		if comp == 351:
			return 0
		if comp == 360:
			return 1
		if comp == 361:
			return 0
		if comp == 362:
			return 0
		if comp == 363:
			return 1
		if comp == 364:
			return 1
		if comp == 374:
			return 1
		if comp == 376:
			return 1
	print "mfcPreprocessor.getTruthValue() failed with input: ", path
	sys.exit()
	return -1

# -1: Silences
def getSpeakerID(path):
	if 'OnlySilences' in path:
		return -1
	path = path[path.rfind("/")+1:]
	# regex match increases running time by 3%
	if regexMatch("C\d{3}_[MF]\d_(INDE|SENT)_\d{3}\.wav\.mfc", path):
		return int(path[1:4])
	elif regexMatch("p\d{3}_\d{3}\.wav\.mfc", path):
		return int(path[1:4])
	print "mfcPreprocessor.getSpeakerID() failed with input: ", path
	sys.exit()
	return -1
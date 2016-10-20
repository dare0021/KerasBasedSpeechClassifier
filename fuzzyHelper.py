# works with nb_classes <= 10 because lazy
# TODO: resolve pathological indecision
nb_classes = 0
ignoreThresh = 3

currentState = None;
lastSolidState = None;

solidSuffix = "solid"
fluidSuffix = "fluid"
timeSameInputs = 0
# kept just in case I figure out what to do with pathological indecision
timeSinceSolidPush = 0

# class will change if there were ignoreThreshold attempts to change to the class already
def init(num_classes, ignoreThreshold):
	global nb_classes, ignoreThresh
	nb_classes = num_classes
	ignoreThresh = ignoreThreshold

def push(state):
	global nb_classes, currentState, ignoreThresh, timeSinceSolidPush, timeSameInputs, lastSolidState
	assert nb_classes > 0
	assert state < nb_classes
	if currentState == None:
		setSolidState(state)
		return state

	if currentState.endswith(solidSuffix):
		if getCurrentState() == state:
			timeSameInputs += 1
			timeSinceSolidPush = 0
			return state
		else:
			timeSameInputs = 1
			if ignoreThresh < 1:
				setSolidState(state)
				return state
			else:
				currentState = str(state) + fluidSuffix
				timeSinceSolidPush += 1
				return lastSolidState
	elif currentState.endswith(fluidSuffix):
		if getCurrentState() == state:
			timeSameInputs += 1
			if timeSameInputs > ignoreThresh:
				setSolidState(state)
				return state
			else:
				timeSinceSolidPush += 1
				return lastSolidState
		else:
			timeSameInputs = 1
			currentState = str(state) + fluidSuffix
			timeSinceSolidPush += 1
			return lastSolidState
	else:
		assert "Invalid currentState" == ""

def setSolidState(state):
	global nb_classes, currentState, lastSolidState
	assert nb_classes > 0
	assert state < nb_classes
	currentState = str(state) + solidSuffix	
	lastSolidState = getCurrentState()
	timeSinceSolidPush = 0

def getCurrentState():
	return int(currentState[0])


# init(3, 2)
# print push(1)
# print push(2)
# print push(2)
# print push(0)
# print push(0)
# print push(0)
# print push(1)
# print push(0)
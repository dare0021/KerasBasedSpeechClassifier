# works with nb_classes <= 10 because lazy
# TODO: resolve pathological indecision
nb_classes = 0
ignoreThresh = 3

# kept just in case I figure out what to do with pathological indecision
currentState = None;
lastSolidState = None;

solidSuffix = "solid"
fluidSuffix = "fluid"
timeSinceSolidPush = 0
timeSameInputs = 0

# class will change if there were ignoreThreshold attempts to change to the class already
def init(num_classes, ignoreThreshold):
	global nb_classes, ignoreThresh
	nb_classes = num_classes
	ignoreThresh = ignoreThreshold

def push(state):
	global nb_classes, currentState, ignoreThresh, timeSinceSolidPush, timeSameInputs
	assert nb_classes > 0
	assert state < nb_classes
	if currentState == None:
		setSolidState(state)
		return state

	if currentState.endswith(solidSuffix):
		if currentState[0] == state:
			timeSameInputs += 1
			timeSinceSolidPush = 0
			return state
		else:
			timeSameInputs = 1
			if ignoreThresh < 1:
				setSolidState(state)
				return state
			else:
				currentState = state + fluidSuffix
				timeSinceSolidPush += 1
				return int(currentState[0])
	elif currentState.endswith(fluidSuffix):
		if currentState[0] == state:
			timeSameInputs += 1
			if timeSameInputs > ignoreThresh:
				setSolidState(state)
				return state
			else:
				timeSinceSolidPush += 1
				return int(currentState[0])
		else:
			timeSameInputs = 1
			currentState = state + fluidSuffix
			timeSinceSolidPush += 1
			return int(currentState[0])
	else:
		assert "Invalid currentState" == ""

def setSolidState(stat):
	global nb_classes, currentState
	assert nb_classes > 0
	assert state < nb_classes
	currentState = state + solidSuffix
	lastSolidState = currentState
	timeSinceSolidPush = 0

# TODO: test this class & integrate
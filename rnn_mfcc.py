# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D, MaxPooling2D
# from keras.utils import np_utils

import numpy as np
import unpackMFC as unmfc

input = "10m_processed.wav.mfc"

input = unmfc.run(input)

f = open("temp.txt", 'w')

for i in input:
	out = "[ "
	for j in i:
		out += str(j) + " "
	out += "]\n"
	print (out)
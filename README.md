# KerasBasedSpeechClassifier
Using Keras to build a classifier for speech. Not necessarily a speaker identifier

## Setup
Python 2.7

numpy is required

Wav files excluded from repo due to possible legal issues.

## How to use

Configure boot.py to use the branch and input that you want.

Or use rnn\_mfcc.py and rnn\_raw.py directly.

1) MFCC based branch

	i) Use CMU Sphinx 4 to generate MFCC feature vector files from your audio.

	ii) Configure rnn_mfcc.py to contain the RNN that you want (or use as provided)

	iii) Configure mfcPreprocessor.py to reflect your data set. By default, it uses the format *_[M,F]*\.mfc, where M and F are the two classes available.

	iv) Run via boot.py or rnn\_mfcc.py directly.

	modelPlayer.py can be used to play with saved models

2) Raw wav data approach

This branch is currently not working. (Results similar to random guessing.)

______________________________________________

Contains 2 branches, neither finished:

1) MFCC based branch

rnn_mfcc.py

Uses a MFCC feature set. MFCC being created via CMU Sphinx 4.

2) Raw wav data approach

rnn_raw.py

Uses data from uncompressed mono wav files. Because that's the corpus we have.

______________________________________________
License: MIT
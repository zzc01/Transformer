# Cleaning the German-English language sentence pair data set

The German-English data set is downloaded from http://www.manythings.org/anki/

This script does the following clean up steps:

1. Ignore all chars that cannot be represented in ASCII
2. Convert all chars to lowercase
3. Remove punctuations
4. Remove all non-pretable chars
5. Remove none alphabet words

This code is refereneced from https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/

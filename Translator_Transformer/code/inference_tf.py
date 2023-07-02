
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_text
from pickle import load, dump




"""# Load the model"""


translator = tf.saved_model.load('./Bootcamp/Tranformer_TF/deu-eng/metadata/translator_1')
result, _, _ = translator('Dieses Haus ist so groß, dass du mit deiner Familie darin leben kannst.')
print(result.numpy())


def print_translation(sentence, tokens, ground_truth):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')




''''''
filename = './Bootcamp/Tranformer_TF/deu-eng/data/deu-eng-test.pkl'

with open( filename, 'rb') as file:
    test_data = load(file)

test_data2 = np.array(test_data)
trainX = test_data2[:, 0]
trainY = test_data2[:, 1]

for i in range(len(trainX)):
    if i == 10: break
    sentence = trainX[i].decode('utf-8')
    ground_truth = trainY[i].decode('utf-8')
    translated_text, translated_tokens, attention_weights = translator(tf.constant(sentence))
    print('\n')
    print_translation(sentence, translated_text, ground_truth)





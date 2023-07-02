#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow_text
from pickle import load, dump
from time import time 
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import gc

# # Tokenizer
# Tokenizer is downloaded and saved in metadata folder so just load it. </br>
# Note the tokenizer is already included in the model. </br>
# Need the tokenzier for BLEU scoring </br>

# In[2]:


reloaded_tokenizers = tf.saved_model.load('./Bootcamp/Tranformer_TF/deu-eng/metadata/tokenizer_deu_eng_1')
string = "When writing a sentence, generally you start with a capital letter and finish with a period (.), an exclamation mark (!), or a question mark (?)."
tokens = reloaded_tokenizers.eng.tokenize([string])
round_trip = reloaded_tokenizers.eng.detokenize(tokens)
print(round_trip.numpy()[0].decode('utf-8'))


# # Data

# In[3]:


filename = './Bootcamp/Tranformer_TF/deu-eng/data/deu-eng-test.pkl'
with open(filename, 'rb') as file:
    test_data = load(file)
type(test_data)    


# In[4]:


test_data[0]


# In[5]:


test_data[0][0].decode('utf-8')


# # Load the model

# In[6]:


translator = tf.saved_model.load('./Bootcamp/Tranformer_TF/deu-eng/metadata/translator_1')


# In[7]:


translation, _, _ = translator('Wir bekommen ein neues Auto nächsten Monat.')
print(translation.numpy())


# In[8]:


idx = 10
raw_source, raw_target = test_data[idx][0].decode('utf-8'), test_data[idx][1].decode('utf-8')
translation, _, _ = translator(raw_source)
print(f"src={raw_source}")
print(f"target={raw_target}")
print(f"predict={translation.numpy().decode('utf-8')}")


# In[9]:


sentence = tf.constant(raw_source)
assert isinstance(sentence, tf.Tensor)
print(sentence.shape)
# sentence = sentence[tf.newaxis]
print(sentence.shape)
print(sentence.numpy())
#
translation, _, _ = translator(sentence)
print(f"src={sentence}")
print(f"target={raw_target}")
print(f"predict={translation.numpy().decode('utf-8')}")


# # Evaluate 

# In[10]:


target = "When writing a sentence, generally you start with a capital letter and finish with a period (.), an exclamation mark (!), or a question mark (?)."
target = reloaded_tokenizers.eng.tokenize([target])
target = reloaded_tokenizers.eng.detokenize(target)
target = target.numpy()[0].decode('utf-8')
# print(target)
actual = [[target.split()]]
print(actual)

predict = "When writing a sentence, generally you start with a capital letter and finish with a period (.), an exclamation mark (!), or a question mark (?)."
predict = reloaded_tokenizers.eng.tokenize([predict])
predict = reloaded_tokenizers.eng.detokenize(predict)
predict = predict.numpy()[0].decode('utf-8')
# print(predict)
predicted = [predict.split()]
print(predicted)

print('BLEU-1    %f' % corpus_bleu(actual, predicted, weights=(1.0, 0.0, 0.0, 0.0)))
print('BLEU-2    %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0.0, 0.0)))    
print('BLEU-3    %f' % corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0.0))) 
print('BLEU-4    %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))  


# In[11]:


print(len(test_data))
print(len(test_data)*0.07/60, 'min to predict all the test data')


# In[ ]:


actual, predicted = list(), list()
BLEU1, BLEU2, BLEU3, BLEU4, length = 0, 0, 0, 0, 0
f = open( "./Bootcamp/Tranformer_TF/deu-eng/metadata/eng_processed.txt", 'rt')
texts = f.read()
f.close()
texts = texts.strip().split('\n')

time0 = time()


for i, source in enumerate(test_data):
    raw_src = source[0].decode('utf-8')
    # raw_target = source[1].decode('utf-8')
    raw_target = texts[i]
    # if i == 100: break
    #
    translation, _, _ = translator(raw_src)
    translation = translation.numpy().decode('utf-8')
    if i < 3: 
        print(f"src = {raw_src}")
        print(f"target = {raw_target}")
        print(f"predict = {translation}")
        print("\n")
    #
    actual.append([raw_target.split()])
    predicted.append(translation.split())
    
    length += 1
    if length % 200 ==0:
        print(length, time()-time0)

    
# print(actual)
# print(predicted)       
 
print(f'Predict time = {time()-time0}')
# print('BLEU-1 %f' % (BLEU1/length))
# print('BLEU-2 %f' % (BLEU2/length))
# print('BLEU-3 %f' % (BLEU3/length))
# print('BLEU-4 %f' % (BLEU4/length))
# print('\n')
print('BLEU-1 %f' % corpus_bleu(actual, predicted, weights=(1.0, 0.0, 0.0, 0.0)))
print('BLEU-2 %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0.0, 0.0)))    
print('BLEU-3 %f' % corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0.0))) 
print('BLEU-4 %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))  
print(f'BLEU time = {time()-time0}')







''' Tokenize English Test Data'''
'''
for i, source in enumerate(test_data):
    if i < 11980: continue
    raw_target = source[1].decode('utf-8')
    # if i == 100: break
    #
    raw_target = reloaded_tokenizers.eng.tokenize([raw_target])
    raw_target = reloaded_tokenizers.eng.detokenize(raw_target)
    raw_target = raw_target.numpy()[0].decode('utf-8')
    f.write(raw_target + '\n')
    #
    length += 1
    if length % 200 ==0:
        del reloaded_tokenizers
        gc.collect()
        reloaded_tokenizers = tf.saved_model.load('./Bootcamp/Tranformer_TF/deu-eng/metadata/tokenizer_deu_eng_1')
        f.close()
        f = open( "./Bootcamp/Tranformer_TF/deu-eng/metadata/eng_processed.txt", 'a' )
        print(length, time()-time0)
'''






''' Extract Raw English Test Data'''
'''
for i, source in enumerate(test_data):
    # if i == 100: break
    raw_target = source[1].decode('utf-8')
    f.write(raw_target + '\n')
    length += 1
    if length % 200 ==0:
        f.close()
        f = open( "./Bootcamp/Tranformer_TF/deu-eng/metadata/eng_notprocessed.txt", 'a', encoding="utf-8" )
        print(length, time()-time0)
'''

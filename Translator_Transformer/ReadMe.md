# The Encoder-Decoder Transformer Machine Translator
<br/>

1. The data set used here is the [English-German sentence pair](http://www.manythings.org/anki/).

The data is cleaned and split into train, validation, and testing sets. The script used to clean the data is [Tokenizer_TF_DEU_ENG.ipynb](/code/Tokenizer_TF_DEU_ENG.ipynb) Here we applied canonical decomposition normalization to the sentence word and encoded the sentence to byte using utf-8. 


3. Here we use the Encoder-Decoder Model with Transformer neural network for the German to English translation task. 

<pre><p align="center">
<img src="https://github.com/zzc01/Transformer/assets/86133411/b05f651e-2c70-4d2c-a164-be27b1f89e3b"  width="300" >
</p></pre>


3. Tokenzie words with sub-word WordPiece tokenization.

Traditional Word-based tokenizer suffers from the very large vocabularies, out-of-vocailary tokens, and loss of meaning across similar words. Sub-word tokenizers solves this by decompoing rare words into smaller meaningful subwords. In [Tokenizer_TF_DEU_ENG.ipynb](/code/Tokenizer_TF_DEU_ENG.ipynb) we first create vocabulary with bert_vocab.bert_vocab_from_dataset(). Next to build the tokenizer we used tensorflow_text.BertTokenizer() to build a tokenizer for german word and another tokenizer for english word. Finaly the tokenizer is saved for later use. 

4. The encodr-decoder transformer model

The model contain different building parts, including the [transformer](/Translator_Transformer/code/transformer_tf.py), [attention](/Translator_Transformer/code/attention_tf.py), [encoder](/Translator_Transformer/code/encoder_tf.py), [decoder](/Translator_Transformer/code/decoder_tf.py), and [positonal encoding](/Translator_Transformer/code/positional_encoding_tf.py). 

6. Training the model 

In [train_tb_tf](/Translator_Transformer/code/train_tb_tf.py) we use an adam optimizer with custom learning rate scheduler. The masked loss is calculated using SparseCategoricalCrossentropy and padding mask.  

7. The model is evluated using BLEU score in [evaluate_tf](/Translator_Transformer/code/evaluate_tf.py). Below shows the BLEU score result. 

The left BLEU scores are the result of scoring the prediction against the raw target sentence. The right BLEU scores are scoring the prediction against the tokenized and then detokenized raw target sentence. The reason to do the tokenized and detokenized step is to seperate the puntuations and the words. For example to convert "don't" to "don ' t", and "the end." to "the end ."

<pre><p align="center">
<img src="https://github.com/zzc01/Transformer/assets/86133411/f1d7e1c5-d9a5-40ea-9b5e-6c8006b6e9c5"  width="400" >
</p></pre>

8. The code is referenced from Tensorflow Tutorials [1] and Minchine Learning Mastery [2]


# References 
[1] [Tensorflow Tutorials](https://www.tensorflow.org/text/tutorials) <br/>
[2] [Machine Learning Mastery](https://machinelearningmastery.com/) <br/> 
[3] [Hugging Face](https://huggingface.co/learn/nlp-course/chapter0/1?fw=pt) <br/>


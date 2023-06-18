# The Encoder-Decoder LSTM Machine Translator

1. The data set used here is the English-German sentence pair. The data is cleaned and store in the [Data_Cleaning](https://github.com/zzc01/Transformer/tree/main/Data_Cleaning) folder <br>
2. Here we use the Encoder-Decoder Model with LSTM neural network for the German to English translation task. [1]
<pre><p align="center">
<img src="https://github.com/zzc01/Transformer/assets/86133411/4d6885bc-495b-4536-b320-65b41085c26c" height="250">
</p></pre>

4. The sentences are first tokenized then sent into the following neural network shown below <br>
5. The model training part is in the notebook [translator_lstm.ipynb](translator_lstm.ipynb) <br>
6. The model is then evaluated using BLEU score in [evaluate.ipynb](evaluate.ipynb) <br>
7. The code references from https://machinelearningmastery.com/ 

<pre><p align="center">
<img src="https://github.com/zzc01/Transformer/assets/86133411/a7ea7f73-692e-4afa-a338-17c6b614bf8a" height="400">
</p></pre>
<br/>

# References
[1] [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf) <br>
[2] https://machinelearningmastery.com/
 

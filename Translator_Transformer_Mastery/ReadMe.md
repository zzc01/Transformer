# A Transformer Machine Translator with Custumized Attention

Here is an implmentation of the machine translator using the encoder-decoder architecture with transformers. The transfomer model is built in Tensorflow [1] and references from code from Machine Learning Mastery [2]. The translator is trained to translate German sentences to English sentences. <br/>


1. The data set used here is the English-German sentence pair from http://www.manythings.org/anki/. The data is cleaned and store in the [Data_Cleaning]() folder

2. Word-based tokenizer. Two word-based tokenizers are built using the Tokenizer from keras.preprocessing.text. One fore German sentences and the other for English sentences. 

3. Here we use the Encoder-Decoder Model with Transformer neural network for the German to English translation task. The transformer model contain different building parts, including the mulit-head attention block, encoder, decoder, and positional encoder. Here is a block diagram of the transformer [3]. 

<pre><p align="center">
<img src="https://github.com/zzc01/Transformer/assets/86133411/b05f651e-2c70-4d2c-a164-be27b1f89e3b"  width="300" >
</p></pre>

6. Training the model 

The training of the model is implemented in [Training.ipynb](/Translator_Transformer_Mastery/Training.ipynb). Here we use an adam optimizer with custom learning rate scheduler. The masked loss is calculated using SparseCategoricalCrossentropy and padding mask. Below is the training loss and validation loss v.s. epochs. 

<pre><p align="center">
<img src="https://github.com/zzc01/Transformer/assets/86133411/8e29e42f-f92c-4e80-b1ca-463fb80d7cf4"  width="400" >
</p></pre>


7. Inference

Below shows the translation result of a German sentence "Fass nichts an, ohne zu fragen!" to English sentence. </br>

**Input:         : Fass nichts an, ohne zu fragen!</br>**
**Prediction     : don ' t touch anything without asking .</br>**
**Ground truth   : Don't touch anything without asking.</br>**

8. The model is evluated using BLEU score in [Evaluate.ipynb](/Translator_Transformer_Mastery/Evaluate.ipynb). Below shows the BLEU score result. 

<pre><p align="center">
<img src="https://github.com/zzc01/Transformer/assets/86133411/4d29b61e-4753-498c-939b-694859b67b5c"  width="400" >
</p></pre>



# References 
[1] [Tensorflow](https://www.tensorflow.org/) <br/>
[2] [Machine Learning Mastery](https://machinelearningmastery.com/) <br/> 
[3] [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)<br/>


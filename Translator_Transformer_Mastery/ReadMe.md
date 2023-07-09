# A Machine Translator with Encoder-Decoder Transformers 

Here is an implmentation of the machine translator using the encoder-decoder architecture with transformers. The transfomer model is built in Tensorflow [1] and references from the tutorials from Machine Learning Mastery [2]. The translator is trained to translate German sentences to English sentences. <br/>


1. The data set used here is the English-German sentence pair from http://www.manythings.org/anki/. The data is cleaned and store using the script [Clean_pairs.ipynb](/Translator_Transformer_Mastery/Clean_pairs.ipynb) folder

2. Word-based tokenizer. Two word-based tokenizers are built using the Tokenizer from keras.preprocessing.text. One fore German sentences and the other for English sentences. 

3. Here we use the Encoder-Decoder Model with Transformer neural network for the German to English translation task. The transformer model contain different building parts, including the mulit-head attention block, encoder, decoder, and positional encoder. Here is a block diagram of the transformer [3]. 

<pre><p align="center">
<img src="https://github.com/zzc01/Transformer/assets/86133411/b05f651e-2c70-4d2c-a164-be27b1f89e3b"  width="300" >
</p></pre>

4. The scaled dot-product attention and multi-head attention [2][3].

<pre><p align="center">
<img src="https://github.com/zzc01/Transformer/assets/86133411/2662ac99-ee81-418f-af30-3d0eaf6e560e"  width="450" >
</p></pre>

A more detailed graphical explaination of multi-head attentions can be found below [4]. 

<pre><p align="center">
<img src="https://github.com/zzc01/Transformer/assets/86133411/17035d2b-dfd1-4b24-aa8e-24736183a9e5"  width="450" >
</p></pre>


5. Training the model 

The training of the model is implemented in [Training.ipynb](/Translator_Transformer_Mastery/Training.ipynb). Here we use an adam optimizer with custom learning rate scheduler. The masked loss is calculated using SparseCategoricalCrossentropy and padding mask. Below is the training loss and validation loss v.s. epochs from [Plot_Training_Results.ipynb](/Translator_Transformer_Mastery/Plot_Training_Results.ipynb). 

<pre><p align="center">
<img src="https://github.com/zzc01/Transformer/assets/86133411/8e29e42f-f92c-4e80-b1ca-463fb80d7cf4"  width="400" >
</p></pre>


6. Inference

Below shows the translation result of a German sentence "ich bin arzt" to English sentence. </br>

**Input:         : ich bin arzt</br>**
**Prediction     : im a medic</br>**
**Ground truth   : im a doctor</br>**

7. The model is evluated using BLEU score in [Evaluate.ipynb](/Translator_Transformer_Mastery/Evaluate.ipynb). Below shows the BLEU score result. 

<pre><p align="center">
<img src="https://github.com/zzc01/Transformer/assets/86133411/daba43e6-b7f1-4bc1-b747-c9b35a756097"  width="200" >
</p></pre>

8. Here lists some possible future improvements.
* Change word embedding to trainable parameters
* There are some word attending to 0 pad.
* Futher research of the multi-head attention module
* Futher research of the padding mask and causal mask 



# References 
[1] [Tensorflow](https://www.tensorflow.org/) <br/>
[2] [Machine Learning Mastery](https://machinelearningmastery.com/) <br/> 
[3] [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)<br/>
[4] [Transformers-explained-visually](https://towardsdatascience.com/transformers-explained-visually-not-just-how-but-why-they-work-so-well-d840bd61a9d3)<br/>



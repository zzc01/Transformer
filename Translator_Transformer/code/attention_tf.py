
import tensorflow as tf 






"""# Attention

RNNs and CNNs have their limitations.

* The RNN allows information to flow all the way across the sequence, but it passes through many processing steps to get there (limiting gradient flow). These RNN steps have to be run sequentially and so the RNN is less able to take advantage of modern parallel devices.
* In the CNN each location can be processed in parallel, but it only provides a limited receptive field. The receptive field only grows linearly with the number of CNN layers, You need to stack a number of Convolution layers to transmit information across the sequence (Wavenet reduces this problem by using dilated convolutions).
"""

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()






"""## Cross Attention"""

class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
                                        query=x,
                                        key=context,
                                        value=context,
                                        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x

"""The output length of CrossAttention is the length of the query sequence, and not the length of the context key/value sequence."""

# sample_ca = CrossAttention(num_heads=2, key_dim=512)

# print(pt_emb.shape)
# print(en_emb.shape)
# print(sample_ca(en_emb, pt_emb).shape)






"""## Global Self Attention"""

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
                          query=x,
                          value=x,
                          key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

# sample_gsa = GlobalSelfAttention(num_heads=2, key_dim=512)

# print(pt_emb.shape)
# print(sample_gsa(pt_emb).shape)






"""# Causal Attention

Like the text generation tutorial, and the NMT with attention tutorial, Transformers are an "autoregressive" model: They generate the text one token at a time and feed that output back to the input. To make this efficient, these models ensure that the output for each sequence element only depends on the previous sequence elements; the models are "causal".
"""

class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
                          query=x,
                          value=x,
                          key=x,
                          use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

# sample_csa = CausalSelfAttention(num_heads=2, key_dim=512)

# print(en_emb.shape)
# print(sample_csa(en_emb).shape)

# """This seems to be an important check. Can do this check in the Mastery version"""

# out1 = sample_csa(embed_en(en[:, :3]))
# out2 = sample_csa(embed_en(en))[:, :3]

# print(out1.shape)
# tf.reduce_max(abs(out1 - out2)).numpy()

# out1 = sample_gsa(embed_en(en[:, :3]))
# out2 = sample_gsa(embed_en(en))[:, :3]

# tf.reduce_max(abs(out1 - out2)).numpy()

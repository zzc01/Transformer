from tensorflow import matmul, math, cast, float32, reshape, shape, transpose
from tensorflow.keras.layers import Layer, Dense
from keras.backend import softmax
from numpy import random 

class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
    def call(self, queries, keys, values, d_k, mask=None):
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32)) # tf.float64 v.s. float32 
        # print(f"scores.shape={scores.shape}")
        # print(f"mask.shape={mask.shape}")
        if mask is not None:
            scores = scores - 1e9*mask
        weights = softmax(scores)
        #print(f"weights.shape={weights.shape}")
        return matmul(weights, values)

class MultiHeadAttention(Layer):
    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = DotProductAttention()
        self.heads = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.W_q = Dense(d_k)
        self.W_k = Dense(d_k)
        self.W_v = Dense(d_v)
        self.W_o = Dense(d_model)
    def reshape_tensor(self, x, heads, flag):
        if flag:
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
        return x
    def call(self, queries, keys, values, mask=None):
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        #print(f"queries.shape={queries.shape}")        
        #print(f"self.W_q(queries).shape={self.W_q(queries).shape}")        
        #print(f"q_reshaped.shape={q_reshaped.shape}")
        #print(f"k_reshaped.shape={k_reshaped.shape}")
        #print(f"v_reshaped.shape={v_reshaped.shape}")
        #
        o_ = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        #print(f"o_.shape={o_.shape}")
        output = self.reshape_tensor(o_, self.heads, False)
        return self.W_o(output)  

from tensorflow.keras.layers import Layer, Dense, ReLU, Dropout, LayerNormalization, Input  
from multihead_attention import MultiHeadAttention
from positional_encoding import PositionEmbeddingFixedWeights
from tensorflow.keras import Model

class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = LayerNormalization()
    def call(self, x, sublayer_x):
        add = x + sublayer_x
        return self.layer_norm(add) 

class FeedForward(Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.fully_connected1 = Dense(d_ff)
        self.fully_connected2 = Dense(d_model)
        self.activation = ReLU()
    def call(self, x):
        x_fc1 = self.fully_connected1(x)
        x_act = self.activation(x_fc1)
        return self.fully_connected2(x_act)


class EncoderLayer(Layer):
    def __init__(self, sequence_length, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.build(input_shape=[None, sequence_length, d_model])
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()

    def build_graph(self):
        input_layer = Input(shape=(self.sequence_length, self.d_model))
        return Model(inputs=[input_layer], outputs=self.call(input_layer, None, True))

    def call(self, x, padding_mask, training):
        #print('***** EncoderLayer MultiHeadAttention *****')
        multihead_output = self.multihead_attention(x, x, x, padding_mask)
        #print(f"multihead_output.shape={multihead_output.shape}")
        #
        multihead_output = self.dropout1(multihead_output, training = training)
        #print(f"multihead_output.shape={multihead_output.shape}")
        #
        addnorm_output = self.add_norm1(x, multihead_output)
        #print(f"addnorm_output.shape={addnorm_output.shape}")
        #
        feedforward_output = self.feed_forward(addnorm_output)
        #print(f"feedforward_output.shape={feedforward_output.shape}")
        #
        feedforward_output = self.dropout2(feedforward_output, training = training)
        #print(f"feedforward_output.shape={feedforward_output.shape}")
        #
        return self.add_norm2(addnorm_output, feedforward_output)                

class Encoder(Layer):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
        self.dropout = Dropout(rate)
        self.encoder_layer = [EncoderLayer(sequence_length, h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]
        
    def call(self, input_sentence, padding_mask, training):
        # print(f"input_sentence.shape={input_sentence.shape}")
        # print(f"input_sentence={input_sentence}")
        pos_encoding_output = self.pos_encoding(input_sentence)
        #print(f"pos_encoding_output.shape={pos_encoding_output.shape}")
        #
        x = self.dropout(pos_encoding_output, training=training)
        #print(f"x.shape={x.shape}")
        #
        for i, layer in enumerate(self.encoder_layer):
            x = layer(x, padding_mask, training)
        # 
        return x        
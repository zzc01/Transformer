from encoder import Encoder 
from decoder import Decoder 
from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis, transpose
from tensorflow.keras import Model 
from tensorflow.keras.layers import Dense 
from keras.backend import softmax


class TransformerModel(Model):
	def __init__(self, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate, **kwargs):
		super(TransformerModel, self).__init__(**kwargs)
		self.encoder = Encoder(enc_vocab_size, enc_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate)
		self.decoder = Decoder(dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate)
		self.model_last_layer = Dense(dec_vocab_size) # where is the soft_max?

	def padding_mask(self, input): 
		# print(f"input.shape={input.shape}")
		# print(input)
		mask = math.equal(input, 0) 
		# print(f"mask.shape={mask.shape}")
		mask = cast(mask, float32) 
		# print(f"mask.shape={mask.shape}")
		

		#
		# mask = mask[:, newaxis, newaxis, :]
		# # print(f"mask.shape={mask.shape}")
		# new_mask = maximum( mask, transpose(mask, perm=(0, 1, 3, 2)) ) 
		# print(f"new_mask.shape={new_mask.shape}")
		# return new_mask
		#

		return mask[:, newaxis, newaxis, :] 

	def lookahead_mask(self, shape): 
		mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0) 
		return mask 

	def call(self, encoder_input, decoder_input, training): 
		enc_padding_mask = self.padding_mask(encoder_input) 
		# 
		encoder_output = self.encoder(encoder_input, enc_padding_mask, training) 
		#
		dec_in_padding_mask   = self.padding_mask(decoder_input) 
		dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1]) 
		dec_in_lookahead_mask = maximum(dec_in_padding_mask, dec_in_lookahead_mask) 
		#
		decoder_output = self.decoder(decoder_input, encoder_output, dec_in_lookahead_mask, enc_padding_mask, training) 
		#
		model_output   = self.model_last_layer(decoder_output) 
		# where is the soft_max?
		# return model_output
		return softmax(model_output)


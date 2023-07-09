from pickle import load, dump, HIGHEST_PROTOCOL
from numpy.random import shuffle
from numpy import savetxt
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64 

class PrepareDataset:
	def __init__(self, **kwargs):
		super(PrepareDataset, self).__init__(**kwargs)
		self.n_sentences = 10000
		self.train_split = 0.8
		self.val_split = 0.1

	def create_tokenizer(self, dataset):
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(dataset)
		return tokenizer
		
	def find_seq_length(self, dataset):
		return max(len(seq.split()) for seq in dataset)

	def find_vocab_size(self, tokenizer, dataset):
		tokenizer.fit_on_texts(dataset)
		return len(tokenizer.word_index) + 1

	def encode_pad(self, dataset, tokenizer, seq_length):
		X = tokenizer.texts_to_sequences(dataset)
		X = pad_sequences(X, maxlen=seq_length, padding='post')
		X = convert_to_tensor(X, dtype=int64)
		return X

	def save_tokenizer(self, tokenizer, name):
		with open(name + '_tokenizer.pkl', 'wb') as handle:
			dump(tokenizer, handle, protocol=HIGHEST_PROTOCOL)


	def __call__(self, filename, **kwargs):
		# Train
		filename = ""
		filename = "./data/german-english-train.pkl"
		dataset = load(open(filename, 'rb'))
		for i in range(dataset[:, 0].size):
			dataset[i, 0] = '<START>' + ' ' + dataset[i, 0] + ' ' + '<EOS>'
			dataset[i, 1] = '<START>' + ' ' + dataset[i, 1] + ' ' + '<EOS>'
		train = dataset
		# Validate
		filename = "./data/german-english-val.pkl"
		dataset = load(open(filename, 'rb'))
		for i in range(dataset[:, 0].size):
			dataset[i, 0] = '<START>' + ' ' + dataset[i, 0] + ' ' + '<EOS>'
			dataset[i, 1] = '<START>' + ' ' + dataset[i, 1] + ' ' + '<EOS>'
		val = dataset
		#
		enc_tokenizer = self.create_tokenizer(train[:, 0])
		enc_seq_length = self.find_seq_length(train[:, 0])
		enc_vocab_size = self.find_vocab_size(enc_tokenizer, train[:, 0])
		# enc_vocab_size = len(enc_tokenizer.word_index) + 1
		#
		trainX = self.encode_pad(train[:,0], enc_tokenizer, enc_seq_length)

		#
		dec_tokenizer = self.create_tokenizer(train[:, 1])
		dec_seq_length = self.find_seq_length(train[:, 1])
		dec_vocab_size = self.find_vocab_size(dec_tokenizer, train[:, 1])
		# dec_vocab_size = len(dec_tokenizer.word_index) + 1
		# 
		trainY = self.encode_pad(train[:,1], dec_tokenizer, dec_seq_length)

		# 
		valX = self.encode_pad(val[:,0], enc_tokenizer, enc_seq_length)
		valY = self.encode_pad(val[:,1], dec_tokenizer, dec_seq_length)

		#
		path = './metadata/'
		self.save_tokenizer(enc_tokenizer, path + 'enc')
		self.save_tokenizer(dec_tokenizer, path + 'dec')
		# self.save_tokenizer(enc_tokenizer, 'enc')
		# self.save_tokenizer(dec_tokenizer, 'dec')
		# savetxt('test_dataset.txt', test, fmt='%s')

		return trainX, trainY, valX, valY, train, val, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size

if __name__ == '__main__':
	dataset = PrepareDataset()
	trainX, trainY, valX, valY, train_org, val_org, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = dataset('english-german-both.pkl')	

	print(trainX.shape)
	print(valX.shape)

	print(train_org[0, 0], '\n', trainX[0, :])
	print('Encoder sequence length: ', enc_seq_length)


	print(train_org[0, 1], '\n', trainY[0, :])
	print('Encoder sequence length: ', dec_seq_length)

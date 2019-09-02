import argparse
import time
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import dataLoader as loader
import seq2seq
import train
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument("--encoder", required = True, help = "Encoder file path to load trained_encoder\'s learned parameters.")
	parser.add_argument("--decoder", required = True, help = "Decoder file path to load trained_decoder\'s learned parameters.")
	parser.add_argument("--embedding_size", type = int, default = 300, help = "Word embedding vector dimension size. default = 300")
	parser.add_argument("--hidden_size", type = int, default = 300, help = "Hidden size of rnn. default = 300")
	parser.add_argument("--srcLang", type = str, default = "eng", required = False, help = "The language of source.")
	parser.add_argument("--tgtLang", type = str, default = "fra", required = False, help = "The language of target.")
	config = parser.parse_args()
	return config

def translate(pair, output):
	print("Source:\t{}\nAnswer:\t{}".format(pair[0], pair[1]))
	print("Translate: {}".format(output), end="\n\n")

def evaluate(sentence, encoder, decoder, max_length = loader.MAX_LENGTH):
	with torch.no_grad():
		input_tensor = train.tensorFromSentence(input_lang, sentence)
		input_length = input_tensor.size(0)

		encoder_hidden = (encoder.initHidden().to(device), encoder.initHidden().to(device))
		encoder_outputs = torch.zeros(max_length, encoder.hidden_size).to(device)

		for index in range(input_length):
			encoder_output, encoder_hidden = encoder(input_tensor[index], encoder_hidden)
			encoder_outputs[index] = encoder_output[0, 0]

		decoder_input = torch.tensor([[loader.SOS_token]]).to(device)
		decoder_hidden = encoder_hidden

		decoded_words = []
		for index in range(max_length):
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
			topv, topi = decoder_output.data.topk(1)
			if topi.item() == loader.EOS_token:
				break
			else:
				decoded_words.append(output_lang.index2word[topi.item()])
			decoder_input = topi.squeeze().detach()
	return decoded_words

def evaluateiters(pairs, encoder, decoder, train_pairs_seed = 0):
	start = time.time()
	cc = SmoothingFunction()
	train_pairs, test_pairs = train_test_split(pairs, test_size = 0.15, random_state = train_pairs_seed)

	scores = []
	for index, pair in enumerate(test_pairs):
		output_words = evaluate(pair[0], encoder, decoder)
		print("source sentence@@ : {}".format(pair[0]))
		output_sentence = " ".join(output_words)
		print("translate sentence@@ : {}".format(output_sentence))

		translate(pair, output_sentence)

		ref = pair[1].split()
		hyp = output_words
		scores.append(sentence_bleu([ref], hyp, smoothing_function = cc.method3) * 100.)
	print("BLEU: {:.4}".format(sum(scores)/len(test_pairs)))

if __name__ == "__main__":
	"""
	Evaluation is mostly the same as training,
	but there are no targets so we simply feed the decoder's predictions back to itself for each step.
	Every time it predict a word, we add it to the output string
	and if it predicts the EOS token we step there.
	"""
	config = argparser()

	input_lang, output_lang, pairs = loader.prepareData(config.srcLang, config.tgtLang, True)

	encoder = seq2seq.LSTMEncoder(input_size = input_lang.n_words, embedding_size \
	= config.embedding_size, hidden_size = config.hidden_size).to(device)
	decoder = seq2seq.LSTMDecoder(output_size = output_lang.n_words, embedding_size \
	= config.embedding_size, hidden_size = config.hidden_size).to(device)

	encoder.load_state_dict(torch.load(config.encoder))
	encoder.eval()
	decoder.load_state_dict(torch.load(config.decoder))
	decoder.eval()

	evaluateiters(pairs, encoder, decoder)

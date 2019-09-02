import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import argparse
import time
import math
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim
import dataLoader as loader
import seq2seq
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument("--epoch_num", required = True, type = int, help = "Number of epoch to train.")
	parser.add_argument("--embedding_size", type = int, default = 300, help = "Word Embedding Vector dimension size, default = 300")
	parser.add_argument("--hidden_size", type = int, default = 300, help = "Hidden size of RNN. default = 300")
	parser.add_argument("--model_path", type = str, default = "model", help = "The path of encoder and decoder models.")
	parser.add_argument("--srcLang", type = str, default = "eng", required = False, help = "The language of source.")
	parser.add_argument("--tgtLang", type = str, default = "fra", required = False, help = "The language of target.")
	config = parser.parse_args()
	return config

def Timesformat(duration):
	minute = math.floor(duration / 60)
	hours = math.floor(minute / 60)
	duration -= minute * 60
	minute -= hours * 60
	if hours > 0 and minute > 0: return "{}hours {}minutes {}s.".format(hours, minute, duration)
	elif hours == 0 and minute > 0: return "{}minutes {}s.".format(minute, duration)
	else: return "{}s.".format(duration)

def indexesFromSentence(lang, sentence):
	return [lang.word2index[word] for word in sentence.split(" ")]

def tensorFromSentence(lang, sentence):
	indexes = indexesFromSentence(lang, sentence)
	indexes.append(loader.EOS_token)
	return torch.tensor(indexes, dtype=torch.long, device = device).view(-1, 1)

def tensorsFromPair(index, sentence_number, pair):
	input_tensor = tensorFromSentence(input_lang, pair[0])
	target_tensor = tensorFromSentence(output_lang, pair[1])
	print("\rsentence to tensor: {} / {}".format(index, sentence_number), end="")
	return (input_tensor, target_tensor)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length = loader.MAX_LENGTH):
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length, target_length = input_tensor.size(0), target_tensor.size(0)
	# |input_length|, |target_length| = (sentence_length)

	encoder_hidden = (encoder.initHidden().to(device), encoder.initHidden().to(device))
	# |encoder_hidden| = (2, num_layers * num_directions, batch_size, hidden_size)
	encoder_outputs = torch.zeros(max_length, encoder.hidden_size).to(device)
	# |encoder_outputs| = (max_length, hidden_size)

	loss = 0
	for word_index in range(input_length):
		encoder_output, encoder_hidden = encoder(input_tensor[word_index], encoder_hidden)
		# |encoder_output| = (batch_size, sequence_length, num_directions * hidden_size)
		# |encoder_hidden| = (2, num_layers * num_directions, batch_size, hidden_size)
		# 2: respectively, hidden state and cell state.
		encoder_outputs[word_index] = encoder_output[0, 0]

	decoder_input = torch.tensor([[loader.SOS_token]]).to(device)
	# |decoder_input| = (1, 1)
	decoder_hidden = encoder_hidden
	# |decoder_hidden| = (2, num_layers * num_directions, batch_size, hidden_size)
	# 2: repectively, hidden state and cell state

	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
	if use_teacher_forcing:
		# Teacher forcing: feed the target as the next input.
		for word_index in range(target_length):
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
			# |decoder_output| = (sequence_length, output_lang.n_words)
			# |decoder_hidden| = (2, num_layers * num_directions, batch_size, hidden_size)
			# 2. respectively, hidden state and cell state.

			loss += criterion(decoder_output, target_tensor[word_index])
			decoder_input = target_tensor[word_index] # teacher forcing
			# |decoder_input|, |target_tensor[word_index]| = (1)
	else:
		# Without teacher forcing: use its own predictions as the next input
		for word_index in range(target_length):
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
			# |decoder_output| = (sequence_length, output_lang.n_words)
			# |decoder_hidden| = (2, num_layers * num_directions, batch_size, hidden_size)
			# 2: respectively, hidden state and cell state.

			topv, topi = decoder_output.topk(1)	# top-1 value, index
			# |topv|, |topi| = (1, 1)

			decoder_input = topi.squeeze().detach()	# detach from history as input
			loss += criterion(decoder_output, target_tensor[word_index])
			# |target_tensor[word_index]| = (1)
			if decoder_input.item() == loader.EOS_token:
				# |decoder_input| = (1)
				break
	loss.backward()
	encoder_optimizer.step()
	decoder_optimizer.step()
	return loss.item()/target_length

def trainiters(pairs, encoder, decoder, epoch_number, model_path, train_pairs_seed = 0, print_every = 1000, plot_every = 1000, learning_rate = .01):
	start = time.time()
	plot_losses = []
	print_loss_total, plot_loss_total = 0, 0

	encoder_model_path = os.path.join(model_path, "encoder.pth")
	decoder_model_path = os.path.join(model_path, "decoder.pth")
	print("encoder model path: {} decoder model path: {}".format(encoder_model_path, decoder_model_path))

	train_pairs, test_pairs = train_test_split(pairs, test_size = 0.15, random_state = train_pairs_seed)
	single_epoch_size = len(train_pairs)
	train_pairs *= epoch_number
	#train_pairs += [random.choice(train_pairs) for i in range(step_target % len(train_pairs))]
	print("start sentence to tensor....")
	sentence_number = len(train_pairs)
	train_pairs = [tensorsFromPair(index, sentence_number, pair) for index, pair in enumerate(train_pairs)]

	encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)
	criterion = nn.NLLLoss()

	print("now start model training...")
	now_epoch_number = 1
	for iter in range(sentence_number):
		pair = train_pairs[iter - 1]
		# |pair| = (2)
		input_tensor, target_tensor = pair[0], pair[1]
		loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
		print_loss_total += loss
		plot_loss_total += loss

		if iter >= single_epoch_size: now_epoch_number += 1

		if iter % print_every == 0:
			print_loss_avg = print_loss_total / print_every
			#plot_losses.append(plot_loss_avg)
			plot_loss_total = 0
			print_loss_total = 0
			print("training step: {}/{} epoch_number: {}/{} average loss: {}".\
			format(iter+1, sentence_number, now_epoch_number, epoch_number, print_loss_avg))

	torch.save(encoder.state_dict(), encoder_model_path)
	torch.save(decoder.state_dict(), decoder_model_path)

if __name__ == "__main__":
	config = argparser()
	teacher_forcing_ratio = 0.5

	input_lang, output_lang, pairs = loader.prepareData(config.srcLang, config.tgtLang, True)

	print("dataset init finish! now start init encoder and decoder!")

	encoder = seq2seq.LSTMEncoder(input_size = input_lang.n_words, embedding_size = config.embedding_size, hidden_size \
	= config.hidden_size).to(device)

	decoder= seq2seq.LSTMDecoder(output_size = output_lang.n_words, embedding_size = config.embedding_size, hidden_size\
	= config.hidden_size).to(device)

	print("encoder and decoder models init finish! now start training!")

	config.model_path = os.path.join(os.getcwd() + "/" + config.model_path)
	print("model output path is: {}".format(config.model_path))

	if not os.path.exists(config.model_path): os.mkdir(config.model_path)
	if not os.path.isdir(config.model_path): raise ValueError("the model path is not dir.")

	trainiters(pairs, encoder, decoder, config.epoch_num, config.model_path)

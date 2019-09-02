import unicodedata
import re
import random

SOS_token, EOS_token = 0, 1
MAX_LENGTH = 15

class Lang(object):
	def __init__(self, class_name):
		self.class_name = class_name
		self.word2index = {'SOS': 0, 'EOS': 1}
		self.word2count = {}
		self.index2word = {0: 'SOS', 1: 'EOS'}
		self.n_words = 2 # SOS, EOS

	def addSentence(self, sentence):
		for word in sentence.split(" "):
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1

def unicodeToAscii(sentence):
	return "".join(char for char in unicodedata.normalize("NFD", sentence) if unicodedata.category(char) != "Mn")

def normalizeString(sentence):
	sentence = unicodeToAscii(sentence.lower().strip())
	sentence = re.sub(r"([.!?])", r"\1", sentence)
	sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)
	return sentence

def readLangs(language_1, language_2, reverse=False):
	print("Reading lines...")
	filein = open("../dataset/{}-{}.txt".format(language_1, language_2), "r", encoding="utf-8").read().strip().split("\n")

	# split every line into pairs and normalices
	pairs = [[normalizeString(sentence) for sentence in line.split("\t")] for line in filein]
	
	if not reverse:
		input_language = Lang(language_1)
		output_language = Lang(language_2)
	else:
		pairs = [list(reversed(pair)) for pair in pairs]
		input_language = Lang(language_2)
		output_language = Lang(language_1)

	return input_language, output_language, pairs

def filterPair(pair):
	return len(pair[0].split(" ")) < MAX_LENGTH and len(pair[1].split(" ")) < MAX_LENGTH

def filterPairs(pairs):
	return [pair for pair in pairs if filterPair(pair)]

def prepareData(language_1, language_2, reverse = False):
	input_lang, output_lang, pairs = readLangs(language_1, language_2, reverse)
	print("Read {} sentence pairs".format(len(pairs)))

	pairs = filterPairs(pairs)
	print("Trimmed to {} sentence pairs".format(len(pairs)))

	for pair in pairs:
		input_lang.addSentence(pair[0])
		output_lang.addSentence(pair[1])
	print("Counted words: {}- {}\t{} - {}".format(input_lang.class_name, input_lang.n_words, output_lang.class_name, output_lang.n_words))

	return input_lang, output_lang, pairs

if __name__ == "__main__":
	"""
	The full process for prepareing the data is:
		1. Read text file and split into lines, split lines into pairs
		2. Normalize text, filter by length and content
		3. Make word lists from sentences in pairs
	"""
	input_lang, output_lang, pairs = prepareData("eng", "fra", True)
	print(random.choice(pairs))

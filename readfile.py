import numpy as np
from tensorflow import keras as k


MAX_NB_WORDS = 20000 # Max number of words for the tokenizer
EMBEDDING_DIM = 100 # Dimensions of the word vectors
MAX_SEQ_LENGTH  = 280 # Max length of the text sequences (for padding purposes)

def read_twitter_handles():
  politicians = np.genfromtxt('senate.csv', delimiter=',', dtype='str', usecols=(0,2))
  return politicians

def sort_politicians(politicians):
  """

  :param politicians: list of the politicians and party retrieved from
      read_twitter_handles()
  :return: returns two lists: dems and repubs (Independents disregarded)
  """
  dems = []
  repubs = []
  for politician in politicians:
    if politician[1] == 'R':
      repubs.append(politician[0])
    elif politician[1] == 'D':
      dems.append(politician[0])
  return dems, repubs

def load_embeddings():
  """
  Loads the word embedding from file
  :return: returns a dictionary of the word vectors
  """
  f = open('glove.6B.%sd.txt' % EMBEDDING_DIM)
  embed_index = {}

  for line in f:
    vals = line.split()
    word = vals[0]
    vector = np.asarray(vals[1:], dtype='float32')
    embed_index[word] = vector

  f.close()
  return embed_index

def load_data():
  """
  loads the tweets stored in the files demTweets and repubTweets and creates labels
  Dems are 0 and Repubs are 1
  :return: returns data and labels for the data
  """
  demTweets = np.load('demTweets.npy')
  repubTweets = np.load('repubTweets.npy')

  tweets = [] # list the text of the tweets
  labels = [] # list of the labels (0 or 1)
  maxlen = 0
  for text in demTweets:
    tweets.append(text)
    labels.append(0)
    if len(text) > maxlen:
      maxlen = len(text)
  for text in repubTweets:
    tweets.append(text)
    labels.append(1)
    if len(text) > maxlen:
      maxlen = len(text)

  print('Tweets found: %s' % len(tweets))
  print('Longest tweet: %s' % maxlen)
  return tweets, labels

def embed_and_token(tweets, labels):
  """
  creates the embedding layer, tokenizes the corpus.
  :param tweets:corpus of the tweets
  :param labels:labels for the tweets
  :return: formatted data, labels, and the embedding layer
  """
  tokenizer = k.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS)
  tokenizer.fit_on_texts(tweets)
  data = tokenizer.texts_to_sequences(tweets)
  word_index = tokenizer.word_index

  data = k.preprocessing.sequence.pad_sequences(data)
  labels = np.asarray(labels)
  # labels = k.utils.to_categorical(np.asarray(labels))

  embedding_index = load_embeddings()

  embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
  for word, i in word_index.items():
    embedding_vec = embedding_index.get(word)
    if embedding_vec is not None:
      embedding_matrix[i] = embedding_vec
  embedding_layer = k.layers.Embedding(len(word_index) + 1,
                                       EMBEDDING_DIM,
                                       weights=[embedding_matrix],
                                       input_length=data.shape[1],
                                       trainable=False)
  return data, labels, embedding_layer
# a = read_twitter_handles()
# b,c = sort_politicians(a)
# print(len(b))
# print(len(c))
# load_embeddings()
# load_data()

from tensorflow import keras
import pandas as pd
import numpy as np
import io
import json
import pickle
# we will tune this params later
MAX_LEN = 14
TRAIN_SPLIT = 0.7
NUM_WORDS = 300
EMBED_SIZE = 512
LIMIT = 8900

# encoding of the texts into something suitable for our use case
# loading
h = open('tokenizer.pickle', 'rb')
tokenizer = pickle.load(h)

def encode_words_one_hot(words):
    seq = tokenizer.texts_to_sequences(words)
    seq_pad = keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post', value=0.0)
    return seq_pad


model = keras.models.load_model('sentient.h5')

while True:
    t = input("> ")
    print(model.predict(encode_words_one_hot(t)))

from tensorflow import keras
import pandas as pd
import numpy as np
import io
import json
import pickle

reviews_data = pd.read_csv('Reviews.csv')

# we will tune this params later
MAX_LEN = 200
TRAIN_SPLIT = 0.7
NUM_WORDS = 3000
EMBED_SIZE = 56
LIMIT = 90000

# basic queries on the dataset
big_text = reviews_data['Text'].head(LIMIT).to_numpy()
labels = reviews_data['Score'].head(LIMIT).to_numpy()

# encoding of the texts into something suitable for our use case
tokenizer = keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(big_text)

def encode_words_one_hot(words):
    print(words)
    seq = tokenizer.texts_to_sequences(words)
    seq_pad = keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post', value=0.0)
    print(seq_pad)
    return seq_pad

def encode_label(label):
    if label > 4:
        return 1.0
    else:
        return 0.0

# process and split the dataset
texts = encode_words_one_hot(big_text)
labels = np.vectorize(encode_label)(labels)

x_train = texts[:round(LIMIT*TRAIN_SPLIT)]
y_train = labels[:round(LIMIT*TRAIN_SPLIT)]
print(x_train.shape) 
print(y_train.shape)
x_eval = texts[round(LIMIT*TRAIN_SPLIT):]
y_eval = labels[round(LIMIT*TRAIN_SPLIT):]

# Optimizer
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Create the model
model = keras.Sequential()
model.add(keras.layers.Embedding(NUM_WORDS, EMBED_SIZE, input_length=MAX_LEN))
model.add(keras.layers.LSTM(30)) 
# model.add(keras.layers.Dense(20))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# For a multi-class classification problem
model.compile(optimizer='RMSprop',
              loss='binary_crossentropy',
              metrics=["accuracy"]
              )

model.fit(x_train, y_train, epochs=10, batch_size=1690)

# Store our model and our tokenizer info
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

model.save('sentient.h5')

print(model.evaluate(x_eval, y_eval))

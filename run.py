from tensorflow import keras
import pandas as pd
import numpy as np
import io
import json

reviews_data = pd.read_csv('Reviews.csv')

# we will tune this params later
MAX_LEN = 140
TRAIN_SPLIT = 0.7
NUM_WORDS = 3000
EMBED_SIZE = 512
LIMIT = 5900

# basic queries on the dataset
big_text = reviews_data['Text'].head(LIMIT).to_numpy()
labels = reviews_data['Score'].head(LIMIT).to_numpy()

# encoding of the texts into something suitable for our use case
tokenizer = keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(big_text)

def encode_words_one_hot(words):
    seq = tokenizer.texts_to_sequences(words)
    seq_pad = keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post', value=0.0)
    return seq_pad

def encode_label(label):
    if label > 3:
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
sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.3, nesterov=True)

# Create the model
model = keras.Sequential()
model.add(keras.layers.Embedding(NUM_WORDS, EMBED_SIZE, input_length=MAX_LEN))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(900))
model.add(keras.layers.Dense(600))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# For a multi-class classification problem
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=["accuracy"]
              )

model.fit(x_train, y_train, epochs=15, batch_size=32)

# Store our model and our tokenizer info
tokenizer_json = tokenizer.to_json()
with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

model.save('sentient.h5')

print(model.evaluate(x_eval, y_eval))

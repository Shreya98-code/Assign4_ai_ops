import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.optimizers import Adam
import os
import tensorflow as tf

def keras_model_fn(_, config):

    embeddings_index = dict()
    file = open(config["embeddings_path"], encoding="utf-8") #Add file path here
    for i in file:
        values = i.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    vocab_size = config["embeddings_dictionary_size"] # Change this part with the correct file and name

    words_rows = len(embeddings_index.keys())

    values_columns = len(embeddings_index['the']) #Didn't understand this part, please confirm with someone

    matrix_emb = np.zeros((words_rows,values_columns))
    for index, key in zip(range(0, words_rows), embeddings_index.keys()):
        matrix_emb[index] = embeddings_index[key]

    model = Sequential()
    model.add(Embedding(vocab_size, 25, weights=[matrix_emb], name='embedding', input_length=100, trainable=True))
    model.add(Conv1D(filters = 100, kernel_size = 2,activation = 'relu', padding = 'valid', strides = 1,))
    model.add(GlobalMaxPool1D())
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    cnn_model = model
    return cnn_model


def save_model(model, output):
    model.save(output)
    print("Model successfully saved at: {}".format(output))

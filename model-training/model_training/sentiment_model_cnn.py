"""
Model definition for CNN sentiment training


"""
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
    """
    Creating a CNN model for sentiment modeling

    """

    embeddings_ind = dict()
    file = open(config["embeddings_path"], encoding="utf-8") #Add file path here
    for row in file:
        values = row.split()
        word = values[0]
        coefficients = asarray(values[1:], dtype='float32')
        embeddings_ind[word] = coefficients
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_ind))

    vocab_size = config["embeddings_dictionary_size"] 
    # Change this part with the correct file and name

    words_rows = len(embeddings_ind.keys())
    
    matrix_emb = np.zeros((words_rows,25))
    for index, key in zip(range(0, words_rows), embeddings_ind.keys()):
        matrix_emb[index] = embeddings_ind[key]

    model = Sequential()
    model.add(Embedding(vocab_size, 25, weights=[matrix_emb], name='embedding', input_length=50, trainable=True))
    model.add(Conv1D(filters = 100, kernel_size = 2,activation = 'relu', padding = 'valid', strides = 1,))
    model.add(GlobalMaxPool1D())
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    cnn_model = model
    return cnn_model

def save_model(model, output):
    """
    Method to save models in SaveModel format with signature to allow for serving


    """

    model.save(output)

    print("Model successfully saved at: {}".format(output))

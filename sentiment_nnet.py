# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:22:15 2018

@author: Ethan Beaman
"""

from gensim.utils import simple_preprocess
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import thinc.extra.datasets as dink
import pickle
import os

class SentimentClassifier:
    def __init__(self, nvocab, embed_dim, max_len, model=None):
        self._NVOCAB = nvocab
        self._EDIM = embed_dim
        self._MAXLEN = max_len
        self.model = load_model(model) if self.model is not None else None
        
    def train(self, epochs=3, batch_size=128, val_split=0.2):
        im, db = dink.imdb()
        train = im + db
        train_raw, train_y = zip(*train)
    
        train_prep = [' '.join(simple_preprocess(review)) for review in train_raw]
        tokenizer = Tokenizer(num_words=self._NVOCAB)
        tokenizer.fit_on_texts(train_prep)
        train_idx = tokenizer.texts_to_sequences(train_prep)
        train_seq = pad_sequences(train_idx, maxlen=self._MAXLEN)
    
        model = Sequential()
        model.add(Embedding(self._NVOCAB + 1, self._EDIM, 
                            input_length=self._MAXLEN, 
                            mask_zero=True))
        model.add(LSTM(100, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
    
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['acc'])
    
        model.fit(train_seq, list(train_y), 
                  epochs=3, 
                  batch_size=128, 
                  validation_split=0.2)
        self.model = model
        self.tokenizer = tokenizer
        
    def save_model(self, dest_path, name):
        self.model.save(os.path.join(dest_path, name))
        with open(os.path.join(dest_path, 'tokenizer.pickle'), 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

# tested earlier with 30 epochs, validation accuracy peaked at 3 at .88
# not bad!






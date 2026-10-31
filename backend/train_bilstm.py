"""OPTIONAL: train the BiLSTM NLI model from the notebook (Assignment 4). Needs `pip install tensorflow`.
Usage: python backend/train_bilstm.py [path_to_full_snli_train.csv]
Saves artifacts/bilstm.keras + tokenizer.pkl; the API then ensembles it automatically."""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Concatenate, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from nlp import clean

ROOT = Path(__file__).resolve().parent.parent; A = ROOT / 'artifacts'
LABELS = ['entailment', 'neutral', 'contradiction']; MAX_LEN = 25
src = sys.argv[1] if len(sys.argv) > 1 else ROOT / 'data/snli/train_40k.csv'
def load(p):
    d = pd.read_csv(p, engine='python').dropna(subset=['sentence1', 'sentence2'])
    d = d[d.gold_label.isin(LABELS)].reset_index(drop=True)
    d['c1'], d['c2'] = d.sentence1.map(clean), d.sentence2.map(clean); return d
tr, dv, te = load(src), load(ROOT / 'data/snli/dev_4k.csv'), load(ROOT / 'data/snli/test_4k.csv')
w2v = Word2Vec.load(str(A / 'word2vec.model')); D = w2v.vector_size
tok = Tokenizer(oov_token='<OOV>'); tok.fit_on_texts(tr.c1.tolist() + tr.c2.tolist())
V = len(tok.word_index) + 1
enc = lambda d: [pad_sequences(tok.texts_to_sequences(d[c]), maxlen=MAX_LEN, padding='post') for c in ('c1', 'c2')]
Y = lambda d: tf.keras.utils.to_categorical(d.gold_label.map(LABELS.index), 3)
emb = np.zeros((V, D))
for w, i in tok.word_index.items():
    if w in w2v.wv: emb[i] = w2v.wv[w]
E = Embedding(V, D, weights=[emb], trainable=True)
def branch(x): return GlobalMaxPooling1D()(Bidirectional(LSTM(64, return_sequences=True))(E(x)))
a, b = Input((MAX_LEN,)), Input((MAX_LEN,))
x = Dropout(0.4)(Dense(128, activation='relu')(Concatenate()([branch(a), branch(b)])))
out = Dense(3, activation='softmax')(Dropout(0.3)(Dense(64, activation='relu')(x)))
m = Model([a, b], out); m.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
m.fit(enc(tr), Y(tr), validation_data=(enc(dv), Y(dv)), epochs=10, batch_size=256,
      callbacks=[EarlyStopping(patience=2, restore_best_weights=True)])
print('Test accuracy:', m.evaluate(enc(te), Y(te), verbose=0)[1])
m.save(A / 'bilstm.keras'); pickle.dump(tok, open(A / 'tokenizer.pkl', 'wb'))

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 19:56:54 2019

@author: User
"""

import pandas as pd
import numpy as np
import h5py
from gensim.models import Word2Vec
import gensim.models
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.models import Sequential
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense, Dropout, Activation, Bidirectional
from keras.layers import LSTM, SimpleRNN, GRU
import keras.callbacks
import re
from nltk.tokenize import word_tokenize
import utils
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
nltk.download('stopwords')
nltk.download('punkt')

##importer le word2vec google
WORD2VEC_VECTORS_BIN = "C:\\Users\\User\\Desktop\\GoogleNews-vectors-negative300.bin"

w2v = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_VECTORS_BIN, binary=True)

##lecture des donnes
data_clickbait=pd.read_csv('clickbait_data',sep='\n',header=None)
data_noclickbait=pd.read_csv('non_clickbait_data',sep='\n',header=None)

data_clickbait.insert(1,"class",np.ones(15999))
data_noclickbait.insert(1,"class",np.zeros(16001))

data_Final=pd.concat((data_clickbait,data_noclickbait),ignore_index=True) # concatenation des click vs nonclick data
df=data_Final.rename(columns={0:"text"}) # renommer la colonne 0
df=df.sample(frac=1).reset_index(drop=True) # faire un shuffle avec reset index ( index remit a zero)

##Data cleaning
df['cleaned_tweet'] = df['text'].apply(preprocess.clean_tweet)


##separer les donnes de façon non aléatoire pour eviter d'avoir un probleme lors de la cross validation a cause des indexs

split=int(2*len(df)/3)

text_train,text_test,y_train,y_test=df.iloc[:split,2],df.iloc[split:,2],df.iloc[:split,1],df.iloc[split:,1]

## appliquer le word2vec de google mais Charactere par Charactere !!

dimsize=300
sequence_size=30

def compute_training_matrix(Text):
    X = np.zeros((len(Text), sequence_size, dimsize))
    for idx, review in enumerate(Text):
        sequence = np.zeros((sequence_size, dimsize))
        tokens = review
        count = 0
        for token in tokens:
            if count == sequence_size:
                break
            try:
                token = token.lower()
                sequence[count] = w2v[token]
                count += 1
            except:
                pass
        X[idx] = sequence

    return X

X_train=compute_training_matrix(text_train)
X_test=compute_training_matrix(text_test)

##creer le model 
batch_size=128
def creat_model():
    
    model = Sequential()
    model.add(Bidirectional(SimpleRNN(128, dropout=0.1), input_shape=(sequence_size, dimsize)))  # try using a GRU and LSTM
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    
    return model

## definir un fichier qui sauvegarde les poids du meilleur modele
cp=ModelCheckpoint("best_model.h5",verbose=1,save_best_only=True) 
earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')

callback_list=[cp,earlystop_cb]  

# try using different optimizers and different optimizer configs
def fit_model(model,X_train,y_train,x_valid,y_valid,batch_size=batch_size):

    #arlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')

    model.fit(X_train, y_train, batch_size=batch_size, epochs=20,validation_data=[x_valid,y_valid],callbacks=callback_list)
    #score, acc = model.evaluate(X_test, y_test,batch_size=batch_size)
    
    return model


##cross_validation 10folds

n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True)


for i, (train, valid) in enumerate(skf.split(X_train,y_train)):
    print(X_train[train].shape,y_train[train].shape,X_train[valid].shape,y_train[valid].shape)
    print ("Running Fold", i+1, "/", n_folds)
    model = None # Clearing the NN.
    model = creat_model()
    hist=fit_model(model, X_train[train], y_train[train], X_train[valid], y_train[valid])
    
##you can plot the loss function for train and valid per epoch
loss_train=hist.history['loss']
loss_val=hist.history['val_loss']
plt.plot(loss_train,"b",label="loss_train")
plt.plot(loss_val,"r",label="loss_valid")
plt.title("loss over training epochs")
plt.legend()
plt.show()    


#sauvegarder les poids du meilleure model pour les utiliser dans le teste
hist.load_weights("best_model.h5")

##EVALUATION
score, acc = hist.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
#prediction
y_pred = model.predict_classes(X_test)
y_scores = model.predict_proba(X_test)

##afficher les résultats de la prediction
roc = roc_auc_score(y_test, y_scores)
print('ROC score:', roc)

metrics = classification_report(y_test, y_pred, digits=4)
print('Classification Report \n')
print(metrics)

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix \n')
print (cm)

    

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 20:01:57 2019

@author: User
"""


import pandas as pd
import numpy as np
import h5py
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Activation, Input, Dropout, MaxPooling1D, Conv1D,Lambda
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from keras.layers import LSTM,concatenate ,SimpleRNN, GRU
from keras.models import Model
import preprocess
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
nltk.download('stopwords')
nltk.download('punkt')

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

##character representation based on the indices
all_txt = ''
for tweet in df['cleaned_tweet'].values:
    all_txt += tweet

chars = set(all_txt)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 250
def binarize(x, sz=37):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))

def create_feature_matrix(docs):
    X = np.ones((len(docs), maxlen), dtype=np.int64) * -1

    for i, doc in enumerate(docs):
        for t, char in enumerate(doc):
            X[i, t] = char_indices[char]
        
    return X

##separer les donnes de façon non aléatoire pour eviter d'avoir un probleme lors de la cross validation a cause des indexs

split=int(2*len(df)/3)

text_train,text_test,y_train,y_test=df.iloc[:split,2],df.iloc[split:,2],df.iloc[:split,1],df.iloc[split:,1]

##appliquer la fonction create feature matrix
x_train, x_test = create_feature_matrix(text_train), create_feature_matrix(text_test)
Y_train, Y_test = np.array(y_train), np.array(y_test)

##charactere Embedding with 3 layer of CONVOLUTION 1D 
batch_size=64
def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 37

def creat_model():
        
    filter_length = [5, 3, 3]
    nb_filter = [196, 196, 300]
    pool_length = 2

    in_sentence = Input(shape=(maxlen,), dtype='int64')
    # binarize function creates a onehot encoding of each character index
    embedded = Lambda(binarize, output_shape=binarize_outshape)(in_sentence)
    # embedded: encodes sentence
    for i in range(len(nb_filter)):
        embedded = Conv1D(nb_filter=nb_filter[i],
                            kernel_size=filter_length[i],
                            padding='valid',
                            activation='relu',
                            kernel_initializer='glorot_normal',
                            strides=1)(embedded)

        embedded = Dropout(0.1)(embedded)
        embedded = MaxPooling1D(pool_length=pool_length)(embedded)

    forward_sent = GRU(128, return_sequences=False, dropout=0.2, consume_less='gpu')(embedded)#use LSTM and SimpleRNN also
    backward_sent = GRU(128, return_sequences=False, dropout=0.2, consume_less='gpu', go_backwards=True)(embedded)

    sent_encode = concatenate([forward_sent, backward_sent])
    sent_encode = Dropout(0.3)(sent_encode)
    output = Dense(1, activation='sigmoid')(sent_encode)
    model = Model(input=in_sentence, output=output)
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model
    
## definir un fichier qui sauvegarde les poids du meilleur modele
cp=ModelCheckpoint("best_model.h5",verbose=1,save_best_only=True)
earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')
 
callback_list=[cp,earlystop_cb]  

# try using different optimizers and different optimizer configs
def fit_model(model,X_train,y_train,x_valid,y_valid,batch_size = batch_size):

    #earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')

    model.fit(X_train, y_train, batch_size=batch_size, epochs=20,validation_data=[x_valid,y_valid],callbacks=callback_list)
    #score, acc = model.evaluate(X_test, y_test,batch_size=batch_size)
    
    return model


##cross_validation 10folds

n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True)


for i, (train, valid) in enumerate(skf.split(x_train,Y_train)):
    #print(X_train[train].shape,y_train[train].shape,X_train[valid].shape,y_train[valid].shape)
    print ("Running Fold", i+1, "/", n_folds)
    model = None # Clearing the NN.
    model = creat_model()
    hist=fit_model(model, x_train[train], Y_train[train], x_train[valid], Y_train[valid])
    
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
score, acc = hist.evaluate(x_test, Y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
#prediction
Y_pred = predict_classes(model,x_test)
Y_scores = model.predict(x_test)

##afficher les résultats de la prediction
roc = roc_auc_score(Y_test, Y_scores)
print('ROC score:', roc)

metrics = classification_report(Y_test, Y_pred, digits=4)
print('Classification Report \n')
print(metrics)

cm = confusion_matrix(Y_test, Y_pred)
print('Confusion Matrix \n')
print (cm)
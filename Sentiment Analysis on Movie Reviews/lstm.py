#encoding=utf-8
# This Program is written by Victor Zhang at 2016-04-13 10:52:00
#
#
import numpy as np
import pandas as pd
import csv
import time


from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.advanced_activations import PReLU
from keras.layers.recurrent import LSTM,GRU


IS_SAMPLE=False
dicSize=0
length=0
embedding=None
index2word={0:'Nil'}
word2index={'Nil':0}
maxlen=20


def output_file(file_name,ans,Id=None,firstrow=None):
    ansSize=ans.shape[0]

    csvfile = file_name
    with open(csvfile, 'w') as output:
        predictions = []

        writer = csv.writer(output, lineterminator='\n')
        if firstrow!=None:
            writer.writerow(firstrow)

        for i in range(ansSize):
            predictions += [[Id[i],ans[i]]]
            if i%50000==0:
                writer.writerows(predictions)
                predictions=[]
        writer.writerows(predictions)
        print("Predicting done",time.ctime())


def feature_extraction(train,test):
    global maxlen
    train_features=[]
    for train_row in train['Phrase'].values:
        train_features.append([word2index[x] for x in train_row.lower().split()])
    print(train_features[:5])

    train_y=np_utils.to_categorical(train['Sentiment'].values)

    print(train_y[:5])

    test_features=[]
    for test_row in test['Phrase'].values:
        test_features.append([word2index[x] for x in test_row.lower().split()])
    print(test_features[:5])


    train_features = sequence.pad_sequences(train_features, maxlen=maxlen)
    test_features = sequence.pad_sequences(test_features, maxlen=maxlen)
    print(train_features[:5])
    print(test_features[:5])

    return train_features,test_features,train_y


def embedding_methods(train_features,test_features,train_y,train,test):
    global maxlen
    print('Build model...')
    model = Sequential()
    print(dicSize,length,embedding.shape)
    model.add(Embedding(dicSize, length, dropout=0.5,input_length=maxlen,mask_zero=True,weights=[embedding]))
    model.add(GRU(length*2,return_sequences=True, dropout_W=0.5, dropout_U=0.1))
    model.add(Dropout(0.3))
    model.add(LSTM(length, dropout_W=0.5, dropout_U=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(200))
    model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])

    model.fit(train_features,train_y,batch_size=128,nb_epoch=3,validation_split=0.3)

    ans=model.predict(test_features,batch_size=128)

    ans=np.argmax(ans,axis=1)
    print(ans)
    output_file('./results/lstm_ans.csv',ans,test['PhraseId'].values,['PhraseId','Sentiment'])


def build_vocab():
    global index2word,word2index,dicSize,length,embedding
    embedding=np.loadtxt('./models/embedding.txt')
    dicSize,length=embedding.shape
    dicSize+=1
    zeros=np.zeros((1,length))
    embedding=np.r_[zeros,embedding]
    print(embedding[:1,:])

    ifile=open("./models/vocab.txt")
    i=1
    for line in ifile:
        x=line.strip().lower()
        word2index[x]=i
        index2word[i]=x
        i+=1


    print(word2index['a'],index2word[5])
    ifile.close()



def read_files():
    if IS_SAMPLE:
        train=pd.read_csv('data/train_min.tsv',sep='\t')
        test=pd.read_csv('data/test_min.tsv',sep='\t')
    else:
        train=pd.read_csv('data/train.tsv',sep='\t')
        test=pd.read_csv('data/test.tsv',sep='\t')
    return train,test

if __name__ == '__main__':
    build_vocab()
    train,test=read_files()
    print(train.head())
    train_features,test_features,train_y =feature_extraction(train,test)
    embedding_methods(train_features,test_features,train_y,train,test)
    # xgboost(train,test,features)

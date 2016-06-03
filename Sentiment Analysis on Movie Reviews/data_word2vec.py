#encoding=utf-8
# This Program is written by Victor Zhang at 2016-04-14 10:02:23
#
#
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec

def readtxt(txt):
    ifile=open(txt,'r',encoding='utf-8')
    st=[]
    for line in ifile:
        line=line.strip().lower().split()
        st.append(line)
    print(st[:5])
    return st

def readSeries(txt):
    st=[]
    for line in txt:
        line=line.strip().lower().split()
        st.append(line)
    print(st[:5])
    return st

def train_model(st):
    model=Word2Vec(st,size=100,window=5,min_count=1,workers=8,sg=0,hs=0,negative=10)
    model.save_word2vec_format("./models/word2vec.model")
    savevocab(model)
    return model

def feature_extraction(train,test):
    total=train['Phrase'].tolist()+test['Phrase'].tolist()
    st=readSeries(total)
    train_model(st)


def savevocab(model):
    np.savetxt("./models/embedding.txt",model.syn0)
    ifile=open("./models/vocab.txt",'w')
    for word in model.index2word:
        ifile.write("%s\n"%(word))
    model=ifile.close()


def read_files():

    train=pd.read_csv('data/train.tsv',sep='\t')
    test=pd.read_csv('data/test.tsv',sep='\t')

    return train,test

if __name__ == '__main__':
    train,test=read_files()
    print(test.head())
    feature_extraction(train,test)
    # xgboost(train,test,features)

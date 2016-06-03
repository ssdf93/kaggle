#encoding=utf-8
# This Program is written by Victor Zhang at 2016-02-08 12:19:47
#
#
import numpy as np
import pandas as pd

import csv
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils

sample = False
target = 'label'


def loadData():
    if sample:
        train=pd.read_csv("data/train_min.csv")
        test=pd.read_csv("data/test_min.csv")
    else:
        train=pd.read_csv("data/train.csv")
        test=pd.read_csv("data/test.csv")

    return train,test

def NN_model(input_dim,output_dim,hn,dp):
    model = Sequential()

    model.add(Dense(output_dim=hn,input_dim=input_dim,init='uniform'))
    print("hn=",hn)
    model.add(Activation('relu'))
    model.add(Dropout(dp))

    model.add(Dense(hn,input_dim=hn,init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(dp))


    model.add(Dense(output_dim))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
    return model

def run_algorithm(train,test):
    EPOCHS=100
    BATCHES=124

    m=train.shape[1]-1

    model=NN_model(m,10,128,0.5)
    X_train=train.iloc[:,1:].values

    y_train=np_utils.to_categorical(train[target].values)

    model.fit(X_train,y_train, nb_epoch=EPOCHS, batch_size=BATCHES,shuffle=True,verbose=1,validation_split=0.2)
    X_test=test.values
    ans_m=X_test.shape[0]
    print("xtest=",X_test.shape)
    ans = model.predict_proba(X_test, verbose=0)
    ans = np_utils.categorical_probas_to_classes(ans)

    # ans = np.array(ans).reshape((ans_m,))

    print("ans.shape=",ans.shape)
    csvfile = 'results/keras-naive.csv'
    writer = csv.writer(open(csvfile,'w'), lineterminator='\n')
    writer.writerow(["ImageId,Label"])
    for i,x in enumerate(ans):
        writer.writerow([i+1,x])


if __name__ == '__main__':
    train,test = loadData()
    run_algorithm(train,test)


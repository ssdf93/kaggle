#encoding=utf-8
# This Program is written by Victor Zhang at 2016-02-01 14:37:28
#
#
import numpy as np
import pandas as pd
import xgboost as xgb
import csv
import time
from sklearn.metrics import log_loss
import operator
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder

from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils

sample = False
target='Category'


def load_data():
    if sample:
        train = pd.read_csv("./data/train_min.csv")
        test = pd.read_csv("./data/test_min.csv")
    else:
        train = pd.read_csv("./data/train.csv")
        test = pd.read_csv("./data/test.csv")
    features=['DayOfWeek','PdDistrict','Address','X','Y']
    non_numerate_features=['DayOfWeek','PdDistrict','Address']
    return train,test,features,non_numerate_features

def findWayName(st):
    wayNames=[""," AV"," ST"," Block"," DR"," WY"," BL"," LN"," RD"," BLVD"," HY"," CT"," PZ"," TR"]
    for i,wayName in enumerate(wayNames):
        if i>0 and (wayName in st):
            return i
    return 0

def data_processing(train,test,features,non_numerate_features):
    print("Adding Features",time.ctime())
    # train['StreetNo'] = train['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
    # test['StreetNo'] = test['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)

    train['wayName']=train['Address'].apply(lambda x:findWayName(x))
    test['wayName']=test['Address'].apply(lambda x:findWayName(x))
    train['isInterSection']=train['Address'].apply(lambda x: 1 if '/' in x else 0)
    test['isInterSection']=test['Address'].apply(lambda x: 1 if '/' in x else 0)
    train['Address'] = train['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
    test['Address'] = test['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)

    train['year']= train['Dates'].apply(lambda x: x[0:4] if len(x) > 4 else 12)
    test['year'] = test['Dates'].apply(lambda x: x[0:4] if len(x) > 4 else 12)
    train['month']= train['Dates'].apply(lambda x: x[5:7] if len(x) > 4 else 12)
    test['month'] = test['Dates'].apply(lambda x: x[5:7] if len(x) > 4 else 12)
    train['day']= train['Dates'].apply(lambda x: x[8:10] if len(x) > 4 else 12)
    test['day'] = test['Dates'].apply(lambda x: x[8:10] if len(x) > 4 else 12)
    train['hour']= train['Dates'].apply(lambda x: x[11:13] if len(x) > 4 else 12)
    test['hour'] = test['Dates'].apply(lambda x: x[11:13] if len(x) > 4 else 12)

    train['dark'] = train['Dates'].apply(lambda x: 1 if (len(x) > 4 and (int(x[11:13]) >= 18 or int(x[11:13]) < 6)) else 0)
    test['dark'] = test['Dates'].apply(lambda x: 1 if (len(x) > 4 and (int(x[11:13]) >= 18 or int(x[11:13]) < 6)) else 0)

    features += ['wayName','isInterSection','hour','dark','month','day','year']

    # print(train[['Dates','month']])

    print("Filling NAs",time.ctime())
    # print(train.mode())
    train = train.fillna(train.median().iloc[0])
    test = test.fillna(test.median().iloc[0])
    print("Label Encoder",time.ctime())
    le=LabelEncoder()
    for col in non_numerate_features:
        le.fit(list(train[col])+list(test[col]))
        train[col]=le.transform(train[col])
        test[col]=le.transform(test[col])

    le.fit(list(train[target]))
    train[target]=le.transform(train[target])

    print("Standard Scalaer",time.ctime())
    scaler=StandardScaler()
    for col in features:
        scaler.fit(list(train[col]))
        train[col]=scaler.transform(train[col])
        test[col]=scaler.transform(test[col])

    return train,test,features

def build_model(input_dim,output_dim,hn=32,dp=0.5,layers=1):
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

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def NN(train,test,features):
    EPOCHS = 10
    BATCHES = 128
    HN = 64
    input_dim=len(features)
    print(input_dim)
    print(train[features].shape)
    output_dim=39

    model = build_model(input_dim, output_dim, HN)

    print("Start Training",time.ctime())
    print(train[target].shape)
    X_train=train[features].values
    y_train=np_utils.to_categorical(train[target].values)
    model.fit(X_train,y_train, nb_epoch=EPOCHS, batch_size=BATCHES,show_accuracy=True,shuffle=True,verbose=1,validation_split=0.2)
    print("Start Predicting",time.ctime())
    ans = model.predict_proba(test[features].values, verbose=0)
    ansSize=ans.shape[0]

    csvfile = 'result/keras-submit.csv'
    with open(csvfile, 'w') as output:
        predictions = []

        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(['Id','ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT',
                         'DRIVING UNDER THE INFLUENCE','DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT',
                         'EXTORTION','FAMILY OFFENSES','FORGERY/COUNTERFEITING','FRAUD','GAMBLING',
                         'KIDNAPPING','LARCENY/THEFT','LIQUOR LAWS','LOITERING','MISSING PERSON',
                         'NON-CRIMINAL','OTHER OFFENSES','PORNOGRAPHY/OBSCENE MAT','PROSTITUTION',
                         'RECOVERED VEHICLE','ROBBERY','RUNAWAY','SECONDARY CODES','SEX OFFENSES FORCIBLE',
                         'SEX OFFENSES NON FORCIBLE','STOLEN PROPERTY','SUICIDE','SUSPICIOUS OCC','TREA',
                         'TRESPASS','VANDALISM','VEHICLE THEFT','WARRANTS','WEAPON LAWS'])

        for i in range(ansSize):
            # import pdb;pdb.set_trace()
            predictions += [[i]+ans[i].tolist()]
            if i%50000==0:
                writer.writerows(predictions)
                predictions=[]
        writer.writerows(predictions)
        print("Predicting done",time.ctime())



if __name__ == '__main__':
    train,test,features,non_numerate_features=load_data()
    train,test,features=data_processing(train,test,features,non_numerate_features)
    NN(train,test,features)

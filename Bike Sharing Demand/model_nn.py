#encoding=utf-8
# This Program is written by Victor Zhang at 2016-04-22 19:47:39
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
from keras.layers.advanced_activations import PReLU



sample = False
target='count'
imax=0


def load_data():
    if sample:
        train = pd.read_csv("./data/train_min.csv")
        test = pd.read_csv("./data/test_min.csv")
    else:
        train = pd.read_csv("./data/train.csv")
        test = pd.read_csv("./data/test.csv")

    return train,test

def birthAge(x):
    sts=x.split(' ')
    if 'day' in x:
        return int(sts[0])
    elif 'week' in x:
        return int(sts[0])+10
    elif 'month' in x:
        return int(sts[0])+30
    elif 'year' in x:
        return int(sts[0])+50
    else:
        return 0

def get_date(st,date_type):
    if date_type=='year':
        return int(st[:4])
    elif date_type=='month':
        return int(st[5:7])
    elif date_type=='day':
        return int(st[8:10])
    elif date_type=='hour':
        return int(st[11:13])
    else:
        return 0

def get_period(x):
    if x>=7 and x<12:
        return "1"
    elif x>=12 and x<15:
        return "2"
    elif x>15 and x<20:
        return "3"
    elif x>=20 and x<24:
        return "4"
    else:
        return "0"


def data_processing(train,test):
    features=['season','holiday','workingday','weather','temp','atemp','humidity','windspeed']
    for data in [train,test]:
        data['month']=data['datetime'].apply(lambda x: get_date(x,'month'))
        data['year']=data['datetime'].apply(lambda x: get_date(x,'year'))
        data['hour']=data['datetime'].apply(lambda x: get_date(x,'hour'))
        data['day']=data['datetime'].apply(lambda x: get_date(x,'day'))
        data['period']=data['hour'].apply(lambda x: get_period(x))


    features+=['year','month','day','hour','period']

    # print("Label Encoder",time.ctime())
    # le=LabelEncoder()
    # for col in features:
    #     train[col]=train[col].fillna('Unknown')
    #     test[col]=test[col].fillna('Unknown')
    #     le.fit(list(train[col])+list(test[col]))
    #     train[col]=le.transform(train[col])
    #     test[col]=le.transform(test[col])

    # le.fit(list(train[target]))
    # train[target]=le.transform(train[target])

    global imax
    imax=np.max(train[target])
    train[target]=train[target]/imax

    print("Standard Scalaer",time.ctime())
    scaler=StandardScaler()
    for col in features:
        scaler.fit(list(train[col])+list(test[col]))
        train[col]=scaler.transform(train[col])
        test[col]=scaler.transform(test[col])

    # print(data['birthAge'][:10])


    return train,test,features


def write_csv(file_name,ans,first_row=None,myId=None):
    """
    Write the data to csv ,
    """
    # csvfile = 'results/xgboost-feature-submit.csv'

    ansSize=ans.shape[0]
    with open(file_name, 'w') as output:
        predictions = []

        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(first_row)

        for i in range(ansSize):
            # import pdb;pdb.set_trace()
            total_cnt=[int(i+0.5) for i in ans[i].tolist() ]
            predictions += [[myId[i]]+total_cnt]
            if (i+1)%50000==0:
                writer.writerows(predictions)
                predictions=[]

        if predictions != None:
            writer.writerows(predictions)

        print("Writing CSV done",time.ctime())

    # outfile = open('result/xgb.fmap', 'w')
    # i = 0
    # for feat in features:
    #     outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    #     i = i + 1
    # outfile.close()
    # importance = classifier.get_fscore(fmap='result/xgb.fmap')
    # importance = sorted(importance.items(), key=operator.itemgetter(1))
    # df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    # df.to_csv('result/importance.csv',index=False)



def neural_networks(train,test,features):

    input_dim=len(features)

    model = Sequential()
    model.add(Dense(output_dim=200,input_dim=input_dim,init='glorot_uniform'))
    model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=100,init='glorot_uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))

    model.add(Dense(output_dim=200,init='glorot_uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(output_dim=100,init='glorot_uniform'))
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('hard_sigmoid'))

    model.compile(loss='mean_squared_logarithmic_error', optimizer='sgd')

    y_train=train[target].values

    model.fit(train[features].values,y_train,batch_size=16, nb_epoch=30, verbose=1,  validation_split=0.3)


    print("Start Predicting",time.ctime())

    ans = model.predict_proba(test[features].values, verbose=0)*imax

    first_row="datetime,count".split(',')
    myId=train['datetime'].values
    write_csv('results/nn-feature-submit.csv',ans,first_row,myId)



if __name__ == '__main__':
    train,test=load_data()
    print(test.head())
    train,test,features=data_processing(train,test)
    print(test.head())
    neural_networks(train,test,features)

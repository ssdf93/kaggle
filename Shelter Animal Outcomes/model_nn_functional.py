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
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.advanced_activations import PReLU


from keras.models import Model
from keras.layers import Input, Dense



sample = False
target='OutcomeType'


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
        return "Morning"
    elif x>=12 and x<15:
        return "Noon"
    elif x>15 and x<20:
        return "AfterNoon"
    elif x>=20 and x<24:
        return "Night"
    else:
        return "MidNight"


def data_processing(train,test):
    features=['Color','SexuponOutcome','AnimalType','Breed']
    for data in [train,test]:
        data['AgeuponOutcome']=data['AgeuponOutcome'].fillna('Unknown')
        data['BirthAge']=data['AgeuponOutcome'].apply(lambda x: birthAge(x))
        data['Month']=data['DateTime'].apply(lambda x: get_date(x,'month'))
        data['Year']=data['DateTime'].apply(lambda x: get_date(x,'year'))
        data['Hour']=data['DateTime'].apply(lambda x: get_date(x,'hour'))
        data['Day']=data['DateTime'].apply(lambda x: get_date(x,'day'))
        data['Period']=data['Hour'].apply(lambda x: get_period(x))


    features+=['BirthAge','Year','Month','Day','Hour','Period']

    print("Label Encoder",time.ctime())
    le=LabelEncoder()
    for col in features:
        train[col]=train[col].fillna('Unknown')
        test[col]=test[col].fillna('Unknown')
        le.fit(list(train[col])+list(test[col]))
        train[col]=le.transform(train[col])
        test[col]=le.transform(test[col])

    le.fit(list(train[target]))
    train[target]=le.transform(train[target])

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
            predictions += [[myId[i]+1]+ans[i].tolist()]
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

    input_data=Input(shape=(input_dim,))
    dense1=Dense(200,init='glorot_uniform',activation='relu')(input_data)
    dense2=Dense(100,init='glorot_uniform',activation='relu')(dense1)
    dense3=Dense(200,init='glorot_uniform',activation='relu')(dense2)
    output_data=Dense(5,init='glorot_uniform',activation='softmax')(dense3)

    model=Model(input=input_data,output=output_data)
    # model = Sequential()
    # model.add(Dense(output_dim=200,input_dim=input_dim,init='glorot_uniform'))
    # model.add(PReLU())
    # model.add(Dropout(0.5))
    # model.add(Dense(output_dim=100,init='glorot_uniform'))
    # model.add(PReLU())
    # model.add(Dropout(0.5))

    # model.add(Dense(output_dim=200,init='glorot_uniform'))
    # model.add(PReLU())
    # model.add(Dropout(0.5))

    # model.add(Dense(output_dim=100,init='glorot_uniform'))
    # model.add(PReLU())
    # model.add(Dropout(0.5))


    # model.add(Dense(5))
    # model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])

    y_train=np_utils.to_categorical(train[target].values)

    model.fit(train[features].values,y_train,batch_size=16, nb_epoch=10, verbose=1,  validation_split=0.3)


    print("Start Predicting",time.ctime())

    ans = model.predict(test[features].values, verbose=0)

    first_row="ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer".split(',')
    myId=list(range(ans.shape[0]))
    write_csv('results/nn-feature-submit.csv',ans,first_row,myId)



if __name__ == '__main__':
    train,test=load_data()
    train,test,features=data_processing(train,test)
    neural_networks(train,test,features)

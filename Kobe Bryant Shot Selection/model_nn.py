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
import random
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder

from keras.models import Model
from keras.layers import Input, Dense, Dropout

# import matplotlib.pyplot as plt


sample = False
target='shot_made_flag'


def load_data():
    if sample:
        data = pd.read_csv("./data/data_min.csv")
    else:
        data = pd.read_csv("./data/data.csv")

    print(data['shot_distance'].max())

    train=data[data[target].notnull()]
    test=data[data[target].isnull()]

    return data,train,test


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
    features=['combined_shot_type','shot_zone_area','shot_zone_basic','action_type','opponent','shot_type']
    #deleted=[]
    for data in [train,test]:
        data['month']=data['game_date'].apply(lambda x: get_date(x,'month'))
        data['year']=data['game_date'].apply(lambda x: get_date(x,'year'))
        data['day']=data['game_date'].apply(lambda x: get_date(x,'day'))
        data['range']=data['shot_zone_range'].map({'16-24 ft.':20, '8-16 ft.':12, 'Less Than 8 ft.':4, '24+ ft.':28, 'Back Court Shot':36})
        data['first_action_type']=data['action_type'].apply(lambda x: x.split(' ')[0])
        data['season']=data['season'].apply(lambda x: int(x.split('-')[1]))


    print(train['first_action_type'].unique())


    features+=['first_action_type']

    print("Label Encoder",time.ctime())
    le=LabelEncoder()
    for col in features:
        # train[col]=train[col].fillna('Unknown')
        # test[col]=test[col].fillna('Unknown')
        ilist=list(train[col])+list(test[col])
        ilist=list(set(ilist))
        random.shuffle(ilist)
        le.fit(ilist)
        train[col]=le.transform(train[col])
        test[col]=le.transform(test[col])


    features+=['loc_x','loc_y','period','minutes_remaining','seconds_remaining','shot_distance','month','year','day','season','range']

    #deleted=[,]

    print("Standard Scalaer",time.ctime())
    scaler=StandardScaler()
    for col in features:
        scaler.fit(list(train[col])+list(test[col]))
        train[col]=scaler.transform(train[col])
        test[col]=scaler.transform(test[col])

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
            predictions += [[myId[i],ans[i][0]]]
            if (i+1)%50000==0:
                writer.writerows(predictions)
                print(predictions[:5])
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



def nn(train,test,features):

    features_cnt=len(features)
    inputs=Input(shape=(features_cnt,))
    dense1=Dense(100,activation='relu')(inputs)
    dropout1=Dropout(0.5)(dense1)
    dense2=Dense(100,activation='relu')(dropout1)
    dropout2=Dropout(0.5)(dense2)
    outputs=Dense(1,activation='sigmoid')(dropout2)
    model=Model(input=inputs,output=outputs)
    model.compile(loss='binary_crossentropy', optimizer='sgd',metrics=['accuracy'])

    print("Start Training",time.ctime())
    model.fit(train[features].values,train[target].values,batch_size=32,nb_epoch=100,validation_split=0.3)



    print("Start Predicting",time.ctime())

    ans=model.predict(test[features].values)
    print(ans)
    first_row="shot_id,shot_made_flag".split(',')
    myId=test['shot_id'].values
    write_csv('results/nn-feature-submit.csv',ans,first_row,myId)




if __name__ == '__main__':
    data,train,test=load_data()
    # iplot(data)
    print(test.head())
    train,test,features=data_processing(train,test)
    nn(train,test,features)

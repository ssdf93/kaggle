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

sample = False
target='Category'
fileName='hadoop1_060223'

def load_data():
    if sample:
        train = pd.read_csv("./data/train_min.csv")
        test = pd.read_csv("./data/test_min.csv")
    else:
        train = pd.read_csv("./data/train.csv")
        test = pd.read_csv("./data/test.csv")

    # print(train.info())
    # print(test.info())
    # print(train['Category'].unique())
    return train,test


wayNames=[" AV"," ST"," Block"," DR"," WY"," BL"," LN"," RD"," BLVD"," HY"," CT"," PZ"," TR","/"]

def applyEuqalFunction(data,apply_name,col,value):
    data[apply_name]=data[col].apply(lambda x: 1 if x==value else 0)
    return data

def applyInFunction(data,apply_name,col,value):
    data[apply_name]=data[col].apply(lambda x: 1 if value in x else 0)
    return data

# def findWayName(st):
#     for i,wayName in enumerate(wayNames):
#         if i>0 and (wayName in st):
#             return i
#     return 0

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



def data_processing(train,test):


    features=[]

    print("Adding Features",time.ctime())

    for way in wayNames:
        train=applyInFunction(train,"wayName_"+way,'Address',way)
        test=applyInFunction(test,"wayName_"+way,'Address',way)
        features.append("wayName_"+way)

    # train['isInterSection']=train['Address'].apply(lambda x: 1 if '/' in x else 0)
    # test['isInterSection']=test['Address'].apply(lambda x: 1 if '/' in x else 0)
    for data in [train,test]:
        data['month']=data['Dates'].apply(lambda x: get_date(x,'month'))
        data['year']=data['Dates'].apply(lambda x: get_date(x,'year'))
        data['day']=data['Dates'].apply(lambda x: get_date(x,'day'))
        data['hour']= data['Dates'].apply(lambda x: int(x[11:13]) if len(x) > 4 else 12)
        data['dark'] = data['Dates'].apply(lambda x: 1 if (len(x) > 4 and (int(x[11:13]) >= 18 or int(x[11:13]) < 6)) else 0)
        data['period'] = data['hour'].apply(lambda x:get_period(x))
        data['AddressNumber'] = data['Address'].apply(lambda x: int(x.split(' ', 1)[0]) if x.split(' ', 1)[0].isdigit() else 0)


    features += ['X','Y','month','year','day','hour','dark','AddressNumber']

    dummy_variables=['period','DayOfWeek','PdDistrict',]


    for var in dummy_variables:
        encoded = pd.get_dummies(pd.concat([train[dummy_variables],test[dummy_variables]], axis=0)[var],prefix=var)

        train_rows = train.shape[0]
        train_encoded = encoded.iloc[:train_rows, :]
        test_encoded = encoded.iloc[train_rows:, :]

        features+=encoded.columns.tolist()
        train = pd.concat([train, train_encoded], axis=1)
        test = pd.concat([test,test_encoded],axis=1)

    # way_name_dummies=pd.get_dummies(train['wayName'],prefix='wayName',columns=wayNames)
    # features += list(way_name_dummies.columns)
    # print(way_name_dummies.columns)
    # train.join(way_name_dummies)
    # way_name_dummies=pd.get_dummies(test['wayName'],prefix='wayName',columns=wayNames)
    # test.join(way_name_dummies)

    # print(train[['Address','StreetNo']])

    # print("Filling NAs",time.ctime())
    # train = train.fillna(train.median().iloc[0])
    # test = test.fillna(test.median().iloc[0])


    print(train[features].head())
    print(test[features].head())

    print("Label Encoder",time.ctime())
    le=LabelEncoder()
    # for col in features:
    #     le.fit(list(train[col])+list(test[col]))
    #     train[col]=le.transform(train[col])
    #     test[col]=le.transform(test[col])

    le.fit(list(train[target]))
    train[target]=le.transform(train[target])

    print("Standard Scalaer",time.ctime())
    scaler=StandardScaler()
    for col in features:
        scaler.fit(list(train[col]))
        train[col]=scaler.transform(train[col])
        test[col]=scaler.transform(test[col])

    return train,test,features

def XG_boost(train,test,features):
    params = {'max_depth':8, 'eta':0.02, 'silent':1,
              'objective':'multi:softprob', 'num_class':39, 'eval_metric':'mlogloss',
              'min_child_weight':3, 'subsample':0.6,'colsample_bytree':0.6, 'nthread':4}
    num_rounds = 800

    print(params.items())


    xgbtrain = xgb.DMatrix(train[features], label=train[target])
    dtest=xgb.DMatrix(test[features])

    print("Start Cross Validation",time.ctime())
    cv_results=xgb.cv(params, xgbtrain, num_rounds, nfold=5, metrics={'mlogloss'})
    # print(cv_results)
    cv_results.to_csv('results/xgboost_epoch%s.csv'%(fileName))


    print("Start Training",time.ctime())
    watchlist = [ (xgbtrain,'train'), (dtest, 'test') ]
    classifier = xgb.train(params, xgbtrain, num_rounds)
    print("Start Predicting",time.ctime())

    ans=classifier.predict(dtest)
    ansSize=ans.shape[0]

    csvfile = 'results/xgboost-feature-submit%s.csv'%(fileName)
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


    importance = classifier.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df.to_csv('results/importance%s.csv'%(fileName),index=False)


if __name__ == '__main__':
    train,test=load_data()
    train,test,features=data_processing(train,test)
    XG_boost(train,test,features)

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

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# import matplotlib.pyplot as plt


sample = False
target='shot_made_flag'


def load_data():
    if sample:
        data = pd.read_csv("./data/data_min.csv")
    else:
        data = pd.read_csv("./data/data.csv")

    print(data['shot_distance'].max())

    train=data[data[target].notnull()].reset_index(drop=True)
    test=data[data[target].isnull()].reset_index(drop=True)

    print(train.info())
    print(test.info())

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

def ireplace(x):
    print(x)
    return x.replace(' ','_')


def data_processing(train,test):
    # if os.path.exists('features/train_features.csv') and os.path.exists('features/test_features.csv')


    features=[]
    #deleted=[]
    for data in [train,test]:
        data['month']=data['game_date'].apply(lambda x: get_date(x,'month'))
        data['year']=data['game_date'].apply(lambda x: get_date(x,'year'))
        data['day']=data['game_date'].apply(lambda x: get_date(x,'day'))
        data['range']=data['shot_zone_range'].map({'16-24 ft.':20, '8-16 ft.':12, 'Less Than 8 ft.':4, '24+ ft.':28, 'Back Court Shot':36})
        # data['first_action_type']=data['action_type'].apply(lambda x: x.split(' ')[0])
        data['season']=data['season'].apply(lambda x: int(x.split('-')[1]))
        data['is_host']=data['matchup'].apply(lambda x: 1 if '@' in x else 0)
        data['shot_type']=data['shot_type'].apply(lambda x: 1 if '2' in x else 0)
        data['is_addTime']=data['period'].apply(lambda x:1 if x>4 else 0)

    # print(train['first_action_type'].unique())

    dummy_variables=['action_type','combined_shot_type','shot_zone_area','shot_zone_basic','opponent']
    for var in dummy_variables:
        encoded = pd.get_dummies(pd.concat([train,test], axis=0)[var],prefix=var)

        train_rows = train.shape[0]
        train_encoded = encoded.iloc[:train_rows, :]
        test_encoded = encoded.iloc[train_rows:, :]
        # print(train_encoded.head())
        # print(test_encoded.head())

        # train[var]=train[var].apply(lambda x: ireplace(x))
        # dummies=pd.get_dummies(train[var],prefix=var)
        features+=encoded.columns.tolist()
        train = pd.concat([train, train_encoded], axis=1)


        # test[var]=test[var].apply(lambda x: ireplace(x))
        # dummies=pd.get_dummies(test[var],prefix=var)
        test = pd.concat([test,test_encoded],axis=1)

    print(len(features))
    print(test[features].head())
    # features+=['first_action_type']

    # print("Label Encoder",time.ctime())
    # le=LabelEncoder()
    # for col in features:
    #     ilist=list(train[col])+list(test[col])
    #     ilist=list(set(ilist))
    #     random.shuffle(ilist)
    #     le.fit(ilist)
    #     train[col]=le.transform(train[col])
    #     test[col]=le.transform(test[col])


    features+=['loc_x','loc_y','period','minutes_remaining','seconds_remaining','shot_distance','month','year','day','season','range','is_host','shot_type','is_addTime']

    #deleted=[,]
    # print(train.columns.tolist())
    # print(train['Hook Shot'].unique())

    print("MinMax Scaler",time.ctime())
    scaler=MinMaxScaler()
    for col in features:
        scaler.fit(list(train[col])+list(test[col]))
        train[col]=scaler.transform(train[col])
        test[col]=scaler.transform(test[col])

    # print(train[features])
    # train.to_csv("train_features.csv",index=None)
    # test.to_csv("test_features.csv",index=None)
    # print(features)
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
            predictions += [[myId[i],ans[i]]]
            if (i+1)%50000==0:
                writer.writerows(predictions)
                predictions=[]

        if predictions != None:
            writer.writerows(predictions)

        print("Writing CSV done",time.ctime())



def XG_boost(train,test,features):
    # params = {'max_depth':8, 'eta':0.05,'silent':1,
    #           'objective':'binary:logistic', 'eval_metric': 'logloss',
    #           'min_child_weight':3, 'subsample':0.6,'colsample_bytree':0.6, 'nthread':4}


    #0.60061
    # params = {'max_depth':8, 'eta':0.02,'silent':1,
    #           'objective':'binary:logistic', 'eval_metric': 'logloss',
    #           'min_child_weight':3, 'subsample':0.5,'colsample_bytree':0.5, 'nthread':4}
    # num_rounds = 290

    params = {'max_depth':9, 'eta':0.015,'silent':1,
              'objective':'binary:logistic', 'eval_metric': 'logloss',
              'min_child_weight':4, 'subsample':0.6,'colsample_bytree':0.6}
    num_rounds = 300
    n_samples=train.shape[0]
    shuffled_index=np.arange(n_samples)
    np.random.shuffle(shuffled_index)
    train_index=shuffled_index[:int(n_samples*.7)]
    dev_index=shuffled_index[int(n_samples*.7):]

    print(train.shape,np.sum(train[target]==1))
    # print(train.loc[train_index,features].shape)
    dev=train.loc[dev_index,target]
    print(dev.shape,np.sum(dev==1))

    xgbtrain = xgb.DMatrix(train.loc[train_index,features], label=train.loc[train_index,target])
    xgbdev = xgb.DMatrix(train.loc[dev_index,features], label=train.loc[dev_index,target])

    dtest=xgb.DMatrix(test[features])
    # print("Start Cross Validation",time.ctime())


    # cv_results=xgb.cv(params, xgbtrain, num_rounds, nfold=5,metrics={'logloss'}, seed = 0)
    # print(cv_results)
    # cv_results.to_csv('models/epoch_score.csv')
    print("Start Training",time.ctime())

    watchlist = [(xgbdev,'eval'), (xgbtrain,'train')]
    evals_result = {}
    classifier = xgb.train(params, xgbtrain, num_rounds, watchlist)

    print("Start Predicting",time.ctime())

    ans=classifier.predict(dtest)
    print(ans)
    first_row="shot_id,shot_made_flag".split(',')
    myId=test['shot_id'].values
    write_csv('results/xgboost-dummy-feature-submit.csv',ans,first_row,myId)

    classifier.dump_model('models/dump.raw.txt')
    # xgb.plot_importance(classifier)
    # print(classifier.get_fscore())

    importance = classifier.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    print(importance)
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df.to_csv('models/importance.csv',index=False)


if __name__ == '__main__':
    data,train,test=load_data()
    # iplot(data)
    # print(test.head())
    # train,test,features=data_processing(train,test)
    XG_boost(train,test,features)

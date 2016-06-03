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
            predictions += [[myId[i],ans[i]]]
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



def XG_boost(train,test,features):
    # params = {'max_depth':8, 'eta':0.05,'silent':1,
    #           'objective':'binary:logistic', 'eval_metric': 'logloss',
    #           'min_child_weight':3, 'subsample':0.6,'colsample_bytree':0.6, 'nthread':4}


    #0.60753
    params = {'max_depth':10, 'eta':0.01,'silent':1,
              'objective':'binary:logistic', 'eval_metric': 'logloss',
              'min_child_weight':3, 'subsample':0.5,'colsample_bytree':0.5, 'nthread':4}





    num_rounds = 400

    xgbtrain = xgb.DMatrix(train[features], label=train[target])
    dtest=xgb.DMatrix(test[features])
    print("Start Cross Validation",time.ctime())


    cv_results=xgb.cv(params, xgbtrain, num_rounds, nfold=5,metrics={'logloss'}, seed = 0)
    print(cv_results)
    cv_results.to_csv('models/haha.csv')
    print("Start Training",time.ctime())

    watchlist = [(dtest,'eval'), (xgbtrain,'train')]
    evals_result = {}
    classifier = xgb.train(params, xgbtrain, num_rounds)

    print("Start Predicting",time.ctime())

    ans=classifier.predict(dtest)
    print(ans)
    first_row="shot_id,shot_made_flag".split(',')
    myId=test['shot_id'].values
    write_csv('results/xgboost-feature-submit.csv',ans,first_row,myId)

    classifier.dump_model('models/dump.raw.txt')
    # xgb.plot_importance(classifier)
    print(classifier.get_fscore())



if __name__ == '__main__':
    data,train,test=load_data()
    # iplot(data)
    print(test.head())
    train,test,features=data_processing(train,test)
    XG_boost(train,test,features)

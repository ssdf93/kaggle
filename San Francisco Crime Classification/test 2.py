#encoding=utf-8
# This Program is written by Victor Zhang at 2016-02-01 14:37:28
#
#
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

from sklearn.preprocessing import StandardScaler, LabelEncoder

sample = True
target='Category'


def load_data():
    if sample:
        train = pd.read_csv("./data/train_min.csv")
        test = pd.read_csv("./data/test_min.csv")
    else:
        train = pd.read_csv("./data/train.csv")
        test = pd.read_csv("./data/test.csv")
    features=['Dates','DayOfWeek','PdDistrict','Address','X','Y']
    return train,test,features

def data_processing(train,test,features):
    # train['StreetNo'] = train['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
    # test['StreetNo'] = test['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
    # train['Address'] = train['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
    # test['Address'] = test['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
    # train['hour'] = train['Dates'].apply(lambda x: x[11:13] if len(x) > 4 else 12)
    # test['hour'] = test['Dates'].apply(lambda x: x[11:13] if len(x) > 4 else 12)
    # train['dark'] = train['Dates'].apply(lambda x: 1 if (len(x) > 4 and int(x[11:13]) >= 18 and int(x[11:13]) < 6) else 0)
    # test['dark'] = test['Dates'].apply(lambda x: 1 if (len(x) > 4 and int(x[11:13]) >= 18 and int(x[11:13]) < 6) else 0)
    # features += ['hour','dark','StreetNo']

    print("Filling NAs")
    # print(train.mode())
    train = train.fillna(train.median().iloc[0])
    test = test.fillna(test.median().iloc[0])
    print("Label Encoder")
    le=LabelEncoder()
    for col in features:
        le.fit(list(train[col])+list(test[col]))
        train[col]=le.transform(train[col])
        test[col]=le.transform(test[col])

    le.fit(list(train[target]))

    print("Standard Scalaer")
    scaler=StandardScaler()
    for col in features:
        scaler.fit(list(train[col])+list(test[col]))
        train[col]=scaler.transform(train[col])
        test[col]=scaler.transform(test[col])

    return train,test,features

def XG_boost(train,test,features):
    params = {'max_depth':8, 'eta':0.05, 'silent':1,
              'objective':'multi:softprob', 'num_class':39, 'eval_metric':'mlogloss',
              'min_child_weight':3, 'subsample':0.6,'colsample_bytree':0.6, 'nthread':4}
    num_rounds = 250
    if sample:
        y=xgb.DMatrix(np.zeros((5,3)))
        # xgbtrain = xgb.DMatrix(train[list(features)], label=train[target])
        classifier = xgb.train(params, xgbtrain, num_rounds)
        score = log_loss(test[target].values, classifier.predict(xgb.DMatrix(test[features])))
        print(score)

if __name__ == '__main__':
    y=xgb.DMatrix(np.zeros((5,3)))
    train,test,features=load_data()
    train,test,features=data_processing(train,test,features)
    XG_boost(train,test,features)

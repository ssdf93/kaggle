#encoding=utf-8
# This Program is written by Victor Zhang at 2016-02-01 14:37:28
#
#
import numpy as np
import pandas as pd
import xgboost as xgb
import csv
from sklearn.metrics import log_loss

from sklearn.preprocessing import StandardScaler, LabelEncoder

sample = False
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
    train[target]=le.transform(train[target])

    print("Standard Scalaer")
    scaler=StandardScaler()
    for col in features:
        scaler.fit(list(train[col]))
        train[col]=scaler.transform(train[col])
        test[col]=scaler.transform(test[col])

    return train,test,features

def XG_boost(train,test,features):
    params = {'max_depth':8, 'eta':0.05, 'silent':1,
              'objective':'multi:softprob', 'num_class':39, 'eval_metric':'mlogloss',
              'min_child_weight':3, 'subsample':0.6,'colsample_bytree':0.6, 'nthread':4}
    num_rounds = 250
    if sample:
        print("Start Training")
        xgbtrain = xgb.DMatrix(train[features], label=train[target])
        classifier = xgb.train(params, xgbtrain, num_rounds)
        print("Start Predicting")
        dtest=xgb.DMatrix(test[features])
        ans=classifier.predict(dtest)
        # print(train[features].shape)
        # print(test[features].shape)
        # print(ans)
        # print(test[target].values)

        # score = log_loss(test[target].values,ans)
        # print(score)


    else:
        from sklearn.svm import SVC
        # lr=SVC()
        # print("Start Training")
        # lr.fit(train[features],train[target])

        # print("Predicting")
        # ans=lr.predict(test[features])
        # print(ans.shape)
        # ansSize=ans.shape[0]

        print("Start Training")
        xgbtrain = xgb.DMatrix(train[features], label=train[target])
        classifier = xgb.train(params, xgbtrain, num_rounds)
        print("Start Predicting")
        dtest=xgb.DMatrix(test[features])
        ans=classifier.predict(dtest)
        ansSize=ans.shape[0]

        csvfile = 'result/xgboost-submit.csv'
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
                if i%10000==0:
                    writer.writerows(predictions)
                    predictions=[]
            writer.writerows(predictions)
            print("Predicting done")

if __name__ == '__main__':
    train,test,features=load_data()
    train,test,features=data_processing(train,test,features)
    XG_boost(train,test,features)

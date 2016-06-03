import numpy as np
import pandas as pd
import logging
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dayofweektable={'Wednesday':3, 'Tuesday':2, 'Monday':1, 'Sunday':7, 'Saturday':6, 'Friday':5, 'Thursday':4}
category_name=["ARSON","ASSAULT","BAD CHECKS","BRIBERY","BURGLARY","DISORDERLY CONDUCT","DRIVING UNDER THE INFLUENCE","DRUG/NARCOTIC","DRUNKENNESS","EMBEZZLEMENT","EXTORTION","FAMILY OFFENSES","FORGERY/COUNTERFEITING","FRAUD","GAMBLING","KIDNAPPING","LARCENY/THEFT","LIQUOR LAWS","LOITERING","MISSING PERSON","NON-CRIMINAL","OTHER OFFENSES","PORNOGRAPHY/OBSCENE MAT","PROSTITUTION","RECOVERED VEHICLE","ROBBERY","RUNAWAY","SECONDARY CODES","SEX OFFENSES FORCIBLE","SEX OFFENSES NON FORCIBLE","STOLEN PROPERTY","SUICIDE","SUSPICIOUS OCC","TREA","TRESPASS","VANDALISM","VEHICLE THEFT","WARRANTS","WEAPON LAWS"]
category_dic={}
for i,cate in enumerate(category_name) :
    category_dic[cate]=i


def init():
    def cleanDate(model):
        # 处理时间
        t=[time.strptime(a, "%Y-%m-%d %H:%M:%S") for a in model.Dates]
        model["Year"]=pd.Series([it.tm_year for it in t])
        model["Month"]=pd.Series([it.tm_mon for it in t])
        model["Day"]=pd.Series([it.tm_mday for it in t])
        model["Hour"]=pd.Series([it.tm_hour for it in t])
        model["Minute"]=pd.Series([it.tm_min for it in t])
        #星期转换(Monday=1 Sunday=7)
        model["DayOfWeek"]=pd.Series([dayofweektable[it] for it in model["DayOfWeek"]])

        return model

    train_raw=pd.read_csv("train.csv",header=0,encoding='utf-8')
    train_raw=cleanDate(train_raw)
    category=train_raw.Category.unique().tolist()
    #平均值
    imean=train_raw.groupby("Category").mean()
    # print(imean)
    # print(model_m)
    test_raw=pd.read_csv("test.csv",header=0,encoding='utf-8')
    test_raw=cleanDate(test_raw)



    def cleanTheData(model_raw):
        model_m=model_raw.shape[0]
        print("Reading Data Ok!")

        #model 是用来训练的
        # model=model_raw.loc[:,["Category"]].copy()
        model=model_raw.loc[:,["Year","Month","Day","Hour","Minute","DayOfWeek"]].copy()

        PdDistrict=model_raw.PdDistrict.unique().tolist()

        # print(model_raw.Address.unique())
        print(category)

        #Season 冬春夏秋(0123)
        model["Season"]=pd.Series(np.zeros(model_m),dtype='int')
        model.loc[(model.Month>=3)&(model.Month<=5),"Season"]=1
        model.loc[(model.Month>=6)&(model.Month<=8),"Season"]=2
        model.loc[(model.Month>=9)&(model.Month<=11),"Season"]=3

        print(model.Season.unique().tolist())

        # for cate in category:
            # print(imean.loc[cate,:])
            # model["Year2"+cate]=model["Year"]-imean.loc[cate,"Year"]
            # model["Month2"+cate]=model["Month"]-imean.loc[cate,"Month"]
            # model["Day2"+cate]=model["Day"]-imean.loc[cate,"Day"]
            # model["Hour2"+cate]=model["Hour"]-imean.loc[cate,"Hour"]
            # model["Minute2"+cate]=model["Minute"]-imean.loc[cate,"Minute"]
            # model["DayOfWeek2"+cate]=model["DayOfWeek"]-imean.loc[cate,"DayOfWeek"]


        ##开始处理地点
        for pdd in PdDistrict:
            model[pdd]=(model_raw.PdDistrict==pdd).astype(int)
            # print(np.sum(model[pd]))

        wayNames=["/"," AV"," ST"," Block"," DR"," WY"," BL"," LN"," RD"," BLVD"," HY"," CT"," PZ"," TR"]

        def findStr(srs,st):
            s=[]
            # print(type(srs))
            # print(type(st))
            for sr in srs:
                # print(sr)
                if sr.find(st)!=-1:
                    s.append(1)
                else:
                    s.append(0)
            ss=pd.Series(s)
            return ss

        for wayName in wayNames:
            model["AddHas_"+wayName]=findStr(model_raw.Address,wayName)
            print(wayName,np.sum(model["AddHas_"+wayName]))

            #定义犯罪距离
        for cate in category:
            MLonB=imean.loc[cate,"X"]
            MLatB=imean.loc[cate,"Y"]
            MLonA=model_raw.X
            MLatA=model_raw.Y
            D=np.sin(MLatA)*np.sin(MLatB)+ np.cos(MLatA)*np.cos(MLatB)*np.cos(MLonA)*np.cos(MLonB)
            D=(D-np.min(D))/(np.max(D)-np.min(D))
            model["Distant2"+cate]=D


        print(model.shape)
        return model

    train_y=[category_dic[t] for t in train_raw.Category]
    train=cleanTheData(train_raw)

    train.to_csv("train_x.data",index=False)
    pd.DataFrame(train_y).to_csv("train_y.data",index=False)
    train=train.values

    test=cleanTheData(test_raw)
    test.to_csv("test.data",index=False)
    test=test.values


def loadData():
    train=pd.read_csv("train_x.data")
    train=train.values
    train_y=pd.read_csv("train_y.data")
    train_y=train_y["Label"].tolist()
    test=pd.read_csv("test.data")
    test=test.values
    return train,train_y,test

# init()
train,train_y,test=loadData()


from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()

train_x, test_x, train_yy, test_yy = train_test_split(train, train_y, test_size=0.33, random_state=42)
print("Spliting OK!")
clf.fit(train_x,train_yy)
print("Training OK!!")
score=clf.score(test_x,test_yy)
test_y=clf.predict(test)
print(score)

file=open("the ans.txt")
# category_name=["Id"]+category_name
ans=pd.DataFrame(np.zeros((test.shape[0],len(category_name)),dtype=int),columns=category_name)

nTest_y=len(test_y)
for i in range(nTest_y):
    ans.iat[i,test_y[i]]=1
    file.write(test_y[i])

ans.to_csv("ans.txt",index_label="Id")

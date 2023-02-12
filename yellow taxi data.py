#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


os.chdir(r"C:\Users\admim\Downloads\pga25")


# In[3]:


os.listdir()


# In[4]:


taxi_data=pd.read_parquet("yellow_tripdata_2022-06.parquet")


# In[5]:


taxi_data.shape


# In[6]:


from sklearn.model_selection import train_test_split
taxi,_=train_test_split(taxi_data, test_size=.98,random_state=0)


# In[7]:


taxi.shape


# In[8]:


taxi.head()


# In[9]:


# here our target is total amoount


# ## EDA

# In[10]:


taxi["VendorID"].value_counts()


# In[11]:


def univariate_cat(data, x):
    missing=data[x].isnull().sum()
    unique_cnt=data[x].nunique()
    unique_cat=list(data[x].unique())
    
    f1=pd.DataFrame(data[x].value_counts())
    f1.rename(columns={x:"Count"}, inplace=True)
    f2=pd.DataFrame(data[x].value_counts(normalize=True))
    f2.rename(columns={x:"percentage"}, inplace=True)
    f2["percentage"]=(f2["percentage"]*100).round(2).astype(str)+"%"
    ff=pd.concat([f1,f2],axis=1)
    
    print(f"Total missing values : {missing}\n")
    print(f"Total count of unique categories: {unique_cnt}\n")
    print(f"Unique categories :\n{unique_cat}")
    print(f"value count and %\n",ff)
    sns.countplot(data=data,x=x)
    plt.show()


# In[12]:


univariate_cat(data=taxi,x="VendorID")


# In[13]:


taxi["VendorID"]=taxi["VendorID"].replace([6,5],"others")
taxi["VendorID"].value_counts()


# In[14]:


taxi.dtypes[taxi.dtypes=="object"]


# In[15]:


univariate_cat(data=taxi,x="store_and_fwd_flag")


# In[16]:


taxi["store_and_fwd_flag"]=taxi["store_and_fwd_flag"].replace(np.nan,"other")


# In[17]:


taxi["store_and_fwd_flag"].isnull().sum()


# In[18]:


taxi["RatecodeID"]=taxi["RatecodeID"].map({1:"standard_rate",
                                          2:"jfk",
                                         3:"Newark",
                                          4:"Nassau_or_westchester",
                                          5:"Negotiated_fare",
                                          6:"Groud_ride",
                                          99:"other"})


# In[19]:


# 1= Standard rate
# 2=JFK
# 3=Newark
# 4=Nassau or Westchester
# 5=Negotiated fare
# 6=Group ride


# In[20]:


univariate_cat(data=taxi,x="RatecodeID")


# In[21]:


taxi["RatecodeID"]=taxi["RatecodeID"].replace(np.nan,"other")


# In[22]:


taxi["RatecodeID"].value_counts(dropna=False)


# In[23]:


univariate_cat(data=taxi,x="payment_type")


# In[24]:


taxi["payment_type"]=taxi["payment_type"].map({1:"Credit_card",2:"Cash",3:"No_charge",
                                               4:"Dispute",0:"Unknown"})


# In[25]:


# 1= Credit card
# 2= Cash
# 3= No charge
# 4= Dispute
# 5= Unknown
# 6= Voided trip


# In[26]:


taxi["payment_type"].value_counts(dropna=False)


# In[27]:


taxi.head()


# In[ ]:





# In[28]:


def univariate_num(data ,x):
    missing=data[x].isnull().sum()
    min1=round(data[x].min(), 2)
    max1=round(data[x].max(), 2)
    mean=round(data[x].mean(), 2)
    var=round(data[x].var(), 2)
    std=round(data[x].std(),2)
    range1=round(max1-min1, 2)
    q1=round(data[x].quantile(.25), 2)
    q2=round(data[x].quantile(.5),2)
    q3=round(data[x].quantile(.75), 2)
    skew=round( data[x].skew(), 2)
    kurt=round(data[x].kurt(), 2)
    myvalue={"missing":missing, "min":min1, "max":max1, "mean":mean,
             "var":var,"std":std, "range":range1,"q1":q1, "q2":q2,"q3":q3,
            "skewness":skew, "kurtosis":kurt}
#     sns.histplot(data[x])
#     plt.show()
#     sns.boxplot(data=data, y=data[x])
#     plt.show()
    return myvalue


# In[29]:


univariate_num(data=taxi,x="passenger_count")


# In[30]:


univariate_num(data=taxi,x="trip_distance")


# In[31]:


sns.boxplot(data=taxi, y="trip_distance")


# In[32]:


taxi.head()


# In[33]:


univariate_num(data=taxi, x="PULocationID")


# In[35]:


univariate_cat(data=taxi,x="PULocationID")


# In[36]:


univariate_cat(data=taxi,x="DOLocationID")


# In[37]:


taxi.drop(columns=["PULocationID","DOLocationID"],inplace=True)


# In[38]:


univariate_num(data=taxi,x="fare_amount")


# In[39]:


taxi[taxi["fare_amount"]<=0].shape


# In[41]:


taxi[taxi["fare_amount"]<2.5].shape


# In[42]:


taxi.shape


# In[43]:


taxi1=taxi[taxi["fare_amount"]>=2.5]


# In[44]:


taxi1.shape


# In[45]:


sns.boxplot(data=taxi,y="fare_amount")


# In[47]:


taxi1[taxi1["fare_amount"]>30.75].shape


# In[50]:


taxi1["fare_amount"].describe(percentiles=[.75,.8,.9,.95,.96,.97,.98,.99])


# In[52]:


taxi1.dtypes[taxi1.dtypes!="object"].index


# In[ ]:


'extra','mta_tax','tip_amount','tolls_amount','improvement_surcharge','total_amount','congestion_surcharge', 'airport_fee'


# In[54]:


univariate_num(data=taxi1,x="extra")


# In[55]:


taxi1[taxi1["extra"]==0].shape


# In[56]:


'mta_tax','tip_amount','tolls_amount','improvement_surcharge','total_amount','congestion_surcharge', 'airport_fee'
univariate_num(data=taxi1,x="mta_tax")


# In[57]:


sns.boxplot(data=taxi1,y="mta_tax")


# In[58]:


'tip_amount','tolls_amount','improvement_surcharge','total_amount','congestion_surcharge', 'airport_fee'
univariate_num(data=taxi1,x="tip_amount")


# In[59]:


sns.boxplot(data=taxi1,y="tip_amount")


# In[60]:


'tolls_amount','improvement_surcharge','total_amount','congestion_surcharge', 'airport_fee'
univariate_num(data=taxi1,x="tolls_amount")


# In[61]:


sns.boxplot(data=taxi1,y="tolls_amount")


# In[63]:


'total_amount','congestion_surcharge', 'airport_fee'
univariate_num(data=taxi1,x="improvement_surcharge")


# In[64]:


sns.boxplot(data=taxi1,y="improvement_surcharge")


# In[65]:


taxi1[taxi1["improvement_surcharge"]==0].shape


# In[67]:


taxi1.drop(columns=["improvement_surcharge"], inplace=True)


# In[68]:


taxi1.head()


# In[69]:


'','congestion_surcharge', 'airport_fee'
univariate_num(data=taxi1,x="total_amount")


# In[70]:


sns.boxplot(data=taxi1,y="total_amount")


# In[71]:


'','', ''
univariate_num(data=taxi1,x="congestion_surcharge")


# In[72]:


univariate_num(data=taxi1,x="airport_fee")


# In[73]:


sns.boxplot(data=taxi1,y="airport_fee")


# In[74]:


taxi1[taxi1["airport_fee"]==0].shape


# ## feature engineering

# In[75]:


taxi1.columns


# In[76]:


taxi1["tpep_pickup_datetime"].min()


# In[77]:


taxi1["tpep_pickup_datetime"].max()


# In[78]:


# Date; month, weekend/weekday, public_holiday, time of pickup (early morning, morning, afternoon, evening, night, late night)
# two date : duration


# In[103]:


taxi1["month"]=taxi1["tpep_pickup_datetime"].dt.month
taxi1["year"]=taxi1["tpep_pickup_datetime"].dt.year
taxi1["week_of_day"]=taxi1["tpep_pickup_datetime"].dt.dayofweek
taxi1["day"]=taxi1["tpep_pickup_datetime"].dt.day
taxi1["hours"]=taxi1["tpep_pickup_datetime"].dt.hour
taxi1["minutes"]=taxi1["tpep_pickup_datetime"].dt.minute


# In[91]:


taxi1["week_of_day"].value_counts()


# In[90]:


taxi1=taxi1[taxi1["month"]==6]
taxi1=taxi1[taxi1["year"]==2022]


# In[93]:


taxi1["isweekend"]=np.where(taxi1["week_of_day"].isin([5,6]),1,0)


# In[95]:


taxi1["day"].value_counts()


# In[100]:


def day_cut(x):
    if x<=10:
        return "starting_of_month"
    elif x>10 and x<=20:
        return "mid_of_month"
    else:
        return "end_of_month"
taxi1["day_cat"]=taxi1["day"].apply(lambda a: day_cut(a))


# In[101]:


taxi1["day_cat"].value_counts()


# In[105]:


# taxi1["hours"].value_counts()


# In[ ]:


early mor:5-8
mor : 8-12
afternoon:12-16
even: 16-20
night: 20-24
mid night: 24-4


# In[106]:


def hour_category(x):
    if x>=4 and x<8:
        return "early_morning"
    elif x>=8 and x<12:
        return "morning"
    elif x>=12 and x<16:
        return "Afternoon"
    elif x>=16 and x<20:
        return "evening"
    elif x>=20 and x<=24:
        return "night"
    elif x>=0 and x<4:
        return "midNight"
    else:
        return "Unknown"
taxi1["hours_cat"]=taxi1["hours"].apply(lambda a: hour_category(a))


# In[107]:


taxi1["hours_cat"].value_counts()


# In[108]:


# taxi1["duration"]=(taxi1["tpep_dropoff_datetime"]-taxi1["tpep_pickup_datetime"])*24


# In[123]:


taxi1["duration"]=((taxi1["tpep_dropoff_datetime"]-taxi1["tpep_pickup_datetime"])/pd.Timedelta(minutes=1))


# In[124]:


taxi1["duration"].min()


# In[125]:


taxi1["duration"].max()


# In[126]:


sns.histplot(data=taxi1,x="duration")


# In[129]:


def duration_cat(x):
    if x<15:
        return "duration_bt_0_15m"
    elif x>=15 and x<30:
        return "duration_bt_0_30m"
    elif x>=30 and x<60:
        return "duration_bt_30_60m"
    elif x>=60 and x<120:
        return "duration_bt_60_120m"
    else:
        return "duration_morethan_120m"
taxi1["duration_cat"]=taxi1["duration"].apply(lambda y: duration_cat(y))


# In[130]:


taxi1["duration_cat"].value_counts()


# In[111]:


taxi1["airport_pick_up"]=np.where(taxi1["airport_fee"]==0,0,1)


# In[113]:


taxi1.head()


# In[114]:


def fare_cut(x):
    if x<=10:
        return "(0, 10]"
    elif x>10 and x<=20:
        return "(10, 20]"
    elif x>20 and x<=30:
        return "(20, 30]"
    elif x>30 and x<=40:
        return "(30, 40]"
    elif x>40 and x<=50:
        return "(40, 50]"
    elif x>50 and x<=60:
        return "(50, 60]"
    elif x>60 and x<=70:
        return "(60, 70]"
    elif x>70 and x<=80:
        return "(70, 80]"
    elif x>80 and x<=90:
        return "(80, 90]"
    elif x>90 and x<=100:
        return "(90, 100]"
    else:
        return "100+"
taxi1["fare_cat"]=taxi1["total_amount"].apply(lambda a: fare_cut(a))


# In[117]:


taxi1["fare_cat"].value_counts().plot(kind="bar")


# ## missing value

# In[131]:


taxi1.isnull().sum()


# In[133]:


temp0=taxi1[taxi1["passenger_count"].isnull()]


# In[134]:


temp0.isnull().sum()


# In[135]:


taxi2=taxi1.dropna()


# In[136]:


taxi2.isnull().sum()


# ## outliers

# In[137]:


taxi2.describe(percentiles=[.01,.02,.03,.04,.05,.25,.5,.75,.9,.95,.96,.97,.98,.99]).T


# In[138]:


taxi2.drop(columns=["tpep_pickup_datetime" ,"tpep_dropoff_datetime"], inplace=True)


# In[140]:


obj_var=taxi2.dtypes[taxi2.dtypes=="object"].index
num_var=taxi2.dtypes[taxi2.dtypes!="object"].index


# In[141]:


taxi_num=taxi2[num_var]
taxi_obj=taxi2[obj_var]


# In[145]:


def outliers(x):
    x=x.clip(upper=x.quantile(.99))
#    x=x.clip(lower=x.quantile(.01))
    return x
taxi_num=taxi_num.apply(lambda a: outliers(a))


# In[146]:


taxi_final=pd.concat([taxi_num,taxi_obj], axis=1)


# ## multicollinearity

# In[149]:


taxi_final.drop(columns=["month","year","fare_amount"],inplace=True)


# In[152]:


taxi_final.drop(columns=["weekend_weekday"],inplace=True)


# In[153]:


cr=taxi_final.corr()
cr=cr[abs(cr)>.7]
plt.figure(figsize=(11,6))
sns.heatmap(cr, annot=True, cmap="coolwarm")


# In[154]:


taxi_final.drop(columns=["airport_fee","week_of_day"],inplace=True)


# In[157]:


taxi_final.drop(columns=["fare_cat"], inplace=True)


# ## dummies creation

# In[158]:


final=pd.get_dummies(data=taxi_final, drop_first=True)


# In[159]:


final.head()


# In[160]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV


# In[166]:


y=final["total_amount"]
x=final.drop(columns=["total_amount","trip_distance"])


# In[167]:


x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=.3, random_state=0)


# In[168]:


rfr=RandomForestRegressor()
rfr.fit(x_train, y_train)


# In[169]:


print("Train_score",rfr.score(x_train,y_train))
print("Test_score",rfr.score(x_test,y_test))


# In[170]:


# # criterion: {"squared_error","absolute_error","poisson"}
# mae: absolute_error:L1 error
# mse: squared_error : L2 error


# In[ ]:





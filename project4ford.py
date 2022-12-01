#!/usr/bin/env python
# coding: utf-8

# In[1]:


cd H:\jupyter_wd


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as sts
import math
from pickle import dump


# In[3]:


ford = pd.read_csv('C://Users//HP//used-car-dataset-ford-and-mercedes//ford.csv')


# # ford EDA

# ford.shape

# In[4]:


ford.describe()


# In[5]:


ford.info()


# In[6]:


print(ford.duplicated().sum())


# In[7]:


ford = ford.drop_duplicates()
ford.shape


# In[8]:


df =ford[['year','price','mileage','tax','mpg','engineSize']]
sns.heatmap(df.corr(),cmap="YlGnBu",annot=True)


# In[9]:


plt.figure(figsize=(10,6))
sns.histplot(ford['price'])
plt.xlabel('Prize',fontsize=15)
plt.ylabel('count',fontsize=15)
plt.show()


# In[10]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.histplot(ford['mileage'])
plt.xlabel('mileage')
plt.ylabel('count')

plt.subplot(1,2,2)
sns.histplot(ford['mileage'])
plt.xlim(135000,326000)
plt.ylim(0,5)
plt.xlabel('mileage')
plt.ylabel('count')


# In[11]:


ford[ford['mileage'] > 160000]


# In[12]:


ford.sort_values('year',axis=0).head(5)


# In[13]:


ford = ford[ford['mileage'] < 160000]


# In[14]:


plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
sns.histplot(ford['mpg'])
plt.xlabel('mpg')
plt.ylabel('count')

plt.subplot(1,2,2)
sns.histplot(ford['mpg'])
plt.xlim(150,500)
plt.ylim(0,50)
plt.xlabel('mpg')
plt.ylabel('count')


# In[15]:


ford[ford['mpg']>100]


# In[16]:


plt.figure(figsize=(20,15))
ax=sns.countplot(x='model', data=ford,order=ford['model'].value_counts().sort_values(ascending=False).index)
for p in ax.patches:
   ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x(), p.get_height()))

plt.show()


# In[17]:


ford.shape


# In[18]:


ford = ford[(ford['model'] != ' Transit Tourneo') & (ford['model'] != ' Escort') & (ford['model'] != ' Ranger') & (ford['model'] != ' Streetka')]


# In[19]:


ford.shape


# In[20]:


plt.figure(figsize=(20,15))
ax=sns.countplot(x='model', data=ford,hue='transmission')
for p in ax.patches:
   ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x(), p.get_height()))

plt.show()


# In[21]:


plt.figure(figsize=(20,15))
ax=sns.countplot(x='model', data=ford,hue='fuelType')
for p in ax.patches:
   ax.annotate((p.get_height()), (p.get_x()+0.1, p.get_height()+3))

plt.show()


# In[22]:


models = ford['model'].unique()
fuel_T = ford['fuelType'].unique()
trans = ford['transmission'].unique()


# In[23]:


rows = []
for x in models:
  for y in fuel_T:
    for z in trans:
      max_Y = np.max(ford['year'][(ford['model'] == x) & (ford['fuelType'] == y) & (ford['transmission'] == z)]) 
      min_Y = np.min(ford['year'][(ford['model'] == x) & (ford['fuelType'] == y) & (ford['transmission'] == z)])
      if math.isnan(max_Y) or math.isnan(min_Y):
        continue
      rows.append([x,y,z,int(max_Y),int(min_Y)])
pd.DataFrame(rows,columns=["model","fuel","transmission","l_manufac", "f_manufac"]).head()  


# # ford Adavance analysis

# In[24]:


ford.head()


# In[25]:


# ln Transform price variable
ford['price'] = np.log(ford['price'])


# In[26]:


model_ohe = pd.get_dummies(ford.model)
model_ohe.head()


# In[27]:


transmission_ohe = pd.get_dummies(ford.transmission)
transmission_ohe.head()


# In[28]:


fuel_ohe = pd.get_dummies(ford.fuelType)
fuel_ohe.head()


# In[29]:


df = ford.copy(deep=True)


# In[30]:


df = df.drop(['model','transmission','fuelType'],axis = 1)


# In[31]:


df.shape


# In[32]:


df = pd.concat([df,model_ohe,transmission_ohe,fuel_ohe],axis=1)
df.head()


# In[33]:


get_ipython().system('pip install scikit-learn')


# In[34]:


#spliting data set
from sklearn.model_selection import train_test_split
x = df.drop(['price'],axis=1)
y = df[['price']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=4)


# In[35]:


x_train.head()


# In[36]:


#normalize data 

from sklearn.preprocessing import  MinMaxScaler
minmax = MinMaxScaler()
normalized_x_train = pd.DataFrame(minmax.fit_transform(x_train) ,columns = x_train.columns)

normalized_x_test = pd.DataFrame(minmax.transform(x_test) ,columns = x_test.columns)

# save the model
dump(minmax, open('ford_scaler.pkl', 'wb'))


# In[37]:


normalized_x_train.head()


# In[38]:


web_table = x_train.iloc[[1],:].copy()
web_table.columns = web_table.columns.str.replace(' ', '')
for x in web_table.columns:
    web_table[x]= 0
    
#save the table
web_table.to_pickle("ford_table.pkl")


# In[39]:


web_table


# In[40]:


# build a linear regression model
from sklearn.linear_model import LinearRegression
linreg = LinearRegression(fit_intercept=False)
linreg.fit(normalized_x_train,y_train,)

# save the model
dump(linreg, open('ford_model.pkl', 'wb'))


print ("iNTERCEPT : ",linreg.intercept_)
print ("CO-EFFICIENT : ",linreg.coef_)
print("\n")
y_pred = linreg.predict(normalized_x_test)

from sklearn.metrics import r2_score
from sklearn import metrics
print("R-Square Value",r2_score(y_test,y_pred))
print ("mean_absolute_error :",metrics.mean_absolute_error(y_test, y_pred))
print ("mean_squared_error : ",metrics.mean_squared_error(y_test, y_pred))
print ("root_mean_squared_error : ",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[41]:


# build a linear Ridge model
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
# create an array of alpha values
alpha_range = 10.**np.arange(-2, 3)
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define model
ridgeregcv = RidgeCV(alphas=np.arange(0, 1, 0.01), cv=cv,fit_intercept=False)
# fit model
ridgeregcv.fit(normalized_x_train, y_train)
# summarize chosen configuration
print('alpha: %f' % ridgeregcv.alpha_,"\n")


# predict method uses the best alpha value
y_pred = ridgeregcv.predict(normalized_x_test)

# calculate R^2 value, MAE, MSE, RMSE
print("R-Square Value",r2_score(y_test,y_pred))
print ("mean_absolute_error :",metrics.mean_absolute_error(y_test, y_pred))
print ("mean_squared_error : ",metrics.mean_squared_error(y_test, y_pred))
print ("root_mean_squared_error : ",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[42]:


from sklearn.linear_model import LassoCV

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define model
lassoregcv = LassoCV(alphas=np.arange(0.01, 1, 0.01), cv=cv, n_jobs=-1,fit_intercept=False)
# fit model
lassoregcv.fit(normalized_x_train,np.ravel(y_train))


# In[43]:


# summarize chosen configuration
print('alpha: %f' % lassoregcv.alpha_)
print(len(lassoregcv.coef_))


# In[44]:


# predict method uses the best alpha value
y_pred = lassoregcv.predict(normalized_x_test)
# calculate R^2 value, MAE, MSE, RMSE

from sklearn import metrics
print("R-Square Value",r2_score(y_test,y_pred))
print("\n")
print ("mean_absolute_error :",metrics.mean_absolute_error(y_test, y_pred))
print("\n")
print ("mean_squared_error : ",metrics.mean_squared_error(y_test, y_pred))
print("\n")
print ("root_mean_squared_error : ",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
0.8938132772483944


# In[45]:


from sklearn.linear_model import ElasticNetCV

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define model
ratios = np.arange(0, 1, 0.01)
alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1,0.0, 1.0, 10.0, 100.0]
elasticnet = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1,fit_intercept=False,tol=10,max_iter=10**5)

elasticnet = elasticnet.fit(normalized_x_train,y_train)

# summarize chosen configuration
print('alpha: %f' % elasticnet.alpha_)
print('l1_ratio_: %f' % elasticnet.l1_ratio_)


# In[46]:


# summarize chosen configuration
print('alpha: %f' % elasticnet.alpha_)
print('l1_ratio_: %f' % elasticnet.l1_ratio_)
print(len(elasticnet.coef_))


# In[47]:


#predict method uses the best alpha value
y_pred = elasticnet.predict(normalized_x_test)

from sklearn import metrics
print("R-Square Value",r2_score(y_test,y_pred))
print("\n")
print ("mean_absolute_error :",metrics.mean_absolute_error(y_test, y_pred))
print("\n")
print ("mean_squared_error : ",metrics.mean_squared_error(y_test, y_pred))
print("\n")
print ("root_mean_squared_error : ",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # further exploration on linear model

# In[48]:


y_test_pred = linreg.predict(normalized_x_test)
test_r2 = r2_score(y_test,y_test_pred)
test_mae = metrics.mean_absolute_error(y_test,y_test_pred)
test_mse = metrics.mean_squared_error(y_test,y_test_pred)
test_rmse = np.sqrt(metrics.mean_squared_error(y_test,y_test_pred))
string_score_test =(f"R^2 on test set: {test_r2:.2f} "+f"\nMAE on test set: {test_mae:.2f} "+
               f"\nMSE on test set: {test_mse:.2f}"+f"\nRMSE on test set: {test_rmse:.2f}")



y_train_pred = linreg.predict(normalized_x_train)
train_r2 = r2_score(y_train,y_train_pred)
train_mae = metrics.mean_absolute_error(y_train,y_train_pred)
train_mse = metrics.mean_squared_error(y_train,y_train_pred)
train_rmse = np.sqrt(metrics.mean_squared_error(y_train,y_train_pred))
string_score_train =(f"R^2 on train set: {train_r2:.2f}"+f"\nMAE on train set: {train_mae:.2f}"+
               f"\nMSE on train set: {train_mse:.2f}"+f"\nRMSE on train set: {train_rmse:.2f}")


# In[49]:


fig, ax = plt.subplots(figsize=(12, 12))
plt.scatter(y_train, y_train_pred)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="red")
plt.text(7, 11, string_score_train,fontsize=15)
plt.title("training set")
plt.ylabel("Model predictions")
plt.xlabel("Truths")
plt.xlim([6,13 ])
_ = plt.ylim([6, 13])


# In[50]:


fig, ax = plt.subplots(figsize=(12, 12))
plt.scatter(y_test, y_test_pred)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="red")
plt.text(7, 11, string_score_test,fontsize=15)
plt.title("test set")
plt.ylabel("Model predictions")
plt.xlabel("Truths")
plt.xlim([6,13 ])
_ = plt.ylim([6, 13])


# In[51]:


coef_table = pd.DataFrame({'variable':list(normalized_x_train.columns)}).copy()
coef_table.insert(len(coef_table.columns),"Coeffecient",linreg.coef_.transpose())
coef_table = coef_table.set_index('variable')
coef_table


# In[52]:


coef_table.plot(kind="barh", figsize=(9, 7))
plt.title("Ridge model, small regularization")
plt.axvline(x=0, color=".5")
plt.subplots_adjust(left=0.3)


# In[53]:


from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(linreg, normalized_x_train, ['year'])


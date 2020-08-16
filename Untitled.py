#!/usr/bin/env python
# coding: utf-8

# In[3]:


import warnings
warnings.filterwarnings('ignore')

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


cars = pd.read_csv(r'C:\Users\Home\Downloads\data.csv')
cars.head()
print(cars)


# In[10]:


cars.shape


# In[11]:


cars.describe()


# In[12]:


cars.info()


# In[13]:


#Splitting car name and company name
CompanyName = cars['CarName'].apply(lambda x : x.split(' ')[0])
cars.insert(3,"CompanyName",CompanyName)
cars.drop(['CarName'],axis=1,inplace=True)
cars.head()


# In[14]:


cars.CompanyName.unique()


# In[15]:


#fixing mistakes
cars.CompanyName = cars.CompanyName.str.lower()

def replace_name(a,b):
    cars.CompanyName.replace(a,b,inplace=True)

replace_name('maxda','mazda')
replace_name('porcshce','porsche')
replace_name('toyouta','toyota')
replace_name('vokswagen','volkswagen')
replace_name('vw','volkswagen')

cars.CompanyName.unique()


# In[16]:


cars.loc[cars.duplicated()]


# In[17]:


cars.columns


# In[19]:


plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('Car Price Distribution Plot')
sns.distplot(cars.price)

plt.show()


# In[25]:


plt.figure(figsize=(25, 6))

df = pd.DataFrame(cars.groupby(['CompanyName'])['price'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Company Name vs Average Price')
plt.show()


# In[26]:


cars['fueleconomy'] = (0.55 * cars['citympg']) + (0.45 * cars['highwaympg'])


# In[27]:


cars['price'] = cars['price'].astype('int')
temp = cars.copy()
table = temp.groupby(['CompanyName'])['price'].mean()
temp = temp.merge(table.reset_index(), how='left',on='CompanyName')
bins = [0,10000,20000,40000]
cars_bin=['Budget','Medium','Highend']
cars['carsrange'] = pd.cut(temp['price_y'],bins,right=False,labels=cars_bin)
cars.head()


# In[31]:


cars_lr = cars[['price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase',
                  'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower', 
                    'fueleconomy', 'carlength','carwidth', 'carsrange']]
cars_lr.head()
def dummies(x,df):
    temp = pd.get_dummies(df[x], drop_first = True)
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df
# Applying the function to the cars_lr

cars_lr = dummies('fueltype',cars_lr)
cars_lr = dummies('aspiration',cars_lr)
cars_lr = dummies('carbody',cars_lr)
cars_lr = dummies('drivewheel',cars_lr)
cars_lr = dummies('enginetype',cars_lr)
cars_lr = dummies('cylindernumber',cars_lr)
cars_lr = dummies('carsrange',cars_lr)


# In[32]:


cars_lr.head()


# In[33]:


from sklearn.model_selection import train_test_split

np.random.seed(0)
df_train, df_test = train_test_split(cars_lr, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[34]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','fueleconomy','carlength','carwidth','price']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[35]:


df_train.head()


# In[36]:


df_train.describe()


# In[39]:


y_train = df_train.pop('price')
X_train = df_train
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[41]:


lm = LinearRegression()
lm.fit(X_train,y_train)
rfe = RFE(lm, 10)
rfe = rfe.fit(X_train, y_train)


# In[42]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[43]:


X_train.columns[rfe.support_]


# In[44]:


X_train_rfe = X_train[X_train.columns[rfe.support_]]
X_train_rfe.head()


# In[45]:


def build_model(X,y):
    X = sm.add_constant(X) #Adding the constant
    lm = sm.OLS(y,X).fit() # fitting the model
    print(lm.summary()) # model summary
    return X
    
def checkVIF(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return(vif)


# In[46]:


X_train_new = build_model(X_train_rfe,y_train)


# In[47]:


X_train_new = X_train_rfe.drop(["twelve"], axis = 1)


# In[48]:


X_train_new = build_model(X_train_new,y_train)


# In[49]:


X_train_new = X_train_new.drop(["fueleconomy"], axis = 1)


# In[50]:


X_train_new = build_model(X_train_new,y_train)


# In[51]:


checkVIF(X_train_new)


# In[52]:


X_train_new = X_train_new.drop(["curbweight"], axis = 1)


# In[53]:


X_train_new = build_model(X_train_new,y_train)


# In[54]:


checkVIF(X_train_new)


# In[55]:


X_train_new = X_train_new.drop(["sedan"], axis = 1)


# In[56]:


X_train_new = build_model(X_train_new,y_train)


# In[57]:


checkVIF(X_train_new)


# In[58]:


X_train_new = X_train_new.drop(["wagon"], axis = 1)


# In[59]:


X_train_new = build_model(X_train_new,y_train)


# In[60]:


checkVIF(X_train_new)


# In[61]:


X_train_new = X_train_new.drop(["dohcv"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
checkVIF(X_train_new)


# In[62]:


lm = sm.OLS(y_train,X_train_new).fit()
y_train_price = lm.predict(X_train_new)


# In[63]:


fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)  


# In[64]:


num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','fueleconomy','carlength','carwidth','price']
df_test[num_vars] = scaler.fit_transform(df_test[num_vars])


# In[65]:


#Dividing into X and y
y_test = df_test.pop('price')
X_test = df_test


# In[66]:


X_train_new = X_train_new.drop('const',axis=1)
# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)


# In[67]:


y_pred = lm.predict(X_test_new)


# In[68]:


from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)


# In[69]:


#EVALUATION OF THE MODEL
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)  


# In[70]:


print(lm.summary())


# In[ ]:





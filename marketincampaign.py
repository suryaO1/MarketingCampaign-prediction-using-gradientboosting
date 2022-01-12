#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('marketing_campaign.csv')


# In[3]:


df.head()


# In[4]:


df.columns.value_counts()


# In[5]:


df.isnull().sum()


# In[6]:


df.Z_CostContact.isnull().values.any()


# In[7]:


m = df.Income.mean()


# In[8]:


df.Income = df.Income.fillna(m)


# In[9]:


df.Income.isnull().sum()


# In[10]:


df.isnull().sum()


# In[11]:


from sklearn.preprocessing import LabelEncoder
le_Education = LabelEncoder()
le_Marital_Status = LabelEncoder()


# In[12]:


df['education']=le_Education.fit_transform(df['Education'])
df['marital_status'] = le_Marital_Status.fit_transform(df['Marital_Status'])


# In[13]:


df.head()


# In[14]:


df.drop(['Education','Marital_Status','Dt_Customer'],axis=1,inplace=True)


# In[15]:


df.info()


# In[16]:


df.head()


# In[17]:


from sklearn.feature_selection import VarianceThreshold
vt=VarianceThreshold(threshold=0)
vt.fit(df)


# In[18]:


constant_columns = [column for column in df.columns
                   if column not in df.columns[vt.get_support()]]
print(len(constant_columns))


# In[19]:


for feature in constant_columns:
    print(feature)


# In[20]:


df.drop(['Z_CostContact','Z_Revenue'],axis=1,inplace=True)


# In[21]:


df.head()


# In[22]:


df['children'] = df['Kidhome']+df['Teenhome']


# In[23]:


df.head()


# In[24]:


df.drop(['Kidhome','Teenhome'],axis=1,inplace=True)


# In[25]:


df.columns


# In[26]:


df['acceptedcmp'] = df['AcceptedCmp3']+df['AcceptedCmp4']+df['AcceptedCmp5']+df['AcceptedCmp1']+df['AcceptedCmp2']


# In[27]:


df.head()


# In[28]:


df.drop(['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
       'AcceptedCmp2'],axis=1,inplace=True)


# In[29]:


df.head()


# In[30]:


df['Total_purchase']=df['MntWines']+df['MntFruits']+df['MntMeatProducts']+df['MntMeatProducts']+df['MntSweetProducts']+df['MntGoldProds']+df['MntFishProducts']


# In[31]:


df.head()


# In[32]:


df.shape


# In[33]:


df.describe()


# In[34]:


max = df['Total_purchase'].quantile(0.99)
max


# In[35]:


min = df['Total_purchase'].quantile(0.05)
min


# In[36]:


df2=df[(df.Total_purchase<max)&(df.Total_purchase>min)]
df2


# In[37]:


df2.shape


# In[38]:


x=df2.drop('Response',axis='columns')
y=df2[['Response']]


# In[39]:


df2.skew()


# In[40]:


corr_matrix=df2.corr()
corr_matrix


# In[41]:


plt.figure(figsize=(20,20))
hm = sns.heatmap(corr_matrix,annot=True,square=True)
hm


# In[42]:


x=df2.drop(['Response','MntWines', 'MntFruits',
       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds'],axis='columns')
y=df2['Response']


# In[43]:


x.columns


# In[44]:


df2.Response.unique()


# In[45]:


df2.Response.value_counts()


# In[46]:


from imblearn.over_sampling import SMOTE
oversampling = SMOTE(sampling_strategy='minority')
x,y=oversampling.fit_resample(x,y)


# In[47]:


x.shape


# In[48]:


y.shape


# In[49]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[58]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train),columns=['ID', 'Year_Birth', 'Income', 'Recency', 'NumDealsPurchases',
       'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
       'NumWebVisitsMonth', 'Complain', 'education', 'marital_status',
       'children', 'acceptedcmp', 'Total_purchase'])
x_train.head()


# In[59]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[60]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(criterion='gini')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[61]:


model.score(x_test,y_test)


# In[62]:


from sklearn.metrics import confusion_matrix
confusion_matrix= confusion_matrix(y_pred,y_test)
confusion_matrix


# In[65]:


from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(loss='deviance',learning_rate=0.1,n_estimators=100,subsample=1.0,criterion='mse',min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_depth=3)
model.fit(x_train,y_train)


# In[66]:


y_pred=model.predict(x_test)
model.score(x_test,y_test)


# In[67]:


from sklearn.metrics import confusion_matrix
confusion_matrix= confusion_matrix(y_pred,y_test)
confusion_matrix


# In[ ]:





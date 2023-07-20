#!/usr/bin/env python
# coding: utf-8

# # lets get our data ready

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


heart_ds=pd.read_csv("C:/Users/dread-miles/Documents/Data Sets/BootCamp/heart-disease.csv")


# In[5]:


heart_ds


# In[7]:


heart_ds.info()


# In[11]:


x=heart_ds.drop("target",axis=1)
y=heart_ds["target"]


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[13]:


x_train.shape,y_train.shape


# In[14]:


x.shape


# In[16]:


len(heart_ds)


# # turning our dataset into numerical

# In[21]:


car=pd.read_csv("C:/Users/dread-miles/Documents/Data Sets/BootCamp/car-sales-extended.csv")


# In[22]:


car.shape


# In[23]:


car.info()


# In[24]:


car.head()


# In[28]:


car["Colour"].value_counts()


# In[34]:


car.head()


# In[48]:


x=car.drop("Price",axis=1)
y=car["Price"]


# ## and know since some of our variables are string lets transform or change  their data type

# In[30]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# In[33]:


onehot=OneHotEncoder()


# In[43]:


categorical_type=["Make","Colour","Doors"]
transformer=ColumnTransformer(transformers=[
                             ("onehot",
                              onehot,
                              categorical_type)],
                              remainder="passthrough"
                             )


# In[49]:


transformed_x=transformer.fit_transform(x)


# In[53]:


pd.DataFrame(transformed_x)


# In[56]:


dummies=pd.get_dummies(car[["Make","Colour","Doors"]])
dummies


# # know lets fit our model

# In[60]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()


# In[61]:


x_train,x_test,y_train,y_test=train_test_split(transformed_x,
                                              y,
                                              test_size=0.2)


# In[64]:


model.fit(x_train,y_train)


# In[65]:


model.score(x_test,y_test)


# In[ ]:





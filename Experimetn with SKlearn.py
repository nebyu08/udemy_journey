#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np


# In[8]:


heart_ds=pd.read_csv("C:/Users/dread-miles/Documents/Data Sets/BootCamp/heart-disease.csv")
heart_ds.head()


# In[10]:


x=heart_ds.drop("target",axis=1)
y=heart_ds["target"]


# In[13]:


x.shape,y.shape


# In[17]:


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()

#parameters are the default hyperparameter
clf.get_params()


# In[19]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[21]:


clf.fit(x_train,y_train);


# In[23]:


#make prediction
y_pred=clf.predict(x_test)
y_pred


# In[28]:


#y_reshape=y_test.reshape(-1)


# In[29]:


clf.score(x_train,y_train)


# In[30]:


clf.score(x_test,y_test)


# In[32]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[33]:


print(classification_report(y_test,y_pred))


# In[34]:


confusion_matrix(y_test,y_pred)


# In[35]:


accuracy_score(y_pred,y_test)


# # time to improve the model

# In[40]:


np.random.seed(42)
for i in range(10,100,10):
    print(f"trying with {i} paramters")
    clf=RandomForestClassifier(n_estimators=i).fit(x_train,y_train)
    print(f"score is: {clf.score(x_test,y_test)*100:.2f}%")
    print("")


# In[42]:


#save the model 
import pickle
pickle.dump(clf,open("random_forest_model_pkl","wb"))


# In[44]:


loaded_model=pickle.load(open("random_forest_model_pkl","rb"))
loaded_model.score(x_test,y_test)


# In[ ]:





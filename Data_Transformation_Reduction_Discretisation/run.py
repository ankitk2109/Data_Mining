#!/usr/bin/env python
# coding: utf-8

# ### Q1

# In[4]:


import pandas as pd
import numpy as np
import sklearn 
SensorData_Path=r".\specs\SensorData_question1.csv"

df= pd.read_csv(SensorData_Path)


# In[5]:


df.head()


# ### Q1.1

# In[6]:


#Inserting a copy of column Input3 
df['Original Input3'] = df['Input3']


# In[7]:


#Inserting a copy of column Input12 
df['Original Input12'] = df['Input12']


# In[8]:


df.head()


# ### Q1.2 

# In[9]:


#Normalizing using Z-Score
#df[col_zscore] = (df[col] - df[col].mean())/df[col].std(ddof=0)


# In[10]:


df['Input3'] = (df['Input3'] - df['Input3'].mean())/ df['Input3'].std()


# In[11]:


df.head(20)


# ### Q1.3

# In[24]:


#MinMax Normalization. Scaling in range [0-1]
'''
Formula:
max = df[col].max()
min = df[col].min()
df[col] = (df[col] - min) / (max - min)
'''


# In[25]:


Input12_max = df['Input12'].max()
Input12_min = df['Input12'].min()


# In[26]:


Input12_max


# In[27]:


Input12_min


# In[28]:


df['Input12'] = (df['Input12'] - Input12_min )/(Input12_max - Input12_min)


# In[29]:


df.head()


# ### Q1.4

# In[30]:


#Generating a new column Average Input which is average from Input1 to Input12
df['Average Input'] = df.loc[:][['Input1','Input12']].mean(axis=1)


# In[31]:


df.head()


# ### Q1.5

# In[33]:


#Saving the newly generated dataframe to output file
export_csv= df.to_csv(r".\output\question1_out.csv", index=False, float_format = '%g')


# ### Q.2

# In[37]:


import numpy as np
import pandas as pd


# In[38]:


filePath=r".\specs\DNAData_question2.csv"
df= pd.read_csv(filePath, header=0)
df


# ### Q2.1

# In[39]:


# Reducing attributes using Principal Component Analysis.
from sklearn.decomposition import PCA
pca = PCA().fit(df)


# In[40]:


import matplotlib.pyplot as plt


# In[41]:


plt.figure(figsize=(10, 8), dpi=80)
plt.plot(pca.explained_variance_ratio_.cumsum(), 'ro')
plt.xlabel("Number of Components")
plt.ylabel("Variance(%)")
plt.title("Dataset Explained Variance")


# In[42]:


pca = PCA(0.95)
data = pca.fit_transform(df) #The dataset variable will store our new data set, now with 29 dimensions.


# In[43]:


print(pca.n_components_)


# In[44]:


print(data)


# ### Q2.2

# In[45]:


#Discretising through equal width
"""
Discretise the PCA-generated attribute subset into 10 bins, using bins of
equal width. For each component X that you discretise, generate a new
column in the original dataset named pcaX width. For example, the first
discretised principal component will correspond to a new column called
pca1 width.
"""


# In[46]:


from sklearn.preprocessing import KBinsDiscretizer 
df1 = pd.read_csv(filePath)
est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
data_transform = est.fit_transform(data)


# In[47]:


data_transform


# In[48]:


transformed_data= pd.DataFrame(data_transform)


# In[49]:


transformed_data


# In[52]:


for i in range(0,22):
    col_name = 'pca'+str(i)+'_width'
    df1[col_name] = transformed_data[i]
df1


# ### Q2.3

# In[53]:


#Discretising through equal frequency
est_freq = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
data_transform_eqFreq= est_freq.fit_transform(data)
data_transform_eqFreq


# In[54]:


transformed_data_eqFreq= pd.DataFrame(data_transform_eqFreq)


# In[55]:


for i in range(0,22):
    col_name = 'pca'+str(i)+'_freq'
    df1[col_name] = transformed_data_eqFreq[i]
df1


# ### Q2.4

# In[57]:


#Saving the file
export= df1.to_csv(r".\output\question2_out.csv", index=False)


# In[ ]:





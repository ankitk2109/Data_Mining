#!/usr/bin/env python
# coding: utf-8

# In[23]:


#importing Required Libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import preprocessing


# In[5]:


#Reading file and Storing it into the dataframe
path = r".\specs\question_1.csv"
df = pd.read_csv(path)
df


# In[8]:


kmeans = KMeans(n_clusters=3, random_state=0).fit(df)
#print(kmeans)
centroids = kmeans.cluster_centers_
#print(centroids)
#setting plot Size
plt.figure(dpi=150)
plt.xlabel('X Values') #X axis values
plt.ylabel('Y Values') #Y axis values
plt.title('K-Means Clustering') #Title
scatter = plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.legend(handles=scatter.legend_elements()[0], labels=scatter.legend_elements()[1])
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.savefig(r'.\output\question_1.pdf')


# In[9]:


'''
#Just checking with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=0).fit(df)
centroids = kmeans.cluster_centers_
plt.figure(dpi=150)
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('K-Means Clustering')
scatter = plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.legend(handles=scatter.legend_elements()[0], labels=scatter.legend_elements()[1])
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
'''


# In[10]:


#Saving newly created labeles into the column called 'cluster' and saving the file to 'question_1.csv'
df['cluster']=kmeans.labels_
df.to_csv(r'.\output\question_1.csv', index=False)


# ## Q2

# In[14]:


#Reading file and Storing it into the dataframe
path_2 = r".\specs\question_2.csv"
df_2 = pd.read_csv(path_2)
df_2 =df_2.drop(['NAME', 'MANUF', 'TYPE', 'RATING'], axis=1) #dropping unwanted columns
kmeans = KMeans(n_clusters=5, max_iter=5, n_init=100, random_state=0).fit(df_2) #Creating 5 clusters which will choose random centroids 100 times and for each time it will run 5 times for optimizing.
centroids = kmeans.cluster_centers_
print(centroids)
df_2['config1'] = kmeans.labels_ #Saving newly created labels to column 'config1'
df_2


# In[15]:


#Changing the max_itr value to 100
kmeans = KMeans(n_clusters=5, max_iter=100, n_init=100, random_state=0).fit(df_2)
centroids = kmeans.cluster_centers_
print(centroids)
df_2['config2'] = kmeans.labels_#Saving newly created labels to column 'config2'
df_2


# In[17]:


kmeans = KMeans(n_clusters=3, max_iter=5, n_init=100,random_state=0).fit(df_2)
centroids = kmeans.cluster_centers_
print(centroids)
df_2['config3'] = kmeans.labels_
df_2.to_csv(r'.\output\question_2.csv', index=False) #Saving the dataframe to 'question_2.csv' file 
df_2


# ## Q3

# In[20]:


#Reading file and Storing it into the dataframe
path_3=r".\specs\question_3.csv"
df_3 = pd.read_csv(path_3)
df_3 = df_3.drop(['ID'],axis=1) #Droping the ID column
df_3


# In[21]:


kmeans = KMeans(n_clusters=7, max_iter=5, n_init=100, random_state=0).fit(df_3) #Taking k =3
centroids = kmeans.cluster_centers_
print(centroids)
df_3['kmeans']=kmeans.labels_
df_3


# In[22]:


plt.figure(dpi=150)
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('K-Means Clustering')
scatter = plt.scatter(df_3['x'], df_3['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.legend( handles = scatter.legend_elements()[0], labels =set(kmeans.labels_) )
plt.savefig(r'.\output\question_3_1.pdf')


# In[24]:


min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df_3[['x','y']])
df_3[['x','y']] = pd.DataFrame(x_scaled, columns=['x','y'])
df_3


# In[25]:


from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.04, min_samples=4).fit(df_3[['x','y']])
plt.figure(dpi=150)
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('DBSCAN Clustering')
scatter = plt.scatter(df_3['x'], df_3['y'], c= dbscan.labels_.astype(float))
plt.legend( handles = scatter.legend_elements()[0], labels =scatter.legend_elements()[1] )
#plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.savefig(r'.\output\question_3_2.pdf')


# In[26]:


df_3['dbscan1'] = dbscan.labels_ #Saving the labels into dbscan1 column
dbscan = DBSCAN(eps=0.08, min_samples=4).fit(df_3[['x','y']])
plt.figure(dpi=150)
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('DBSCAN Clustering')
scatter = plt.scatter(df_3['x'], df_3['y'], c= dbscan.labels_.astype(float))
plt.legend( handles = scatter.legend_elements()[0], labels =scatter.legend_elements()[1])
plt.savefig(r'.\output\question_3_3.pdf')


# In[27]:


df_3['dbscan2'] = dbscan.labels_
df_3.to_csv(r'.\output\question_3.csv')
df_3


# In[ ]:





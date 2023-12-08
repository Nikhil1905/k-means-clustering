#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import all libraries

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans


# In[2]:


#The code below will build a 2D dataset with four blobs.

from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)


# In[3]:


#the code below will assist us in visualizing the dataset.

plt.scatter(X[:, 0], X[:, 1], s=20);
plt.show()


# In[4]:


# create a K – means object while specifying the number of clusters, train the model, and estimate as follows-

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)


# In[5]:


#Now, using the code below, we can plot and visualize the cluster’s centers as determined by the k-means Python estimator-

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=20, cmap='summer')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='blue', s=100, alpha=1);
plt.show()


# In[ ]:





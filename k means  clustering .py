#!/usr/bin/env python
# coding: utf-8

# In[1]:


#To begin, we will import the essential packages-
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans


# In[2]:


#load the digit dataset from sklearn and create an object out of it. Additionally, we can get the total number of rows and the total number of columns in this dataset by doing the following:

from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape


# In[3]:


#According to the result, this dataset has 1797 samples with 64 features.

# We may cluster the data in the same way that we did in Example 1 above.

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape


# In[4]:


# the output above indicates that K-means generated 10 clusters with 64 features.

fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
   axi.set(xticks=[], yticks=[])
   axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)


# In[6]:


from scipy.stats import mode
labels = np.zeros_like(clusters)
for i in range(10):
   mask = (clusters == i)
   labels[mask] = mode(digits.target[mask])[0]
    
# Following that, we can check the accuracy as follows:

from sklearn.metrics import accuracy_score
accuracy_score(digits.target, labels)


# In[ ]:





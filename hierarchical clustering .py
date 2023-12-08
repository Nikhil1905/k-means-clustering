#!/usr/bin/env python
# coding: utf-8

# # Problem Statements:
# The average retention rate in the insurance industry is 84%, with the top-performing agencies in the 93% - 95% range. Retaining customers is all about the long-term relationship you build. Offering a discount on the client’s current policy will ensure he/she buys a new product or renews the current policy. Studying clients' purchasing behavior to determine which products they're most likely to buy is essential. 
# The insurance company wants to analyze their customer’s behavior to strategies offers to increase customer loyalty.

# Objective: Maximize the Sales 
# 
# Constraints: Minimize the Customer Retention
# 
# Success Criteria: 
# 
# Business Success Criteria: Increase the Sales by 10% to 12% by targeting cross-selling opportunities on current customers.
# 
# ML Success Criteria: Achieve a Silhouette coefficient of at least 0.6
# 
# Economic Success Criteria: The insurance company will see an increase in revenues by at least 8% 

# In[37]:


import pandas as pd
import matplotlib.pyplot as plt
import sweetviz
from AutoClean import AutoClean
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering 
from sklearn import metrics
from clusteval import clusteval
import numpy as np


# In[2]:


df=pd.read_csv("C:\\Users\\nikhil\\Downloads\\Data Set\\AutoInsurance (2).csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.describe()


# In[ ]:


# Data Preprocessing

import sweetviz
my_report = sweetviz.analyze([df, "df"])

my_report.show_html('Report.html')


# In[6]:


df.columns


# In[7]:


df.drop(['Customer','Policy','Effective To Date','Location Code','Vehicle Size'], axis = 1, inplace = True)


# In[8]:


df


# In[12]:


from sklearn.preprocessing import LabelEncoder


# In[10]:


clean_pipeline = AutoClean(df, mode = 'manual', missing_num = 'auto',
                           outliers = 'winz', encode_categ = 'auto')


# In[11]:


df_clean = clean_pipeline.output
df_clean.head()


# In[13]:


data_categorial = df_clean.select_dtypes(include=["object"])
categories = list(data_categorial.columns)
categories


# In[14]:


lb = LabelEncoder()

for i in categories:
    df_clean[i] = lb.fit_transform(df_clean[i])


# In[15]:


df_clean.info()

cols = list(df_clean.columns)
print(cols)


# In[38]:


from sklearn.preprocessing import StandardScaler


# In[80]:


from sklearn.preprocessing import RobustScaler


# In[94]:


pipe1 = make_pipeline(StandardScaler())


# In[95]:


df_pipelined = pd.DataFrame(pipe1.fit_transform(df_clean), columns = cols, index = df_clean.index)
df_pipelined.head()


# In[96]:


df_clean


# In[97]:


df_pipelined.describe()


# In[98]:


######### Model Building #########
# # CLUSTERING MODEL BUILDING


# In[99]:


plt.figure(1, figsize = (16, 8))
tree_plot = dendrogram(linkage(df_pipelined, method  = "complete"))

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Euclidean distance')
plt.show()


# In[100]:


hc1 = AgglomerativeClustering(n_clusters = 30, affinity = 'euclidean', linkage = 'complete')

y_hc1 = hc1.fit_predict(df_pipelined)
y_hc1


# In[101]:


hc1.labels_ 


# In[102]:


cluster_labels = pd.Series(hc1.labels_) 


# In[103]:


df_clust = pd.concat([cluster_labels, df_clean], axis = 1) 

df_clust.head()


# In[104]:


df_clust.columns
df_clust = df_clust.rename(columns = {0: 'cluster'})
df_clust.head()


# In[105]:


# # Clusters Evaluation

# **Silhouette coefficient:**  
# Silhouette coefficient is a Metric, which is used for calculating 
# goodness of the clustering technique, and the value ranges between (-1 to +1).
# It tells how similar an object is to its own cluster (cohesion) compared to 
# other clusters (separation).
# A score of 1 denotes the best meaning that the data point is very compact 
# within the cluster to which it belongs and far away from the other clusters.
# Values near 0 denote overlapping clusters.

# from sklearn import metrics
metrics.silhouette_score(df_pipelined, cluster_labels)


# In[106]:


metrics.calinski_harabasz_score(df_pipelined, cluster_labels)


# In[107]:


metrics.davies_bouldin_score(df_pipelined, cluster_labels)


# In[108]:


ce = clusteval(evaluate = 'silhouette')

df_array = np.array(df_pipelined)

# Fit
ce.fit(df_array)

# Plot
ce.plot()


# In[109]:


hc_2clust = AgglomerativeClustering(n_clusters = 12, affinity = 'euclidean', linkage = 'ward')

y_hc_2clust = hc_2clust.fit_predict(df_pipelined)
y_hc_2clust


# In[110]:


hc_2clust.labels_


# In[111]:


cluster_labels = pd.Series(hc_2clust.labels_) 


# In[112]:


df_clust = pd.concat([cluster_labels, df_clean], axis = 1) 

df_clust.head()


# In[113]:


metrics.silhouette_score(df_pipelined, cluster_labels)


# In[114]:


hc_3clust = AgglomerativeClustering(n_clusters = 11, affinity = 'euclidean', linkage = 'ward')

y_hc_3clust = hc_3clust.fit_predict(df_pipelined)
y_hc_3clust


# In[115]:


hc_3clust.labels_


# In[116]:


cluster_labels = pd.Series(hc_3clust.labels_) 


# In[117]:


df_clust = pd.concat([cluster_labels, df_clean], axis = 1) 

df_clust.head()


# In[118]:


metrics.silhouette_score(df_pipelined, cluster_labels)


# In[ ]:





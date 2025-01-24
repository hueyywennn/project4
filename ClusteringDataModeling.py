#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import unique
from numpy import where
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances

from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error


# In[2]:


# load data from csv into data frame and check header
dataset_url = 'https://raw.githubusercontent.com/usman2155904/WQD7001/main/CC%20GENERAL.csv'
dfRaw = pd.read_csv(dataset_url)
dfRaw.head()


# In[3]:


#rename variables and display header
dfUpdated = dfRaw.rename(columns = {'CUST_ID':'custID',
                         'BALANCE':'remainBal', 
                         'BALANCE_FREQUENCY':'balF', 
                         'PURCHASES':'purDone', 
                         'ONEOFF_PURCHASES':'oneOffPur', 
                         'INSTALLMENTS_PURCHASES':'insPur', 
                         'CASH_ADVANCE':'cashAdv', 
                         'PURCHASES_FREQUENCY':'purF', 
                         'ONEOFF_PURCHASES_FREQUENCY':'oneOffPurF', 
                         'PURCHASES_INSTALLMENTS_FREQUENCY': 'insPurF', 
                         'CASH_ADVANCE_FREQUENCY':'cashAdvF', 
                         'CASH_ADVANCE_TRX':'cashAdvTRX', 
                         'PURCHASES_TRX':'purTRX', 
                         'CREDIT_LIMIT':'creditLim', 
                         'PAYMENTS':'pymtDone', 
                         'MINIMUM_PAYMENTS':'minPymt', 
                         'PRC_FULL_PAYMENT':'fullPymtPCT', 
                         'TENURE':'tenure'})
dfUpdated.head()


# In[4]:


#calculate missing and null values ratio for all attributes
mnv_ratios = [ratio for ratio in (dfUpdated.isna().sum() / len(dfUpdated))] 
print([pair for pair in list(zip(dfUpdated.columns, mnv_ratios)) if pair[1] > 0])


# In[5]:


#BEFORE
#count null values of of all attributes
dfUpdated.isnull().sum()


# In[6]:


#AFTER
#fill mean values for null value of attributes and count again
dfUpdated['creditLim'].fillna(dfUpdated['creditLim'].mean(), inplace = True)
dfUpdated['minPymt'].fillna(dfUpdated['minPymt'].mean(), inplace = True)
dfUpdated.isnull().sum()


# In[7]:


#round off all attribute values of float data type in 4dp
dfNew = dfUpdated.round(4)

#remove custID attribute
dfNew.drop('custID', axis = 1, inplace = True)
dfNew.head(5)


# In[8]:


dfModel = dfNew.drop(['purF', 'oneOffPurF', 'insPurF', 'cashAdvF', ], axis=1)
dfModel.head()


# ##### A. Data Normalization

# In[9]:


# variables have different range of values 
# so first perform data scaling using standard scaler 
scaler = StandardScaler() 
scaled_df = scaler.fit_transform(dfModel) 

# Normalizing the scaled data 
normalizedDF = normalize(scaled_df) 
  
# Converting the numpy array into a pandas DataFrame 
normalizedDF = pd.DataFrame(normalizedDF) 

# view normalized data header
normalizedDF.head()


# ##### B. Feature selection

# In[10]:


normalizedDF.corr()


# In[11]:


#correlation in heatmap
fig = px.imshow(normalizedDF.corr(),  
                title = "<b>Pairwise Correlation Heatmap Plot</b>",
                text_auto = ".2f", 
                color_continuous_scale = px.colors.sequential.RdBu,
                height = 800,
                width = 800)
fig.update_layout(title_font_family="Times New Roman", 
                  title_font_color="maroon", 
                  font_family = "Arial", 
                  font_color = "navy")
fig.update_xaxes(side = "top")

fig.show()


# In[12]:


normalized_df = normalizedDF.drop([8, 9, 10], axis = 1)
normalized_df.head()


# ##### C. PCA

# In[13]:


# Using PCA for dimension reduction
# It transforms the larger number of variables to smaller group containing most of information
pca = PCA(n_components = 2) 
dfPCA = pca.fit_transform(normalized_df) 
dfPCA = pd.DataFrame(dfPCA) 
dfPCA.columns = ['PCA1', 'PCA2'] 
dfPCA


# ##### (i) Elbow Method

# In[14]:


# Elbow Method
distortions = []
K = range(2, 15)
for k in K:
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(dfPCA)
    distortions.append(kmeans.inertia_)

distortions


# In[15]:


plt.figure()
plt.plot(K, distortions, 'bx-')
plt.grid(True)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Distortions")
plt.title('The Elbow Method showing the optimal k')
plt.show()


# ##### (ii) Silhouette Score Method

# In[16]:


silhouette = []

K = range(2, 15)
for k in K:
    kmeans2 = KMeans(n_clusters = k, random_state = 100)
    predict = kmeans2.fit_predict(dfPCA)

    score = silhouette_score(dfPCA, predict, random_state = 100)
    silhouette.append(score)
    
silhouette


# In[17]:


plt.figure()
plt.plot(K, silhouette, marker = 'o')
plt.grid(True)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title('The Silhouette Method showing the optimal k')
plt.show()


# ##### (iii) Davies Bouldin Method

# In[18]:


davies_bouldin = []

K = range(2, 15)
for k in K:
    kmeans3 = KMeans(n_clusters = k)
    predict = kmeans3.fit_predict(dfPCA)

    score = davies_bouldin_score(dfPCA, predict)
    davies_bouldin.append(score)
    
davies_bouldin


# In[19]:


plt.figure()
plt.plot(K, davies_bouldin, marker = 'o')
plt.grid(True)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Davies Bouldin score")
plt.title('The Davies Bouldin Method showing the optimal k')
plt.show()


# ##### (iv) Calinski Harabasz Method

# In[20]:


calinski_harabasz = []

K = range(2, 15)
for k in K:
    kmeans4 = KMeans(n_clusters = k)
    predict = kmeans4.fit_predict(dfPCA)

    score = calinski_harabasz_score(dfPCA, predict)
    calinski_harabasz.append(score)
    
calinski_harabasz


# In[21]:


plt.figure()
plt.plot(K, calinski_harabasz, marker = 'o')
plt.grid(True)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Calinski Harabasz score")
plt.title('The Calinski Harabasz Method showing the optimal k')
plt.show()


# ##### clustering without PCA

# In[22]:


kmean = KMeans(n_clusters = 3, init = 'random', random_state = 1, n_init = 10)
results = kmean.fit_predict(normalized_df)

labels = kmean.labels_
clusters = pd.concat([normalized_df, pd.DataFrame({'cluster':labels})], axis = 1)

print("K-means inertia: ")
print(kmean.inertia_)
print("\n")
print("K-means cluster centers: ")
print(kmean.cluster_centers_)
print("\n")

print("Silhouette Score: ", silhouette_score(normalized_df, results))
print("Calinski Harabasz Score: ", calinski_harabasz_score(normalized_df, results))
print("Davies Bouldin Score: ", davies_bouldin_score(normalized_df, results))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax = sns.scatterplot(data = normalized_df, palette = "Accent")

plt.legend()
plt.show()


# In[23]:


#linkage : {‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
aggModel = AgglomerativeClustering(n_clusters = 3, 
                                   affinity = 'euclidean', 
                                   linkage = 'ward')
results = aggModel.fit_predict(normalized_df)

labels = aggModel.labels_
clusters = pd.concat([normalized_df, pd.DataFrame({'cluster':labels})], axis = 1)

print("Silhouette Score: ", silhouette_score(normalized_df, results))
print("Calinski Harabasz Score: ", calinski_harabasz_score(normalized_df, results))
print("Davies Bouldin Score: ", davies_bouldin_score(normalized_df, results))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax1 = sns.scatterplot(data = normalized_df, palette = "tab20")
plt.show()


# In[24]:


gmmModel = GaussianMixture(n_components = 3)

labels = gmmModel.fit_predict(normalized_df)
clusters = pd.concat([normalized_df, pd.DataFrame({'cluster':labels})], axis = 1)

print("Silhouette Score: ", silhouette_score(normalized_df, labels))
print("Calinski Harabasz Score: ", calinski_harabasz_score(normalized_df, labels))
print("Davies Bouldin Score: ", davies_bouldin_score(normalized_df, labels))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax2 = sns.scatterplot(data = normalized_df, palette = "Dark2")
plt.show()


# ##### Applying Modelling Techniques

# ##### PCA with feature selection

# In[25]:


pca = PCA(n_components = 2)
dfPCA = pca.fit_transform(normalized_df)

dfPCA = pd.DataFrame(dfPCA)
dfPCA.columns = ['PCA1', 'PCA2']
dfPCA


# In[26]:


dfPCACluster = pd.concat([dfPCA, pd.DataFrame({'cluster':labels})], axis = 1)
dfPCACluster


# ##### PCA without feature selection

# In[27]:


#without drop
# Using PCA for dimension reduction
# It transforms the larger number of variables to smaller group containing most of information
pca = PCA(n_components = 2) 
dfPCA = pca.fit_transform(normalizedDF) 
dfPCA2 = pd.DataFrame(dfPCA) 
dfPCA2.columns = ['PCA1', 'PCA2'] 
dfPCA2


# In[28]:


dfPCACluster2 = pd.concat([dfPCA2, pd.DataFrame({'cluster':labels})], axis = 1)
dfPCACluster2


# ##### (i) K-Means Clustering

# ##### when k = 2, PCA with feature selection

# In[29]:


kmean = KMeans(n_clusters = 2, init = 'random', random_state = 42, n_init = 10)
results = kmean.fit_predict(dfPCACluster)

labels = kmean.labels_
clusters = pd.concat([dfNew, pd.DataFrame({'cluster':labels})], axis = 1)

print("K-means inertia: ")
print(kmean.inertia_)
print("\n")
print("K-means cluster centers: ")
print(kmean.cluster_centers_)
print("\n")

print("Silhouette Score: ", silhouette_score(dfPCACluster, results))
print("Calinski Harabasz Score: ", calinski_harabasz_score(dfPCACluster, results))
print("Davies Bouldin Score: ", davies_bouldin_score(dfPCACluster, results))
print("Rand index: ", metrics.rand_score(dfPCACluster['cluster'], results))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax = sns.scatterplot(dfPCACluster['PCA1'], dfPCACluster['PCA2'], x = "PCA1", y = "PCA2", hue = labels, c = labels, palette = "Accent")
plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1], marker = "X", c = "r", s = 80, label = "centroids")
plt.legend()
plt.show()


# ##### when k = 2, PCA without feature selection

# In[30]:


kmean = KMeans(n_clusters = 2, init = 'random', random_state = 42, n_init = 10)
results = kmean.fit_predict(dfPCACluster2)

labels = kmean.labels_
clusters = pd.concat([dfNew, pd.DataFrame({'cluster':labels})], axis = 1)

print("K-means inertia: ")
print(kmean.inertia_)
print("\n")
print("K-means cluster centers: ")
print(kmean.cluster_centers_)
print("\n")

print("Silhouette Score: ", silhouette_score(dfPCACluster2, results))
print("Calinski Harabasz Score: ", calinski_harabasz_score(dfPCACluster2, results))
print("Davies Bouldin Score: ", davies_bouldin_score(dfPCACluster2, results))
print("Rand index: ", metrics.rand_score(dfPCACluster2['cluster'], results))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax = sns.scatterplot(dfPCACluster2['PCA1'], dfPCACluster2['PCA2'], x = "PCA1", y = "PCA2", hue = labels, c = labels, palette = "Accent")
plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1], marker = "X", c = "r", s = 80, label = "centroids")
plt.legend()
plt.show()


# ##### when k = 3, PCA with feature selection

# In[31]:


kmean = KMeans(n_clusters = 3, init = 'random', random_state = 42, n_init = 10)
results = kmean.fit_predict(dfPCACluster)

labels = kmean.labels_
clusters = pd.concat([dfNew, pd.DataFrame({'cluster':labels})], axis = 1)

print("K-means inertia: ")
print(kmean.inertia_)
print("\n")
print("K-means cluster centers: ")
print(kmean.cluster_centers_)
print("\n")

print("Silhouette Score: ", silhouette_score(dfPCACluster, results))
print("Calinski Harabasz Score: ", calinski_harabasz_score(dfPCACluster, results))
print("Davies Bouldin Score: ", davies_bouldin_score(dfPCACluster, results))
print("Rand index: ", metrics.rand_score(dfPCACluster['cluster'], results))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax = sns.scatterplot(dfPCACluster['PCA1'], dfPCACluster['PCA2'], x = "PCA1", y = "PCA2", hue = labels, c = labels, palette = "Accent")
plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1], marker = "X", c = "r", s = 80, label = "centroids")
plt.legend()
plt.show()


# ##### when k = 3, PCA without feature selection

# In[32]:


kmean = KMeans(n_clusters = 3, init = 'random', random_state = 42, n_init = 10)
results = kmean.fit_predict(dfPCACluster2)

labels = kmean.labels_
clusters = pd.concat([dfNew, pd.DataFrame({'cluster':labels})], axis = 1)

print("K-means inertia: ")
print(kmean.inertia_)
print("\n")
print("K-means cluster centers: ")
print(kmean.cluster_centers_)
print("\n")

print("Silhouette Score: ", silhouette_score(dfPCACluster2, results))
print("Calinski Harabasz Score: ", calinski_harabasz_score(dfPCACluster2, results))
print("Davies Bouldin Score: ", davies_bouldin_score(dfPCACluster2, results))
print("Rand index: ", metrics.rand_score(dfPCACluster2['cluster'], results))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax = sns.scatterplot(dfPCACluster2['PCA1'], dfPCACluster2['PCA2'], x = "PCA1", y = "PCA2", hue = labels, c = labels, palette = "Accent")
plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1], marker = "X", c = "r", s = 80, label = "centroids")
plt.legend()
plt.show()


# ##### when k = 4, PCA with feature selection

# In[33]:


kmean = KMeans(n_clusters = 4, init = 'random', random_state = 0, n_init = 500, max_iter = 200)
results = kmean.fit_predict(dfPCACluster)

labels = kmean.labels_
clusters = pd.concat([dfNew, pd.DataFrame({'cluster':labels})], axis = 1)

print("K-means inertia: ")
print(kmean.inertia_)
print("\n")
print("K-means cluster centers: ")
print(kmean.cluster_centers_)
print("\n")

print("Silhouette Score: ", silhouette_score(dfPCACluster,results))
print("Calinski Harabasz Score: ", calinski_harabasz_score(dfPCACluster, results))
print("Davies Bouldin Score: ", davies_bouldin_score(dfPCACluster, results))
print("Rand index: ", metrics.rand_score(dfPCACluster['cluster'], results))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax1 = sns.scatterplot(dfPCACluster['PCA1'], dfPCACluster['PCA2'], x = "PCA1", y = "PCA2", hue = labels, c = labels, palette = "Accent")
plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1], marker = "X", c = "r", s = 80, label = "centroids")
plt.legend()
plt.show()


# ##### when k = 4, PCA without feature selection

# In[34]:


kmean = KMeans(n_clusters = 4, init = 'random', random_state = 0, n_init = 500, max_iter = 200)
results = kmean.fit_predict(dfPCACluster2)

labels = kmean.labels_
clusters = pd.concat([dfNew, pd.DataFrame({'cluster':labels})], axis = 1)

print("K-means inertia: ")
print(kmean.inertia_)
print("\n")
print("K-means cluster centers: ")
print(kmean.cluster_centers_)
print("\n")

print("Silhouette Score: ", silhouette_score(dfPCACluster2, results))
print("Calinski Harabasz Score: ", calinski_harabasz_score(dfPCACluster2, results))
print("Davies Bouldin Score: ", davies_bouldin_score(dfPCACluster2, results))
print("Rand index: ", metrics.rand_score(dfPCACluster2['cluster'], results))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax1 = sns.scatterplot(dfPCACluster2['PCA1'], dfPCACluster2['PCA2'], x = "PCA1", y = "PCA2", hue = labels, c = labels, palette = "Accent")
plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1], marker = "X", c = "r", s = 80, label = "centroids")
plt.legend()
plt.show()


# ##### (ii) Agglomerative Clustering 

# ##### when no clusters is defined, PCA with feature selection

# In[35]:


#linkage : {‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
aggModel = AgglomerativeClustering(affinity = 'euclidean', 
                                   linkage = 'ward')
results = aggModel.fit_predict(dfPCACluster)

labelsNoCluster = aggModel.labels_
clusters = pd.concat([dfNew, pd.DataFrame({'cluster':labels})], axis = 1)

print("Silhouette Score: ", silhouette_score(dfPCACluster, results))
print("Calinski Harabasz Score: ", calinski_harabasz_score(dfPCACluster, results))
print("Davies Bouldin Score: ", davies_bouldin_score(dfPCACluster, results))
print("Rand index: ", metrics.rand_score(dfPCACluster['cluster'], results))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax2 = sns.scatterplot(dfPCACluster['PCA1'], dfPCACluster['PCA2'], x = "PCA1", y = "PCA2", hue = labelsNoCluster,  palette = "tab20")
plt.show()


# ##### when no clusters is defined, PCA without feature selection

# In[36]:


#linkage : {‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
aggModel = AgglomerativeClustering(affinity = 'euclidean', 
                                   linkage = 'ward')
results = aggModel.fit_predict(dfPCACluster2)

labelsNoCluster = aggModel.labels_
clusters = pd.concat([dfNew, pd.DataFrame({'cluster':labels})], axis = 1)

print("Silhouette Score: ", silhouette_score(dfPCACluster2, results))
print("Calinski Harabasz Score: ", calinski_harabasz_score(dfPCACluster2, results))
print("Davies Bouldin Score: ", davies_bouldin_score(dfPCACluster2, results))
print("Rand index: ", metrics.rand_score(dfPCACluster2['cluster'], results))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax2 = sns.scatterplot(dfPCACluster2['PCA1'], dfPCACluster2['PCA2'], x = "PCA1", y = "PCA2", hue = labelsNoCluster,  palette = "tab20")
plt.show()


# ##### when clusters = 3, PCA with feature selection

# In[37]:


#linkage : {‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
aggModel = AgglomerativeClustering(n_clusters = 3, 
                                   affinity = 'euclidean', 
                                   linkage = 'ward')
results = aggModel.fit_predict(dfPCACluster)

labels = aggModel.labels_
clusters = pd.concat([dfNew, pd.DataFrame({'cluster':labels})], axis = 1)

print("Silhouette Score: ", silhouette_score(dfPCACluster, results))
print("Calinski Harabasz Score: ", calinski_harabasz_score(dfPCACluster, results))
print("Davies Bouldin Score: ", davies_bouldin_score(dfPCACluster, results))
print("Rand index: ", metrics.rand_score(dfPCACluster['cluster'], results))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax3 = sns.scatterplot(dfPCACluster['PCA1'], dfPCACluster['PCA2'], x = "PCA1", y = "PCA2", hue = labels,  palette = "tab20")
plt.show()


# ##### when clusters = 3, PCA without feature selection

# In[38]:


#linkage : {‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
aggModel = AgglomerativeClustering(n_clusters = 3, 
                                   affinity = 'euclidean', 
                                   linkage = 'ward')
results = aggModel.fit_predict(dfPCACluster2)

labels = aggModel.labels_
clusters = pd.concat([dfNew, pd.DataFrame({'cluster':labels})], axis = 1)

print("Silhouette Score: ", silhouette_score(dfPCACluster2, results))
print("Calinski Harabasz Score: ", calinski_harabasz_score(dfPCACluster2, results))
print("Davies Bouldin Score: ", davies_bouldin_score(dfPCACluster2, results))
print("Rand index: ", metrics.rand_score(dfPCACluster2['cluster'], results))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax3 = sns.scatterplot(dfPCACluster2['PCA1'], dfPCACluster2['PCA2'], x = "PCA1", y = "PCA2", hue = labels,  palette = "tab20")
plt.show()


# ##### when clusters = 4, PCA with feature selection

# In[39]:


#linkage : {‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
aggModel = AgglomerativeClustering(n_clusters = 4, 
                                   affinity = 'euclidean', 
                                   linkage = 'ward')
results = aggModel.fit_predict(dfPCACluster)

labels = aggModel.labels_
clusters = pd.concat([dfNew, pd.DataFrame({'cluster':labels})], axis = 1)

print("Silhouette Score: ", silhouette_score(dfPCACluster, results))
print("Calinski Harabasz Score: ", calinski_harabasz_score(dfPCACluster, results))
print("Davies Bouldin Score: ", davies_bouldin_score(dfPCACluster, results))
print("Rand index: ", metrics.rand_score(dfPCACluster['cluster'], results))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax3 = sns.scatterplot(dfPCACluster['PCA1'], dfPCACluster['PCA2'], x = "PCA1", y = "PCA2", hue = labels,  palette = "tab20")
plt.show()


# ##### when clusters = 4, PCA without feature selection

# In[40]:


#linkage : {‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
aggModel = AgglomerativeClustering(n_clusters = 4, 
                                   affinity = 'euclidean', 
                                   linkage = 'ward')
results = aggModel.fit_predict(dfPCACluster2)

labels = aggModel.labels_
clusters = pd.concat([dfNew, pd.DataFrame({'cluster':labels})], axis = 1)

print("Silhouette Score: ", silhouette_score(dfPCACluster2, results))
print("Calinski Harabasz Score: ", calinski_harabasz_score(dfPCACluster2, results))
print("Davies Bouldin Score: ", davies_bouldin_score(dfPCACluster2, results))
print("Rand index: ", metrics.rand_score(dfPCACluster2['cluster'], results))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax3 = sns.scatterplot(dfPCACluster2['PCA1'], dfPCACluster2['PCA2'], x = "PCA1", y = "PCA2", hue = labels,  palette = "tab20")
plt.show()


# ##### (iii) Gaussian Mixture Clustering 

# ##### when components = 2, PCA with feature selection

# In[41]:


gm = GaussianMixture(n_components = 2, random_state = 42)

labels = gm.fit_predict(dfPCACluster)
clusters = pd.concat([dfNew, pd.DataFrame({'cluster':labels})], axis = 1)

print("Gaussian Mixture converged log-likelihood: ")
print(gm.lower_bound_)
print("\n")
print("Number of iterations required for log-likelihood to converge: ")
print(gm.n_iter_)
print("\n")

print("Silhouette Score: ", silhouette_score(dfPCACluster, labels))
print("Calinski Harabasz Score: ", calinski_harabasz_score(dfPCACluster, labels))
print("Davies Bouldin Score: ", davies_bouldin_score(dfPCACluster, labels))
print("Rand index: ", metrics.rand_score(dfPCACluster['cluster'], labels))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax4 = sns.scatterplot(dfPCACluster['PCA1'], dfPCACluster['PCA2'], x = "PCA1", y = "PCA2", hue = labels, s=40, palette = "Dark2")
plt.show()


# ##### when components = 2, PCA without feature selection

# In[42]:


gm = GaussianMixture(n_components = 2, random_state = 42)

labels = gm.fit_predict(dfPCACluster2)
clusters = pd.concat([dfNew, pd.DataFrame({'cluster':labels})], axis = 1)

print("Gaussian Mixture converged log-likelihood: ")
print(gm.lower_bound_)
print("\n")
print("Number of iterations required for log-likelihood to converge: ")
print(gm.n_iter_)
print("\n")

print("Silhouette Score: ", silhouette_score(dfPCACluster2, labels))
print("Calinski Harabasz Score: ", calinski_harabasz_score(dfPCACluster2, labels))
print("Davies Bouldin Score: ", davies_bouldin_score(dfPCACluster2, labels))
print("Rand index: ", metrics.rand_score(dfPCACluster2['cluster'], labels))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax4 = sns.scatterplot(dfPCACluster2['PCA1'], dfPCACluster2['PCA2'], x = "PCA1", y = "PCA2", hue = labels, s=40, palette = "Dark2")
plt.show()


# ##### when components = 3, PCA with feature selection

# In[43]:


gm = GaussianMixture(n_components = 3, random_state = 42)

labels = gm.fit_predict(dfPCACluster)
clusters = pd.concat([dfNew, pd.DataFrame({'cluster':labels})], axis = 1)

print("Gaussian Mixture converged log-likelihood: ")
print(gm.lower_bound_)
print("\n")
print("Number of iterations required for log-likelihood to converge: ")
print(gm.n_iter_)
print("\n")

print("Silhouette Score: ", silhouette_score(dfPCACluster, labels))
print("Calinski Harabasz Score: ", calinski_harabasz_score(dfPCACluster, labels))
print("Davies Bouldin Score: ", davies_bouldin_score(dfPCACluster, labels))
print("Rand index: ", metrics.rand_score(dfPCACluster['cluster'], labels))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax4 = sns.scatterplot(dfPCACluster['PCA1'], dfPCACluster['PCA2'], x = "PCA1", y = "PCA2", hue = labels, s = 40, palette = "Dark2")
plt.show()


# ##### when components = 3, PCA without feature selection

# In[44]:


gm = GaussianMixture(n_components = 3, random_state = 42)

labels = gm.fit_predict(dfPCACluster2)
clusters = pd.concat([dfNew, pd.DataFrame({'cluster':labels})], axis = 1)

print("Gaussian Mixture converged log-likelihood: ")
print(gm.lower_bound_)
print("\n")
print("Number of iterations required for log-likelihood to converge: ")
print(gm.n_iter_)
print("\n")

print("Silhouette Score: ", silhouette_score(dfPCACluster2, labels))
print("Calinski Harabasz Score: ", calinski_harabasz_score(dfPCACluster2, labels))
print("Davies Bouldin Score: ", davies_bouldin_score(dfPCACluster2, labels))
print("Rand index: ", metrics.rand_score(dfPCACluster2['cluster'], labels))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax4 = sns.scatterplot(dfPCACluster2['PCA1'], dfPCACluster2['PCA2'], x = "PCA1", y = "PCA2", hue = labels, s = 40, palette = "Dark2")
plt.show()


# ##### when components = 4, PCA with feature selection

# In[45]:


gm = GaussianMixture(n_components = 4, random_state = 42)

labels = gm.fit_predict(dfPCACluster)
clusters = pd.concat([dfNew, pd.DataFrame({'cluster':labels})], axis = 1)

print("Gaussian Mixture converged log-likelihood: ")
print(gm.lower_bound_)
print("\n")
print("Number of iterations required for log-likelihood to converge: ")
print(gm.n_iter_)
print("\n")

print("Silhouette Score: ", silhouette_score(dfPCACluster, labels))
print("Calinski Harabasz Score: ", calinski_harabasz_score(dfPCACluster, labels))
print("Davies Bouldin Score: ", davies_bouldin_score(dfPCACluster, labels))
print("Rand index: ", metrics.rand_score(dfPCACluster['cluster'], labels))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax4 = sns.scatterplot(dfPCACluster['PCA1'], dfPCACluster['PCA2'], x = "PCA1", y = "PCA2", hue = labels, s = 40, palette = "Dark2")
plt.show()


# ##### when components = 4, PCA without feature selection

# In[46]:


gm = GaussianMixture(n_components = 4, random_state = 42)

labels = gm.fit_predict(dfPCACluster2)
clusters = pd.concat([dfNew, pd.DataFrame({'cluster':labels})], axis = 1)

print("Gaussian Mixture converged log-likelihood: ")
print(gm.lower_bound_)
print("\n")
print("Number of iterations required for log-likelihood to converge: ")
print(gm.n_iter_)
print("\n")

print("Silhouette Score: ", silhouette_score(dfPCACluster2, labels))
print("Calinski Harabasz Score: ", calinski_harabasz_score(dfPCACluster2, labels))
print("Davies Bouldin Score: ", davies_bouldin_score(dfPCACluster2, labels))
print("Rand index: ", metrics.rand_score(dfPCACluster2['cluster'], labels))
print("\n")

print("Clusters count: ")
print(clusters['cluster'].value_counts())

plt.figure(figsize = (10,5))
ax4 = sns.scatterplot(dfPCACluster2['PCA1'], dfPCACluster2['PCA2'], x = "PCA1", y = "PCA2", hue = labels, s = 40, palette = "Dark2")
plt.show()


# ### Reference
# https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

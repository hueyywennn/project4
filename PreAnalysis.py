#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#to download and include plotly graphs in pdf
import plotly.offline as pyo
import plotly.graph_objs as go

pyo.init_notebook_mode()


# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px


# In[2]:


# load data from csv into data frame and check header
dataset_url = 'https://raw.githubusercontent.com/usman2155904/WQD7001/main/CC%20GENERAL.csv'
dfRaw = pd.read_csv(dataset_url)
dfRaw.head()


# In[3]:


#display dataset shape
print("Dataset Shape: ")
print(dfRaw.shape)


# In[4]:


#display dataset size
print("Dataset Size: ")
print(dfRaw.size)


# In[5]:


#display data types
print("Dataset Variable types: ")
print(dfRaw.info())


# In[6]:


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


# In[7]:


#calculate missing and null values ratio for all attributes
mnv_ratios = [ratio for ratio in (dfUpdated.isna().sum() / len(dfUpdated))] 
print([pair for pair in list(zip(dfUpdated.columns, mnv_ratios)) if pair[1] > 0])


# In[8]:


#BEFORE
#count null values of of all attributes
dfUpdated.isnull().sum()


# In[9]:


#AFTER
#fill mean values for null value of attributes and count again
dfUpdated['creditLim'].fillna(dfUpdated['creditLim'].mean(), inplace = True)
dfUpdated['minPymt'].fillna(dfUpdated['minPymt'].mean(), inplace = True)
dfUpdated.isnull().sum()


# In[10]:


#round off all attribute values of float data type in 4dp
dfNew = dfUpdated.round(4)

#remove custID attribute
dfNew.drop('custID', axis = 1, inplace = True)
dfNew


# In[11]:


#display dataset variance
print(dfNew.var())

x = ["remainBal", "balF", "purDone", "oneOffPur", "insPur", "cashAdv", "purF", "oneOffPurF", "insPurF", "cashAdvF", "cashAdvTRX", "purTRX", "creditLim", "pymtDone", "minPymt", "fullPymtPCT", "tenure"]

fig = px.line(x = x, 
              y = dfNew.var(), 
              title = "<b>(1) Variance of each Feature</b>", 
              color_discrete_sequence = ["magenta"])
fig.update_layout(xaxis_title = "Features", 
                  yaxis_title = "Feature Variance",
                  title_font_family = "Times New Roman", 
                  title_font_color = "maroon", 
                  font_family = "Arial", 
                  font_color = "navy")

print(fig.show())


# In[12]:


#display data dispersion
dfNew.describe().T


# In[13]:


#colors = ["yellow", "red", "green", "blue", "orange", "goldenrod", "magenta", "pink", "purple", "black", "olive", "navy", "brown", "cyan", "violet"]

#display histogram for all variables
colors = ['goldenrod']
dfNew.hist(bins = 30, 
           figsize = (20,15), 
           color = colors)
plt.suptitle("(2) Histogram for all features")
plt.rcParams.update({'font.size': 8})

plt.show()


# In[14]:


#pairwise correlation
dfNew.corr()


# In[15]:


#correlation in heatmap
fig = px.imshow(dfNew.corr(),  
                title = "<b>(3) Pairwise Correlation Heatmap Plot</b>",
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


# In[16]:


#for all frequencies
frequenciesType = ["balF", "purF", "oneOffPurF", "insPurF", "cashAdvF"]

fig = px.histogram(dfNew, 
                   x = frequenciesType, 
                   title = "<b>(4) Frequencies Histogram Plot</b>", 
                   color_discrete_sequence = ["orange", "red", "green", "blue", "purple"], 
                   barmode = 'overlay')
fig.update_layout(xaxis_title = "Type of Frequencies", 
                  legend_title = "Features", 
                  title_font_family = "Times New Roman", 
                  title_font_color = "maroon", 
                  font_family = "Arial", 
                  font_color = "navy")

fig.show()


# In[17]:


#for all transactions
trxType = ["cashAdvTRX", "purTRX"]
fig = px.scatter(dfNew, 
                 x = trxType, 
                 title = "<b>(5) Transactions Scatter Plot</b>", 
                 color_discrete_sequence = ["blue", "orange"])
fig.update_layout(xaxis_title = "Type of Transactions", 
                  legend_title = "Features", 
                  title_font_family = "Times New Roman", 
                  title_font_color = "maroon", 
                  font_family = "Arial", 
                  font_color = "navy")
fig.update_traces(marker = dict(size = 10, line = dict(width = 1, color = 'DarkSlateGrey')), 
                  selector = dict(mode = 'markers'))

fig.show()


# In[18]:


#for one off and installment purchase
purchaseType = ["oneOffPur", "insPur"]

fig = px.scatter(dfNew, 
                 x = purchaseType, 
                 title = "<b>(6) Credit card-Purchase Method Scatter Plot</b>", 
                 color_discrete_sequence = ["yellow", "cyan"])
fig.update_layout(xaxis_title = "Type of Credit Card Purchase", 
                  legend_title = "Features",
                  title_font_family = "Times New Roman", 
                  title_font_color = "maroon", 
                  font_family = "Arial", 
                  font_color = "navy")
fig.update_traces(marker = dict(size = 12, symbol = 'square', line = dict(width = 1, color = 'DarkSlateGrey')), 
                  selector = dict(mode = 'markers'))

fig.show()


# In[19]:


#for spending methods
#showing positive correlation
purchaseType = ["oneOffPur", "insPur", "cashAdv"]

fig = px.scatter_matrix(dfNew, 
                        dimensions = purchaseType, 
                        color = "purTRX",
                        title = "<b>(7) Comparison among three payment methods</b>",
                        color_continuous_scale = "tempo",
                        width = 800, height = 800)
fig.update_layout(title_font_family = "Times New Roman", 
                  title_font_color = "maroon", 
                  font_family = "Arial", 
                  font_color = "navy")

fig.show()


# In[20]:


#balance / credit limit
#purchase / credit limit
#one off / credit limit
#installment / credit limit
types = ["remainBal", "purDone", "oneOffPur", "insPur"]
colours = ["green", "magenta", "yellow", "pink"]

fig1 = px.histogram(dfNew, 
                    x = "creditLim", 
                    y = types, 
                    labels = {"creditLim" : "Credit Limit", "variable" : "Features"},
                    title = "<b>(8) Purchase and Balance against Credit Limit</b>",
                    color_discrete_sequence = colours)
fig1.update_layout(title_font_family="Times New Roman", 
                   title_font_color="maroon", 
                   font_family = "Arial", 
                   font_color = "navy")

fig2 = px.scatter(dfNew, 
                  x = "creditLim", 
                  y = types, 
                  title = "<b>(8) Purchase and Balance against Credit Limit</b>",
                  color_discrete_sequence = colours)
fig2.update_layout(xaxis_title = "Credit Limit", 
                   yaxis_title = " ", 
                   legend_title = "Features", 
                   title_font_family = "Times New Roman", 
                   title_font_color = "maroon", 
                   font_family = "Arial", 
                   font_color = "navy")
fig2.update_traces(marker = dict(size = 15, symbol = 'star-diamond', line = dict(width = 1, color = 'DarkSlateGrey')), 
                   selector = dict(mode = 'markers'))

fig1.show()
fig2.show()


# In[21]:


#purchase / balance
#one off / balance
#installment / balance
#cash / balance
##credit limit / balance
#payment done / balance
types = ["purDone", "oneOffPur", "insPur", "cashAdv", "creditLim", "pymtDone"]
colours = ["green", "red", "cyan", "blue", "yellow", "orange"]

fig1 = px.histogram(dfNew, 
                    x = "remainBal", 
                    y = types,
                    labels = {"remainBal" : "Balance", "variable" : "Features"},
                    title = "<b>(9) Features against Balance</b>", 
                    color_discrete_sequence = colours)
fig1.update_layout(title_font_family = "Times New Roman", 
                  title_font_color = "maroon", 
                  font_family = "Arial", 
                  font_color = "navy")

fig2 = px.scatter(dfNew, 
                  x = "remainBal", 
                  y = types, 
                  title = "<b>(9) Features against Balance</b>", 
                  color_discrete_sequence = colours)
fig2.update_layout(xaxis_title = "Balance", 
                   yaxis_title = " ", 
                   legend_title = "Features",
                   title_font_family = "Times New Roman", 
                   title_font_color = "maroon", 
                   font_family = "Arial", 
                   font_color = "navy")
fig2.update_traces(marker = dict(size = 15, symbol = 'triangle-right', line = dict(width = 1, color = 'DarkSlateGrey')), 
                   selector = dict(mode = 'markers'))

fig1.show()
fig2.show()


# In[22]:


#balance / purchase
#one off / purchase
#installment / purchase
#cash / purchase
##credit limit / purchase
#payment done / purchase
types = ["remainBal", "oneOffPur", "insPur", "cashAdv", "creditLim", "pymtDone"]

fig1 = px.histogram(dfNew, 
                    x = "purDone", 
                    y = types,
                    labels = {"purDone" : "Purchase", "variable" : "Features"},
                    title = "<b>(10) Features against Purchase</b>")
fig1.update_layout(title_font_family = "Times New Roman", 
                  title_font_color = "maroon", 
                  font_family = "Arial", 
                  font_color = "navy")

fig2 = px.scatter(dfNew, 
                  x = "purDone", 
                  y = types, 
                  title = "<b>(10) Features against Purchase</b>")
fig2.update_layout(xaxis_title = "Purchase", 
                   yaxis_title = " ", 
                   legend_title = "Features", 
                   title_font_family = "Times New Roman", 
                   title_font_color = "maroon", 
                   font_family = "Arial", 
                   font_color = "navy")
fig2.update_traces(marker = dict(size = 12, symbol = 'pentagon', line = dict(width = 1, color = 'DarkSlateGrey')), 
                   selector = dict(mode = 'markers'))

fig1.show()
fig2.show()


# In[23]:


#purchase / payment / tenure
fig = px.scatter(dfNew, 
                 x = "purDone", 
                 y = "pymtDone", 
                 color = "tenure", 
                 labels = {"purDone" : "Purchase Done", 
                           "pymtDone" : "Payment Made", 
                           "tenure" : "Tenure"},
                 title = "<b>(11) Purchase against Payment in Tenure</b>",
                 color_continuous_scale = "blackbody")
fig.update_layout(title_font_family = "Times New Roman", 
                  title_font_color = "maroon", 
                  font_family = "Arial", 
                  font_color = "navy")
fig.update_traces(marker = dict(size = 12, symbol = 'octagon-dot', line = dict(width = 1, color = 'DarkSlateGrey')), 
                  selector = dict(mode = 'markers'))


fig.show()


# In[24]:


#display skewness
dfNew.skew()


# In[25]:


#display boxplot to check features
x = ["remainBal", "purDone", "oneOffPur", "insPur", "cashAdv", "creditLim", "pymtDone", "minPymt"]

fig = px.box(dfNew, 
             x = x, 
             color_discrete_sequence = ["olive"],
             title = "<b>(12) All features in Box Plot</b>")
fig.update_layout(yaxis_title = "Features",
                  title_font_family = "Times New Roman", 
                  title_font_color = "maroon", 
                  font_family = "Arial", 
                  font_color = "navy")

fig.show()


# In[26]:


dfSelected = dfNew[["remainBal", "purDone", "oneOffPur", "insPur", "cashAdv", "creditLim", "pymtDone", "minPymt"]]

#create fx to check outliers using interquartile range
def checkOutliers(df, feature):
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    
    lowerBound = q1 - 1.5 * iqr
    upperBound = q3 + 1.5 * iqr
    
    outlierList = df.index[(df[feature] < lowerBound) | (df[feature] > upperBound)]
    return outlierList

#store all outliers into a list using for loop
storeList = []

for features in dfSelected:
    storeList.extend(checkOutliers(dfSelected, features))

print("Number of outliers: " )
print(len(storeList))

#create a fx to remove outliers and update df
def removeOutliers(df, outlierList):
    outlierList = sorted(set(outlierList))
    df = df.drop(outlierList)
    return df

dfIQR = removeOutliers(dfSelected, storeList)

outlierPCT = (dfIQR.size / dfSelected.size) * 100

print("Cleaned Data Shape: ")
print(dfIQR.shape)
print("Cleaned Data Size: ")
print(dfIQR.size)
print("Non-Outliers proportions in dataset: ")
print(str(outlierPCT.round(2)) + "%")


# In[27]:


dfSelected = dfNew[["remainBal", "purDone", "oneOffPur", "insPur", "cashAdv", "creditLim", "pymtDone", "minPymt"]]

#create fx to check outliers using z-sc0re (mean and std)

def checkOutliers(df, feature):
    lowerBound =  df[feature].mean() - (3 * df[feature].std())
    upperBound =  df[feature].mean() + (3 * df[feature].std())
    
    outlierList = df.index[(df[feature] < lowerBound) | (df[feature] > upperBound)]
    return outlierList

#store all outliers into a list using for loop
storeList = []

for features in ["remainBal", "purDone", "oneOffPur", "insPur", "cashAdv", "creditLim", "pymtDone", "minPymt"]:
    storeList.extend(checkOutliers(dfSelected, features))

print("Number of outliers: " )
print(len(storeList))

#create a fx to remove outliers and update df
def removeOutliers(df, outlierList):
    outlierList = sorted(set(outlierList))
    df = df.drop(outlierList)
    return df

dfZScore = removeOutliers(dfSelected, storeList)

outlierPCT = (dfZScore.size / dfSelected.size) * 100

print("Cleaned Data Shape: ")
print(dfZScore.shape)
print("Cleaned Data Size: ")
print(dfZScore.size)
print("Non-Outliers proportions in dataset: ")
print(str(outlierPCT.round(2)) + "%")


# In[28]:


#for iqr
#dfIQR.to_csv('cleanedDataIQR.csv')

#for z score
#dfZScore.to_csv('cleanedDataZScore.csv')


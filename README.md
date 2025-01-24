# Credit Card Customer Segmentation 
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### Project Overview
classify and segment credit card customers based on their transaction behaviours, and spending patterns.

### Dataset 
(https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
| Name            | Description                                                                                                                      |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------|
|`custID`               |Identification of Credit Card holder (Categorical)                                                                                |
|`remainBal`            | Balance amount left in their account to make purchases                                                                           |
|`balF`                 | How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)                |
|`purDone`              | Amount of purchases made from account                                                                                            |
|`oneOffPur`            | Maximum purchase amount done in one-go                                                                                           |
|`insPur`               | Amount of purchase done in installment                                                                                           |
|`cashAdv`              | Cash in advance given by the user                                                                                                |
|`purF`                 | How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)      |
|`oneOffPurF`           | How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)                        |
|`insPurF`              | How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)                           |
|`cashAdvF`             | How frequently the cash in advance being paid                                                                                    |
|`cashAdvTRX`           | Number of Transactions made with "Cash in Advanced"                                                                              |
|`purTRX`               | Numbe of purchase transactions made                                                                                              |
|`creditLim`            | Limit of Credit Card for user                                                                                                    |
|`pymtDone`             | Amount of Payment done by user                                                                                                   |
|`minPymt`              | Minimum amount of payments made by user                                                                                          |
|`fullPymtPCT`          | Percentage of full payment paid by user                                                                                          |
|`tenure`               | Tenure of credit card service for user                                                                                           |

### Features
1. **Data Cleaning**
  -	Feature renamed
  -	Handle missing values: cold deck imputation – mean
  -	Handle outliers: interquartile range, z-score
2. **Exploratory Data Analysis (EDA)**
  -	Statistical Summaries: mean, median, variance, and standard deviation
  -	Correlation Analysis: heatmap correlations
  -	Distribution Plots: histograms, scatter plots,
  -	Outlier Detection: boxplot
3. **Machine Learning Models**
  -	Data Normalization: Standard Scaler
  -	Clustering Algorithms: k-means, Agglomerative, Gaussian Mixture
  -	Dimensionality Reduction: Principal Component Analysis
  -	Segmentation Model Evaluation: elbow method, silhouette score, Davies-Bouldin index, Calinski Harabasz Method
  -	Predictive Modeling: clustering
4. **Interactive Visualizations**
  -	Cluster Visualization: dimensionality reduction with PCA

## Tools used
1. **Programming Language** 
  - Python
2. **Libraries**
  - pandas, numpy, scikit-learn, matplotlib
3. **Visualization Tools**
  - plotly, seaborn

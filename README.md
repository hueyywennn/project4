# Credit Card Customer Segmentation 
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### Project Overview
This project focuses on segmenting credit card customers based on their spending behavior and demographics. Using clustering techniques such as K-means, Agglomerative Clustering, and Gaussian Mixture Models, the analysis aims to uncover distinct customer groups. The insights generated help support targeted marketing strategies and personalized customer experiences.

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

## Project Objectives  
1. **Data Cleaning & Preprocessing**  
   - Rename inconsistent feature names for clarity  
   - Handle missing values using **cold deck imputation (mean substitution)**  
   - Detect and remove outliers using **interquartile range (IQR) and Z-score analysis**  

2. **Exploratory Data Analysis (EDA)**  
   - Compute key statistical summaries: mean, median, variance, and standard deviation  
   - Perform correlation analysis using **heatmaps**  
   - Visualize distributions with **histograms and scatter plots**  
   - Detect outliers using **boxplots**  

3. **Machine Learning Models for Customer Segmentation**  
   - **Data Normalization**: Apply **Standard Scaler** to standardize feature distributions  
   - **Clustering Algorithms**: Implement and compare  
     - **K-means**  
     - **Agglomerative Clustering**  
     - **Gaussian Mixture Model (GMM)**  
   - **Dimensionality Reduction**: Use **Principal Component Analysis (PCA)** for feature compression  
   - **Cluster Validation & Model Evaluation**:  
     - **Elbow Method**: Determine optimal cluster count  
     - **Silhouette Score**: Evaluate cluster compactness  
     - **Davies-Bouldin Index** & **Calinski-Harabasz Method**: Assess clustering performance  
   - **Predictive Modeling**: Assign new customers to clusters based on trained segmentation model  

## Technologies Used  
- **Programming Language**: Python  
- **Libraries**: pandas, numpy, scikit-learn, matplotlib  
- **Data Visualization Tools**: seaborn, plotly  

## Project Workflow  
1. **Data Collection**: Import and inspect datasets  
2. **Data Cleaning & Preprocessing**: Handle missing values, normalize data, and remove outliers  
3. **Exploratory Data Analysis (EDA)**: Visualize distributions, correlations, and patterns  
4. **Feature Engineering**: Transform variables for better clustering  
5. **Model Training**: Implement and compare different clustering algorithms  
6. **Model Evaluation**: Analyze clustering performance using multiple validation metrics  
7. **Results Interpretation**: Identify customer segments and derive actionable insights  

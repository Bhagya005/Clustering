# EDA and K-Clustering
## Introduction

The educational landscape is increasingly shaped by the interplay of student behaviors and academic performance. Understanding the factors influencing student engagement and success is crucial for educators and policymakers. In this exploratory data analysis (EDA), we delve into a dataset containing information on student behavior and academic outcomes. The goal is to identify trends, patterns, and correlations that can provide valuable insights into the dynamics of student performance.
## Problem Statement

The challenge lies in deciphering the intricate relationship between various aspects of student behavior and their academic achievements. Key questions include understanding how demographic factors, parental involvement, and student engagement metrics impact performance. By addressing these questions, we aim to uncover actionable insights that can inform educational strategies and interventions tailored to diverse student needs.

## Methodology

### 1. Data Collection

- The dataset encompasses information such as gender, nationality, educational level, and multiple metrics related to student engagement and performance.

### 2. Exploratory Data Analysis (EDA)

- Analyzing the distribution of categorical variables to understand the demographic makeup of the student population.
- Summarizing key statistics for numerical variables to gain insights into student engagement metrics.
- Examining correlations between different student behaviors using a correlation matrix.

### 3. K-Means Clustering

- Identifying distinct clusters among students based on their behaviors and characteristics.
- Focusing on features such as gender, education stage, and absence days to group students with similar profiles.
# Implementation
# Importing libraries
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```
# Exploratory data analysis
```python
## creating a percentage analysis of RP-wise distribution of data
round(df_class["gender"].value_counts(normalize=True)*100,2)
```
![image](https://github.com/Bhagya005/clustering/assets/106422457/4d80a1e2-8704-4c47-8588-eaae954eee06)

```python
## creating a percentage analysis of RP-wise distribution of data
round(df_class["NationalITy"].value_counts(normalize=True)*100,2)
```
![image](https://github.com/Bhagya005/clustering/assets/106422457/90598650-255c-44e1-93b9-0d10e7e3e22f)

```python
## creating a percentage analysis of RP-wise distribution of data
round(df_class["PlaceofBirth"].value_counts(normalize=True)*100,2)
```
![image](https://github.com/Bhagya005/clustering/assets/106422457/d54b3f55-af39-4636-8c86-5c863e3ebb70)

```python
## creating a percentage analysis of RP-wise distribution of data
round(df_class["StageID"].value_counts(normalize=True)*100,2)
```
![image](https://github.com/Bhagya005/clustering/assets/106422457/89b9715b-0c7c-427a-87fd-4a3fac4b1d47)

```python
## creating a percentage analysis of RP-wise distribution of data
round(df_class["GradeID"].value_counts(normalize=True)*100,2)
```
![image](https://github.com/Bhagya005/clustering/assets/106422457/f6e6b3d3-11c6-41e1-ba49-926c81d90b04)
 # K-means Clustering
 ## Finding the best value of k using elbow method
```python
input_col=["raisedhands","VisITedResources","AnnouncementsView","Discussion"]
X=df_class[input_col].values
```
```python
# Initialize an empty list to store the within-cluster sum of squares
from sklearn.cluster import KMeans
wcss = []

# Try different values of k
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)# here inertia calculate sum of square distance in each cluster

```
```python
# Plot the within-cluster sum of squares for different values of k
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method')
plt.show()
```
![image](https://github.com/Bhagya005/clustering/assets/106422457/a7305563-a3be-4da4-8de8-4496cc7444fb)
## Using Gridsearch method
```python
# Define the parameter grid
from sklearn.model_selection import GridSearchCV

param_grid = {'n_clusters': [2, 3, 4, 5, 6]}

# Create a KMeans object
kmeans = KMeans(n_init='auto',random_state=42)

# Create a GridSearchCV object
grid_search = GridSearchCV(kmeans, param_grid, cv=5)

# Perform grid search
grid_search.fit(X)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
```
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```
```python
print("Best Parameters:", best_params)
print("Best Score:", best_score)
```
## Implementing K-means clustering
```python
# Perform k-means clustering
k = 3 # Number of clusters
kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
kmeans.fit(X)
```
![image](https://github.com/Bhagya005/clustering/assets/106422457/2b985b5a-6e12-4b09-a967-2640a423fbed)
## Extracting labels and cluster centers
```python
# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Add the cluster labels to the DataFrame
df_class['Cluster'] = labels

```
```python
df_class.head()
```
![image](https://github.com/Bhagya005/clustering/assets/106422457/0aa552ac-8c7e-4370-b580-785c3c66115a)
## Visualizing the clustering using first two features
```python
# Visualize the clusters
plt.scatter(X[:, 1], X[:, 2], c=labels, cmap='viridis')
plt.scatter(centroids[:,1], centroids[:, 2], marker='X', s=200, c='red')
plt.xlabel(input_col[1])
plt.ylabel(input_col[2])
plt.title('K-means Clustering')
plt.show()
```
![image](https://github.com/Bhagya005/clustering/assets/106422457/0130df68-892b-4d58-829a-24867504c135)
## Perception on raisedhands over Clustors
```python
pd.crosstab(columns = df_class['Cluster'], index = df_class['raisedhands'])

```
![image](https://github.com/Bhagya005/clustering/assets/106422457/ae029428-8d6e-42f7-985a-c77f249ceb27)

## Results

### 1. Categorical Variables Distribution

- Balanced gender distribution.
- Diverse educational levels.
- Even distribution across semesters.
- Significant parental participation in surveys.
- Mixed levels of parental school satisfaction.
- Inclusion of students with varying absence rates.

### 2. Numerical Variables Summary

- Varied classroom participation levels.
- Diverse resource utilization patterns.
- Moderate engagement with announcements.
- Wide-ranging discussion participation rates.

### 3. Correlation Matrix Insights

- Positive correlations between raised hands, visited resources, and announcements view.
- Lower correlation with discussion participation, suggesting independence.

### 4. K-Means Clustering

- Identified three clusters with varying levels of engagement.
- Cluster 0: Lower engagement.
- Cluster 1: Moderate engagement.
- Cluster 2: Higher engagement.

## Conclusion

This EDA provides a comprehensive overview of the dataset, shedding light on the intricate connections between student behaviors and academic performance. The identified clusters offer a nuanced understanding of student engagement patterns. These findings can guide educators in tailoring interventions to meet the diverse needs of students, ultimately contributing to enhanced academic outcomes. Further research can explore the impact of these insights on targeted educational strategies and assess the longitudinal effects on student success.







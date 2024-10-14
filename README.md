# Fuzzy-logic-pbl
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Data Loading
# Load the customer data into a DataFrame
data = pd.read_csv(r"C:\Users\trish\OneDrive\Documents\Downloads\customer_data.csv")
  # Replace with your data file path

# Step 2: Data Preprocessing
# Check for missing values
print(data.isnull().sum())

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Age', 'Income']])  # Focus on Age and Income

# Step 3: Finding the Optimal Number of Clusters
# Use the Elbow method to find the optimal number of clusters
inertia = []
range_clusters = range(1, 11)
for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(8, 6))
plt.plot(range_clusters, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Use Silhouette score to validate the number of clusters
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    silhouette_scores.append(silhouette_score(scaled_data, labels))

# Plot the Silhouette scores
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Analysis for Optimal k')
plt.grid(True)
plt.show()

# Step 4: K-means Clustering with the Optimal Number of Clusters
optimal_k = 4  # Replace with the optimal k found from the Elbow method and Silhouette score
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to the original data
data['Cluster'] = clusters

# Step 5: Data Visualization and Cluster Analysis
# Visualize the clusters using a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Age', y='Income', hue='Cluster', palette='viridis', s=100)
plt.title('Market Segmentation using K-means Clustering')
plt.xlabel('Customer Age')
plt.ylabel('Customer Income')
plt.grid(True)
plt.show()

# Analyze each cluster's characteristics
cluster_summary = data.groupby('Cluster').mean()
print(cluster_summary)

# Step 6: Challenges in Selecting the Optimal Number of Clusters
# Discuss the difficulties in selecting k, such as subjective interpretation, and variations in silhouette score.

print("Project completed: K-means clustering successfully applied for market segmentation using Age and Income.")

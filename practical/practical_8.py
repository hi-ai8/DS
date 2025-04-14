import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Sample dataset
data = {
    'Annual_Income': [15, 16, 17, 18, 80, 85, 86, 87, 50, 52, 53, 54],
    'Spending_Score': [39, 35, 40, 42, 81, 77, 79, 83, 60, 62, 65, 63]
}
df = pd.DataFrame(data)

# Step 2: Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Step 3: Elbow Method to find optimal k
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Step 4: Apply K-Means with optimal k (let's choose k=3)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# Step 5: Visualize clusters
plt.figure(figsize=(8, 6))
for cluster in range(k):
    clustered_data = df[df['Cluster'] == cluster]
    plt.scatter(clustered_data['Annual_Income'], clustered_data['Spending_Score'], label=f'Cluster {cluster}')
    
# Plot centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='black', marker='X', label='Centroids')
plt.title('K-Means Clustering Results')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Analyze clusters
print("\nCluster Centers (Scaled Back):")
for i, center in enumerate(centers):
    print(f"Cluster {i + 1} center: Income = {center[0]:.2f}, Spending Score = {center[1]:.2f}")

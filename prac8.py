import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Create synthetic dataset
df = pd.DataFrame({
    'Feature1': [1, 2, 3, 8, 9, 10, 15, 16, 17],
    'Feature2': [2, 3, 4, 7, 8, 9, 12, 13, 14]
})

# Standardize data
X_scaled = StandardScaler().fit_transform(df)

# Elbow Method for optimal k
inertia = []
for k in range(1, 6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(5, 3))
plt.plot(range(1, 6), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.tight_layout()
plt.show()

# Apply K-Means with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Calculate and print silhouette score
print(f'Silhouette Score: {silhouette_score(X_scaled, df["Cluster"]):.4f}')

# Visualize clusters
plt.figure(figsize=(5, 3))
plt.scatter(df['Feature1'], df['Feature2'], c=df['Cluster'], cmap='viridis')
plt.title('K-Means Clustering Results')
plt.tight_layout()
plt.show()
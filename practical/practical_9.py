import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Step 1: Load dataset (you can also load your own CSV)
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
print("Original Data Shape:", df.shape)

# Step 2: Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Step 3: Apply PCA
pca = PCA()
pca_data = pca.fit_transform(scaled_data)

# Step 4: Explained Variance Ratio
explained_variance = pca.explained_variance_ratio_
print("\nExplained Variance Ratio:", explained_variance)
print("Cumulative Explained Variance:", np.cumsum(explained_variance))

# Step 5: Plot cumulative explained variance
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(explained_variance), marker='o', linestyle='--')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

# Step 6: Reduce dimensions to 2 for visualization
pca_2d = PCA(n_components=2)
reduced_data = pca_2d.fit_transform(scaled_data)

# Step 7: Visualization
plt.figure(figsize=(8,6))
plt.scatter(reduced_data[:,0], reduced_data[:,1], c=iris.target, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: 2D Visualization')
plt.colorbar(label='Target Class')
plt.grid(True)
plt.show()

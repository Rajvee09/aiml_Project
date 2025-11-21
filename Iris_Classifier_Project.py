import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- 1. Data Generation and Representation (CO3) ---

np.random.seed(42)
data_size = 50


spending = np.concatenate([
    np.random.normal(3000, 500, data_size // 3), # Low Spenders
    np.random.normal(7000, 800, data_size // 3), # Medium Spenders
    np.random.normal(12000, 1000, data_size - 2 * (data_size // 3)) # High Spenders
])

frequency = np.concatenate([
    np.random.normal(5, 1, data_size // 3), # Low Frequency
    np.random.normal(15, 3, data_size // 3), # Medium Frequency
    np.random.normal(25, 5, data_size - 2 * (data_size // 3)) # High Frequency
])

# Combine into a DataFrame (CO3 - Data Representation)
data = pd.DataFrame({
    'Annual_Spending': spending,
    'Visit_Frequency': frequency
})

print("--- 1. Synthetic Customer Data Sample (CO3) ---")
print(data.head())
print(f"\nStatistical Summary:\n{data.describe().loc[['mean', 'std']]}")

# --- 2. Data Preprocessing (Scaling) ---

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# --- 3. Unsupervised Clustering (CO4 - K-Means) ---
K = 3
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)

# Train the model (Unsupervised Learning)
kmeans.fit(X_scaled)

# Add the resulting cluster labels back to the original data
data['Cluster'] = kmeans.labels_

print("\n--- 4. Clustering Results (CO4) ---")
print(f"Centroids (Centers of the Clusters):\n{kmeans.cluster_centers_}")
print(f"Customer Counts per Cluster:\n{data['Cluster'].value_counts()}")

# --- 5. Visualization of Segments ---
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    data['Annual_Spending'],
    data['Visit_Frequency'],
    c=data['Cluster'],
    cmap='viridis',
    s=100,
    alpha=0.7,
    edgecolors='k'
)

# Plotting the cluster centers (the means)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c='red',
    marker='X',
    s=200,
    label='Cluster Centers'
)

plt.title(f'Customer Segmentation using K-Means (K={K})')
plt.xlabel('Annual Spending (in USD)')
plt.ylabel('Visit Frequency (Times per Year)')
plt.legend()
plt.grid(True)
plt.show()
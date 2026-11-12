# K-Means Clustering on the Iris Dataset
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Load the Iris dataset and use only the first two features
iris = load_iris()
X = iris.data[:, :2]

# 2. Perform K-Means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# 3. Visualize clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap="viridis")

# Plot cluster centroids
centers = kmeans.cluster_centers_
plt.scatter(
    centers[:, 0], centers[:, 1],
    c="red", s=200, marker="X", label="Centroids"
)

plt.title("K-Means Clustering on Iris Dataset (k=3)")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.show()

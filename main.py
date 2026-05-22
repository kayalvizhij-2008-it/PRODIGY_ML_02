import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv(r"c:\Users\kayal\Downloads\archive (1)\Mall_Customers.csv")

# Select features
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(6,5))
plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Train KMeans Model
kmeans = KMeans(n_clusters=5, random_state=42)

# Predict clusters
y_kmeans = kmeans.fit_predict(X)

# Plot Customer Segmentation
plt.figure(figsize=(6,5))

plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=y_kmeans
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")

plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def load_customer_data(csv_file):
    """
    Load customer data from a CSV file into a pandas DataFrame.

    Parameters:
        csv_file (str): Path to the CSV file containing customer data.

    Returns:
        pd.DataFrame: DataFrame containing customer data.
    """
    # Read the CSV file into a DataFrame
    customer_data = pd.read_csv(csv_file)
    return customer_data

def preprocess_data(data):
    """
    Preprocess the customer data.

    Parameters:
        data (pd.DataFrame): DataFrame containing customer data.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Extract relevant features for clustering
    features = ['age', 'income', 'spending_score']
    X = data[features]
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled

def determine_optimal_clusters(X_scaled, max_clusters=10):
    """
    Determine the optimal number of clusters using the Elbow Method.

    Parameters:
        X_scaled (pd.DataFrame): Preprocessed DataFrame.
        max_clusters (int): Maximum number of clusters to consider.

    Returns:
        int: Optimal number of clusters.
    """
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, max_clusters + 1), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()
    return

def perform_kmeans_clustering(X_scaled, num_clusters):
    """
    Perform K-means clustering on the preprocessed data.

    Parameters:
        X_scaled (pd.DataFrame): Preprocessed DataFrame.
        num_clusters (int): Number of clusters for K-means clustering.

    Returns:
        np.array: Array of cluster labels.
        KMeans: Fitted KMeans model.
    """
    # Apply K-means clustering with the specified number of clusters
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    return kmeans.labels_, kmeans

def visualize_clusters(X_scaled, labels, centroids):
    """
    Visualize the clusters in a 2D space.

    Parameters:
        X_scaled (pd.DataFrame): Preprocessed DataFrame.
        labels (np.array): Array of cluster labels.
        centroids (np.array): Array of cluster centroids.
    """
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids')
    plt.title('Customer Segmentation')
    plt.xlabel('Income')
    plt.ylabel('Spending Score')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Step 1: Load Customer Data
    csv_file_path = 'customer_data.csv'
    customer_data = load_customer_data(csv_file_path)

    # Step 2: Preprocess Data
    X_scaled = preprocess_data(customer_data)

    # Step 3: Determine Optimal Number of Clusters
    determine_optimal_clusters(X_scaled)

    # Step 4: Perform K-means Clustering
    num_clusters = 5  # You can adjust this based on the Elbow Method plot
    labels, kmeans_model = perform_kmeans_clustering(X_scaled, num_clusters)

    # Step 5: Visualize the Clusters
    centroids = kmeans_model.cluster_centers_
    visualize_clusters(X_scaled, labels, centroids)

    # Step 6: Evaluate Clustering Quality (Optional)
    silhouette_avg = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {silhouette_avg}")

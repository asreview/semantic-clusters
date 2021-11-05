# Imports

# System stuff
import os

# Clustering
from sklearn.cluster import KMeans

# data
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def run_KMeans(features, n_clusters, n_init):
    """Function to perform KMeans clustering
    Args:
      features: (np array) Shape n_samples x n_features containing input data (t-sne embeddings)
      n_clusters: (int) Number of clusters to be used in KMeans algorithm
      n_init: (int) Number of restarts for KMeans algorithm
    Returns:
      kmeans.labels_: (np array) Contains the predicted cluster labels for the datapoints
    """
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init).fit(features)
    return kmeans.labels_

def visualize_clusters(df):
    """Function to plot and visualize the KMeans clusters"""
    
    # Retrieve features and clusters
    x = df.x
    y = df.y
    cord_uid = df.cord_uid
    cluster_id = df.cluster_id

    # Get fig and ax
    fig, ax = plt.subplots()
    ax.set_title("Cord-19 Database Semantic Clusters")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")

    # Do actual plotting and save image
    #ax.plot(x,y, 'o', linestyle="", c=x, cmap="jet")
    ax.scatter(x,y,c=cluster_id,cmap="Set3")
    if not os.path.exists("img"):
        os.makedirs("img")
    filename = f"clusters.png"
    img_path = os.path.join("img",filename)
    fig.savefig(img_path)
    
if __name__ == "__main__":

    # Get t-SNE features
    tsne_df_path = os.path.join("data","dataframes","tsne_df.csv")
    df = pd.read_csv(tsne_df_path)
    print(df.head())
    features = df.iloc[:,1:].values
    
    # Define parameters
    n_clusters = 10
    n_init = 10

    # Either check if we have kmeans df and load
    kmeans_df_path = os.path.join("data","dataframes","kmeans_df.csv")
    if os.path.exists(kmeans_df_path):
        print("Loading saved KMeans dataframe!")
        df = pd.read_csv(kmeans_df_path)

    # .. or Run KMeans and add resulting clusters to dataframe
    else:
        labels = run_KMeans(features, n_clusters, n_init)
        df['cluster_id'] = labels
        print()
        print(df.head())
        df.to_csv(kmeans_df_path, index=None)

    # Do some preliminary matplotlib plotting
    visualize_clusters(df)

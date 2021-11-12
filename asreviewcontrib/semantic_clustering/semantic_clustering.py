#!/usr/bin/python
# -*- coding: utf-8 -*-
# Path: asreviewcontrib\semantic_clustering\semantic_clustering.py

import os
from tqdm import tqdm
import numpy as np

from sklearn.cluster import KMeans
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModel
from transformers import logging
import matplotlib.pyplot as plt
import seaborn as sns

from asreviewcontrib.semantic_clustering.dim_reduct import run_pca
from asreviewcontrib.semantic_clustering.dim_reduct import t_sne
from asreviewcontrib.semantic_clustering.clustering import run_KMeans

# Setting environment
logging.set_verbosity_error()
sns.set()
tqdm.pandas()


def SemanticClustering(asreview_data_object):

    # load data
    print("Loading data...")
    data = _load_data(asreview_data_object)

    # since processing the data can take a long time, for now the data is cut
    # down to decrease test duration. This will be removed in future versions
    # data = data.iloc[:30, :]

    # load scibert transformer
    print("Loading scibert transformer...")
    transformer = 'allenai/scibert_scivocab_uncased'

    # load transformer and tokenizer
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(transformer)
    model = AutoModel.from_pretrained(transformer)

    # tokenize abstracts and add to data
    print("Tokenizing abstracts...")
    data['tokenized'] = data['abstract'].progress_apply(
        lambda x: tokenizer.encode_plus(
            x,
            add_special_tokens=False,
            truncation=True,
            max_length=512,
            # padding='max_length',
            return_tensors='pt'))

    # generate embeddings and format correctly
    print("Generating embeddings...")
    data['embeddings'] = data['tokenized'].progress_apply(lambda x: model(
        **x,
        output_hidden_states=False)[-1].detach().numpy().squeeze())

    # from here on the data is not directly attached to the dataframe anymore,
    # as a result of legacy code. This will be fixed in a future PR.

    # run pca
    print("Running PCA...")
    pca = run_pca(data['embeddings'].tolist(), n_components=.98)

    # run t-sne
    print("Running t-SNE...")
    tsne = t_sne(pca, n_iter=1000)

    # calculate optimal number of clusters
    print("Calculating optimal number of clusters...")
    n_clusters = _calc_optimal_n_clusters(tsne)
    print("Optimal number of clusters: ", n_clusters)

    # run k-means. n_init is set to 10, this indicated the amount of restarts
    # for the KMeans algorithm. 10 is the sklearn default.
    print("Running k-means...")
    labels = run_KMeans(tsne, n_clusters, 10)

    # visualize clusters
    print("Visualizing clusters...")
    _visualize_clusters(tsne, labels)

    # create file for use in interactive dashboard
    _create_file(data, tsne, labels)


# Create functional dataframe and store to file for use in interactive
def _create_file(data, coords, labels):
    data['x'] = coords[:, 0]
    data['y'] = coords[:, 1]
    data['cluster_id'] = labels

    if not os.path.exists("data"):
        os.makedirs("data")

    kmeans_df_path = os.path.join("data", "kmeans_df.csv")
    data.to_csv(kmeans_df_path, index=None)


# Calculate the optimal amount of clusters. It checks the inertia for 1 to 25
# clusters, and picks the optimal inertia based on an elbow graph and some cool
# trigonometry.
def _calc_optimal_n_clusters(features):

    sum_of_squared_distances = []

    K = range(1, 25)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(features)
        sum_of_squared_distances.append(km.inertia_)

    max = 0
    clusters = 1

    for i in K:
        p1 = np.asarray((sum_of_squared_distances[0], 1))
        p2 = np.asarray(
            (sum_of_squared_distances[-1], (len(sum_of_squared_distances) + 1)))
        p3 = np.asarray((sum_of_squared_distances[i - 1], i))

        m = np.cross(p2 - p1, p3 - p1) / norm(p2 - p1)

        if m > max:
            max = m
            clusters = i

    return clusters


def _visualize_clusters(tsne, labels):
    fig, ax = plt.subplots()
    ax.set_title("semantic clustering")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")

    x = tsne[:, 0]
    y = tsne[:, 1]

    # Do actual plotting and save image
    ax.scatter(x, y, c=labels, cmap="Set3")
    if not os.path.exists("img"):
        os.makedirs("img")
    filename = "clusters.png"
    img_path = os.path.join("img", filename)
    fig.savefig(img_path)


def _load_data(asreview_data_object):

    # extract title and abstract, drop empty abstracts and reset index
    data = asreview_data_object.df[['title', 'abstract']].copy()
    data['abstract'] = data['abstract'].replace('', np.nan, inplace=False)
    data.dropna(subset=['abstract'], inplace=True)
    data = data.reset_index(drop=True)

    return data

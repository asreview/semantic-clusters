#!/usr/bin/python
# -*- coding: utf-8 -*-
# Path: asreviewcontrib\semantic_clustering\semantic_clustering.py

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModel
from transformers import logging
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Setting environment
logging.set_verbosity_error()
sns.set()
tqdm.pandas()

REMOVE_DUPLICATES = True


def run_clustering_steps(
        asreview_data_object,
        output_file,
        transformer='allenai/scibert_scivocab_uncased'):

    # load data
    print("Loading data...")
    data = pd.DataFrame({
        "title": asreview_data_object.title,
        "abstract": asreview_data_object.abstract,
        "included": asreview_data_object.included
    })

    # remove emptry abstracts
    data = data[data['abstract'] != '']

    if REMOVE_DUPLICATES:
        try:
            data["dup"] = asreview_data_object.df["duplicate_record_id"]
            print("Size before removing duplicates is {0}".format(len(data)))
            data = data[data['dup'].isna()]
            print("size after removing duplicates is {0}".format(len(data)))
        except KeyError:
            pass

    # reset index
    data.reset_index(drop=True, inplace=True)

    # load transformer and tokenizer
    print(f"Loading tokenizer and model {transformer}...")
    tokenizer = AutoTokenizer.from_pretrained(transformer)
    model = AutoModel.from_pretrained(transformer)

    # tokenize abstracts and add to data
    print("Tokenizing abstracts...")
    encoded = tokenizer.batch_encode_plus(
        data['abstract'].tolist(),
        add_special_tokens=False,
        truncation=True,
        max_length=200,
        padding='max_length',
        return_tensors='pt')

    # generate embeddings and format correctly
    print("Generating embeddings...")
    embeddings = []
    for x in tqdm(encoded.input_ids):
        embeddings.append(model(x.unsqueeze(0), output_hidden_states=False)[-1].detach().numpy().squeeze())  # noqa: E501

    # from here on the data is not directly attached to the dataframe anymore,
    # as a result of legacy code. This will be fixed in a future PR.

    # run pca
    print("Running PCA...")
    pca = PCA(n_components=.98)
    pca = pca.fit_transform(embeddings)

    # run t-sne
    print("Running t-SNE...")
    tsne = TSNE(n_components=2,
                n_iter=1000,
                perplexity=6,
                n_jobs=4,
                learning_rate=2000,
                early_exaggeration=12).fit_transform(pca)

    # calculate optimal number of clusters
    print("Calculating optimal number of clusters...")
    n_clusters = _calc_optimal_n_clusters(tsne)
    print("Optimal number of clusters: ", n_clusters)

    # run k-means. n_init is set to 10, this indicated the amount of restarts
    # for the KMeans algorithm. 10 is the sklearn default.
    print("Running k-means...")
    labels = KMeans(n_clusters).fit(tsne).labels_

    # create file for use in interactive dashboard
    print("Creating file {0}...".format(output_file))
    data['x'] = tsne[:, 0]
    data['y'] = tsne[:, 1]
    data['cluster_id'] = labels

    data.to_csv(output_file, index=None)


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

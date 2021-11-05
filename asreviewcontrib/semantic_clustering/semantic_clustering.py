# Copyright 2021 The ASReview Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# imports
from sklearn.cluster import KMeans
from numpy.linalg import norm
import os
from tqdm import tqdm
from dim_reduct import run_pca
from dim_reduct import t_sne
from clustering import run_KMeans
from asreview.data import ASReviewData
import numpy as np
from transformers import AutoTokenizer, AutoModel
from transformers import logging
import matplotlib.pyplot as plt
import seaborn as sns
logging.set_verbosity_error()
sns.set()
tqdm.pandas()


def SemanticClustering(asreview_data_object):

    # load data
    print("Loading data...")
    data = load_data(asreview_data_object)

    # cut data for testing
    data = data.iloc[:30, :]

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
    n_clusters = calc_optimal_n_clusters(tsne)
    print("Optimal number of clusters: ", n_clusters)

    # run k-means
    print("Running k-means...")
    labels = run_KMeans(tsne, n_clusters, 10)

    # visualize clusters
    print("Visualizing clusters...")
    tsne_data = [tsne[:, 0], tsne[:, 1]]
    visualize_clusters(tsne_data, labels)


# Optimal n clusters, very inefficient, to be done over later
def calc_optimal_n_clusters(features):

    Sum_of_squared_distances = []

    K = range(1, 25)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(features)
        Sum_of_squared_distances.append(km.inertia_)

    max = 0
    clusters = 1

    for i in K:
        p1 = np.asarray((Sum_of_squared_distances[0], 1))
        p2 = np.asarray(
            (Sum_of_squared_distances[-1], (len(Sum_of_squared_distances)+1)))
        p3 = np.asarray((Sum_of_squared_distances[i-1], i))

        m = np.cross(p2-p1, p3-p1)/norm(p2-p1)

        if m > max:
            max = m
            clusters = i

    return clusters


def visualize_clusters(data, labels):
    fig, ax = plt.subplots()
    ax.set_title("semantic clustering")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")

    x = data[0]
    y = data[1]

    # Do actual plotting and save image
    ax.scatter(x, y, c=labels, cmap="Set3")
    if not os.path.exists("img"):
        os.makedirs("img")
    filename = "clusters.png"
    img_path = os.path.join("img", filename)
    fig.savefig(img_path)


def load_data(asreview_data_object):

    # extract title and abstract, drop empty abstracts and reset index
    data = asreview_data_object.df[['title', 'abstract']].copy()
    data['abstract'] = data['abstract'].replace('', np.nan, inplace=False)
    data.dropna(subset=['abstract'], inplace=True)
    data = data.reset_index(drop=True)

    return data


if __name__ == "__main__":
    filepath = "https://raw.githubusercontent.com/asreview/systematic-review-datasets/master/datasets/van_de_Schoot_2017/output/van_de_Schoot_2017.csv"
    SemanticClustering(ASReviewData.from_file(filepath))

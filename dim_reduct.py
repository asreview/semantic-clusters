##################################################
##### Functions for dimensionality reduction #####
##################################################

# imports

# System stuff
import os
import sys
import pickle

# Data stuff
import numpy as np 
import pandas as pd 

# Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Plotting
import matplotlib.pyplot as plt

def get_embs(embs_path, emb_shape):
    """Function to retrieve all embedding vectors from corresponding files
    Args:
      embs_path: Path to folder containing all files with embeddings
      emb_shape: Shape for each embedding (e.g. 1x768 for BERT models)
    """
    
    # If embeddings array already exists read embeddings
    embs_file = "PCA_array.pickle"
    pca_array_path = os.path.join("data",embs_file)
    if os.path.exists(pca_array_path):
        with open(pca_array_path, 'rb') as file:
            embs = pickle.load(file)
            return embs

    # Initialize empty array of emb_shape
    embs = np.empty((0,emb_shape[1]))

    for i, file in enumerate(os.listdir(embs_path)):
        #filename = file[:-7] # Remove .pickle

        # Load each embedding and put them in dict with cord_uid
        with open(os.path.join(embs_path,file), "rb") as emb_file:
            emb = pickle.load(emb_file)
            embs = np.concatenate((embs,emb),axis=0)
    
    # Pickle the embs array
    with open(pca_array_path, "wb+") as file:
        pickle.dump(embs, file)

    return embs

def run_pca(embs, n_components):
    """Function to perform Principal Components Analysis
    Args:
      embs: 2-dimensional Numpy array of n_samples x n_features containing embeddings
      n_components: Number of components: num \in [0,1] == percentage of variance retained
    """

    # Number of components is either raw number or percentage of variance retained
    pca = PCA(n_components = n_components)
    pca.fit(embs)
    #print(pca.explained_variance_ratio_)

    # Get array with reduced dimensions
    embs = pca.fit_transform(embs)
    
    return embs

def t_sne(embs, n_iter):
    """Function to perform t-distributed Stochastic Neighbor Embedding
    Args:
      embs: Numpy array of n_samples x n_pca_components with PCA embeddings
    """

    # Build 
    t_sne_embeddings = TSNE(n_components=2,
                            #n_iter=100000,
                            n_iter=n_iter,
                            perplexity=6,
                            n_jobs=4,
                            learning_rate=2000,
                            early_exaggeration=12).fit_transform(embs)

    return t_sne_embeddings

def plot_embs(embs, filenames, n_iter):
    """Function to plot (t-sne) embedding
    Args:
      embs: Numpy array of n_samples x n_t_sne_components with (t-sne) embeddings
    """

    # Get fig and ax
    fig, ax = plt.subplots()

    # Get t-sne components
    x = embs[:,0]
    y = embs[:,1]

    # Generate DataFrame
    data = {'cord_uid': filenames, 'x': x, 'y': y}
    df = pd.DataFrame(data)
    df.to_csv('tsne_df.csv', index=None)

    # Do actual plotting and save image
    ax.plot(x,y, 'o')
    if not os.path.exists("img"):
        os.makedirs("img")
    filename = f"tsne{n_iter}.png"
    img_path = os.path.join("img",filename)
    fig.savefig(img_path)

if __name__ == "__main__":

    # Path to embeddings
    embs_path = os.path.join("data","embs")

    # Get embedding size
    for file in os.listdir(embs_path):
        with open(os.path.join(embs_path,file), "rb") as emb_file:
            emb = pickle.load(emb_file)
            emb_shape = emb.shape

        # Break because we only need one
        break

    # Retrieve embeddings from file as 2-dim feature array: n_samples x n_features
    embs = get_embs(embs_path, emb_shape)
    print(embs.shape)

    # Now we can do actual PCA!
    n_components = .98
    pca_embs = run_pca(embs, n_components)
    print(f"PCA dim: {pca_embs.shape}")

    # Get all filenames
    filenames = []
    for i, filename in enumerate(os.listdir(embs_path)):
        filenames.append(filename[:-7])

    # And proceed with t-SNE
    n_iter = 200000
    t_sne_embs = t_sne(pca_embs,n_iter)

    # Plot result
    plot_embs(t_sne_embs, filenames,n_iter)
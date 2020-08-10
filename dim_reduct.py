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

# PCA
from sklearn.decomposition import PCA

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
    embs = np.empty(emb_shape)

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

def run_PCA(embs):
    """Function to perform Principal Components Analysis
    Args:
      embs: 2-dimensional Numpy array of n_samples x n_features containing embeddings
    """


if __name__ == "__main__":

    # Path to embeddings
    embs_path = os.path.join("data","embs_np")

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
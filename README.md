# ASReview-Semantic-clustering
This repository contains the Semantic Clustering plugin for ASReview. It applies multiple techniques (SciBert, PCA, T-SNE, KMeans, a custom Cluster Optimizer) to an ASReview data object, in order to cluster records based on semantic differences. The end result is an interactive dashboard:

![Alt Text](/docs/cord19_semantic_clusters.gif)

## Usage
Currently, the Dashboard is still being implemented, but all other functionality is in [`asreviewcontrib/semantic_clustering/semantic_clustering.py`](asreviewcontrib/semantic_clustering/semantic_clustering.py).

## License

MIT license

# ASReview-Semantic-clustering
This repository contains the Semantic Clustering plugin for ASReview. It applies
multiple techniques (SciBert, PCA, T-SNE, KMeans, a custom Cluster Optimizer) to
an ASReview data object, in order to cluster records based on semantic
differences. The end result is an interactive dashboard:

![Alt Text](/docs/cord19_semantic_clusters.gif)

## Usage
The usage of the semantic clustering app is found in the main.py file. The
following commands can be run:

```bash
py asreviewcontrib\semantic_clustering\main.py -f
py asreviewcontrib\semantic_clustering\main.py --filepath
```

The filepath argument starts the processing of a file for clustering. This file
will be saved to `\data` after the processing is done. It can be run the
following way:

```bash
py asreviewcontrib\semantic_clustering\main.py -f "https://raw.githubusercontent.com/asreview/systematic-review-datasets/master/datasets/van_de_Schoot_2017/output/van_de_Schoot_2017.csv"
```





        print('Please use the following format:')
        print('test.py -f <filepath>')
        print('test.py --testfile')
        print('test.py --app')

## License

MIT license

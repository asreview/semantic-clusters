# ASReview-Semantic-clustering
This repository contains the Semantic Clustering plugin for ASReview. It applies
multiple techniques (SciBert, PCA, T-SNE, KMeans, a custom Cluster Optimizer) to
an ASReview data object, in order to cluster records based on semantic
differences. The end result is an interactive dashboard:

![Alt Text](/docs/cord19_semantic_clusters.gif)

## Usage
The usage of the semantic clustering app is found in the main.py file. The
following commands can be run:

### Processing
```console
py asreviewcontrib\semantic_clustering\main.py -f <url or local file>
py asreviewcontrib\semantic_clustering\main.py --filepath <url or local file>
```

The filepath argument starts the processing of a file for clustering. This file
will be saved to the `data` folder after the processing is done. An example of
usage can be:

```console
py asreviewcontrib\semantic_clustering\main.py -f "https://raw.githubusercontent.com/asreview/systematic-review-datasets/master/datasets/van_de_Schoot_2017/output/van_de_Schoot_2017.csv"
```

### Processing testfile
```console
py asreviewcontrib\semantic_clustering\main.py -t
py asreviewcontrib\semantic_clustering\main.py --testfile
```

This argument will start the processing file using the `van_de_Schoot_2017`
dataset, and can be used as a quick functionality test.

### Interactive app
```console
py asreviewcontrib\semantic_clustering\main.py -a
py asreviewcontrib\semantic_clustering\main.py --app
```

After the processing has finished with either a new file or the test file, a
file called `kmeans_df.csv` has appeared in the data folder. This file can be
used in the interactive app. When the server has been started with the command
above, it can be found at [`http://127.0.0.1:8050/`](http://127.0.0.1:8050/) in your browser.

## License

MIT license

# ASReview Semantic Clustering
This repository contains the Semantic Clustering plugin for
[ASReview](https://github.com/asreview/asreview). It applies multiple techniques
(SciBert, PCA, T-SNE, KMeans, a custom Cluster Optimizer) to an [ASReview data
object](https://asreview.readthedocs.io/en/latest/API/generated/asreview.data.ASReviewData.html#asreview.data.ASReviewData),
in order to cluster records based on semantic differences. The end result is an
interactive dashboard:

![Alt Text](/docs/cord19_semantic_clusters.gif)


## Getting started

The packaged is called `asreview-semantic-clustering` and can be installed with:

```console
pip install .
```
from the download folder,
or run the following to install directly:

```console
python -m pip install git+https://github.com/asreview/semantic-clusters.git
```

### Commands

For help use:

```console
asreview semantic-clustering -h
asreview semantic-clustering --help
```

Other options are:

```console
asreview semantic-clustering -f <input.csv or url> -o <output.csv>
asreview semantic-clustering --filepath <input.csv or url> --output <output.csv>
```

```console
asreview semantic-clustering -t -o <output.csv>
asreview semantic-clustering --testfile --output <output.csv>
```

```console
asreview semantic-clustering -a <output.csv>
asreview semantic-clustering --app <output.csv>
```

```console
asreview semantic-clustering -v
asreview semantic-clustering --version
```


## Usage
The functionality of the semantic clustering extension is implemented in a [subcommand extension](https://asreview.readthedocs.io/en/latest/API/extension_dev.html#subcommand-extensions). The
following commands can be run:

### Processing
In the processing phase, a dataset is processed and clustered for use in the interactive interface. The following options are available:

```console
asreview semantic-clustering -f <input.csv or url> -o <output_file.csv>
asreview semantic-clustering -t -o <output_file.csv>
```

`-f` will process a file and store the results in the file specified in `-o`. Semantic-clustering uses an [ASReview data object](https://asreview.readthedocs.io/en/latest/API/generated/asreview.data.ASReviewData.html#asreview.data.ASReviewData), and can handle either a file or url:

```console
asreview semantic-clustering -f "https://raw.githubusercontent.com/asreview/systematic-review-datasets/master/datasets/van_de_Schoot_2017/output/van_de_Schoot_2017.csv" -o output.csv
asreview semantic-clustering -f van_de_Schoot_2017.csv -o output.csv
```

Using `-t` instead of `-f` uses the [`van_de_Schoot_2017`](https://asreview.readthedocs.io/en/latest/intro/datasets.html?highlight=ptsd#featured-datasets) dataset instead. 

If an output file is not specified, `output.csv` is used.

### Dashboard
Running the dashboard server is also done from the command line. This command will start a Dashy server in the console and visualize the processed file.

```console
asreview semantic-clustering -a output.csv
asreview semantic-clustering --app output.csv
```

When the server has been started with the command above, it can be found at [`http://127.0.0.1:8050/`](http://127.0.0.1:8050/) in
your browser.

## License

MIT license

## Contact
Got ideas for improvement? For any questions or remarks, please send an email to
[asreview@uu.nl](mailto:asreview@uu.nl).


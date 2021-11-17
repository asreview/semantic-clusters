# ASReview Semantic Clustering
This repository contains the Semantic Clustering plugin for
[ASReview](https://github.com/asreview/asreview). It applies multiple techniques
(SciBert, PCA, T-SNE, KMeans, a custom Cluster Optimizer) to an [ASReview data
object](https://asreview.readthedocs.io/en/latest/API/generated/asreview.data.ASReviewData.html#asreview.data.ASReviewData),
in order to cluster records based on semantic differences. The end result is an
interactive dashboard:

![Alt Text](/docs/cord19_semantic_clusters.gif)


# Getting started

The packaged is called `semantic_clustering` and can be installed from the
download folder with:

```shell
pip install .
```
or from the command line directly with:

```shell
python -m pip install git+https://github.com/asreview/semantic-clusters.git
```

## Commands

For help use:

```shell
asreview semantic_clustering -h
asreview semantic_clustering --help
```

Other options are:

```shell
asreview semantic_clustering -f <input.csv or url> -o <output.csv>
asreview semantic_clustering --filepath <input.csv or url> --output <output.csv>
```

```shell
asreview semantic_clustering -t -o <output.csv>
asreview semantic_clustering --testfile --output <output.csv>
```

```shell
asreview semantic_clustering -a <output.csv>
asreview semantic_clustering --app <output.csv>
```

```shell
asreview semantic_clustering -v
asreview semantic_clustering --version
```

```shell
asreview semantic_clustering --transformer
```


# Usage
The functionality of the semantic clustering extension is implemented in a
[subcommand
extension](https://asreview.readthedocs.io/en/latest/API/extension_dev.html#subcommand-extensions).
The following commands can be run:

## Processing
In the processing phase, a dataset is processed and clustered for use in the
interactive interface. The following options are available:

```shell
asreview semantic_clustering -f <input.csv or url> -o <output_file.csv>
```

Using `-f` will process a file and store the results in the file specified in
`-o`. 

Semantic_clustering uses an [`ASReviewData`
object](https://asreview.readthedocs.io/en/latest/API/generated/asreview.data.ASReviewData.html#asreview.data.ASReviewData),
and can handle either a file or url:

```shell
asreview semantic_clustering -f "https://raw.githubusercontent.com/asreview/systematic-review-datasets/master/datasets/van_de_Schoot_2017/output/van_de_Schoot_2017.csv" -o output.csv
asreview semantic_clustering -f van_de_Schoot_2017.csv -o output.csv
```

If an output file is not specified, `output.csv` is used as output file name.

### Test file
```shell
asreview semantic_clustering -t -o <output_file.csv>
```

Using `-t` instead of `-f` uses the
[`van_de_Schoot_2017`](https://asreview.readthedocs.io/en/latest/intro/datasets.html#featured-datasets)
dataset as input file. This way, the plugin can easily be tested.

### Transformer
Semantic Clustering uses the
[`allenai/scibert_scivocab_uncased`](https://github.com/allenai/scibert)
transformer model as default setting. Using the `--transformer <model>` option,
another model can be selected for use instead:

```shell
asreview semantic_clustering -t -o <output_file.csv> --transformer bert-base-uncased
```

Any pretrained model will work.
[Here](https://huggingface.co/transformers/pretrained_models.html) is an example
of models, but more exist.

## Dashboard
Running the dashboard server is also done from the command line. This command
will start a [Dash](https://plotly.com/dash/) server in the console and
visualize the processed file.

```shell
asreview semantic_clustering -a output.csv
asreview semantic_clustering --app output.csv
```

When the server has been started with the command above, it can be found at
[`http://127.0.0.1:8050/`](http://127.0.0.1:8050/) in your browser.

# License

MIT license

# Contact
Got ideas for improvement? For any questions or remarks, please send an email to
[asreview@uu.nl](mailto:asreview@uu.nl).


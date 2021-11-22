# ASReview Semantic Clustering
This repository contains the Semantic Clustering plugin for
[ASReview](https://github.com/asreview/asreview). It applies multiple techniques
(SciBert, PCA, T-SNE, KMeans, a custom Cluster Optimizer) to a [ASReview compatible](https://asreview.readthedocs.io/en/latest/intro/datasets.html) dataset
in order to cluster records based on semantic differences. The end result is an
interactive dashboard.


## Installation

The packaged is called `semantic_clustering` and can be installed from the
download folder with:

```shell
pip install .
```
or from the command line directly with:

```shell
python -m pip install git+https://github.com/asreview/semantic-clusters.git
```

For help with usage, and to see available commands, use:

```shell
asreview semantic_clustering -h
```
The semantic clustering extension is a subcommand extension, meaning its usage is implemented via the command line interface. For more information on subcommand extensions, see the [subcommand extension documentation](https://asreview.readthedocs.io/en/latest/extensions/overview_extensions.html#subcommand-extensions).

## Usage
Before the dashboard can be initiated, the dataset has to be prepared and clustered.

### Processing
Processing a file is done using the `filepath` and `output` options. The `-f` argument points towards a file to be processed, and the results are stored in the file specified in `-o`:

```shell
asreview semantic_clustering -f <input.csv or url> -o <output_file.csv>
```

Semantic_clustering uses the ASReview data format and can handle files, urls and benchmark sets:

```shell
asreview semantic_clustering -f benchmark:van_de_schoot_2017 -o output.csv
asreview semantic_clustering -f input\van_de_Schoot_2017.csv -o output.csv
asreview semantic_clustering -f https://url-to-file.org/file.csv -o output.csv
```
For information on how to prepare a file for use with ASReview or the extension, see the [ASReview dataset documentation](https://asreview.readthedocs.io/en/latest/intro/datasets.html).

*Note: If an output file is not specified, `output.csv` is used instead.*

### Dashboard
Running the dashboard server is also done from the command line. This command
will start a [Dash](https://plotly.com/dash/) server in the console and
visualize the processed file.

```shell
asreview semantic_clustering --app output.csv
```

When the server has been started with the command above, it can be found at
[`http://127.0.0.1:8050/`](http://127.0.0.1:8050/) in your browser.


### Transformer
An advanced option is the usage of special transformers. Semantic Clustering uses the
[`allenai/scibert_scivocab_uncased`](https://github.com/allenai/scibert)
transformer model by default, but by using the `--transformer <model>` option,
another model can be selected for use:

```shell
asreview semantic_clustering -f <file> -o <output_file.csv> --transformer bert-base-uncased
```

Any pretrained model will work.
[Here](https://huggingface.co/transformers/pretrained_models.html) is an example
of models, but more exist.

## License

MIT license

## Contact
Got ideas for improvement? For any questions or remarks, please send an email to
[asreview@uu.nl](mailto:asreview@uu.nl).


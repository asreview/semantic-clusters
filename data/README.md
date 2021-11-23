# Dataset examples
The following files are processed datasets used for the example gifs. The
following commands were used:

## PTSD dataset:
```shell
<<<<<<< Updated upstream
asreview semantic-clustering --filepath https://raw.githubusercontent.com/asreview/systematic-review-datasets/master/datasets/van_de_Schoot_2017/output/van_de_Schoot_2017.csv --output ptsd_scibert.csv --transformer allenai/scibert_scivocab_uncased
asreview semantic-clustering --app ptsd_scibert.csv
```

=======
asreview semantic_clustering -f benchmark:van_de_schoot_2017 --output output\ptsd_scibert.csv --transformer bert-base-uncased
asreview semantic_clustering --app output\ptsd_scibert.csv
```

![ptsd.gif](img\ptsd.gif)	

>>>>>>> Stashed changes
## CORD19 dataset:
```shell
asreview semantic-clustering --filepath https://github.com/asreview/asreview-covid19/raw/master/datasets/cord19-2020/cord19_latest_20191201_new.csv --output cord19_scibert.csv --transformer allenai/scibert_scivocab_uncased
asreview semantic-clustering --app cord19_scibert.csv
```

## Depression dataset:
```shell
asreview semantic-clustering --filepath "" --output depression_scibert.csv --transformer allenai/scibert_scivocab_uncased
asreview semantic-clustering --app depression_scibert.csv 
```
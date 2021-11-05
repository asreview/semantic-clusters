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

# import ASReview
from tqdm import tqdm
from dim_reduct import run_pca
from dim_reduct import t_sne
from dim_reduct import plot_embs
from asreview.data import ASReviewData

# import numpy
import numpy as np

# import transformer autotokenizer and automodel
from transformers import AutoTokenizer, AutoModel

# disable transformer warning
from transformers import logging
logging.set_verbosity_error()

#import tqdm
tqdm.pandas()


def SemanticClustering(asreview_data_object):

    # load data
    print("Loading data...")
    data = load_data(asreview_data_object)

    # cut data for testing
    data = data.iloc[:300, :]

    # load scibert transformer
    print("Loading scibert transformer...")
    transformer = 'allenai/scibert_scivocab_uncased'

    # load transformer and tokenizer
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(transformer)
    model = AutoModel.from_pretrained(transformer)

    # tokenize abstracts and add to data
    print("Tokenizing abstracts...")
    data['tokenized'] = data['abstract'].progress_apply(lambda x: tokenizer.encode_plus(
        x,
        add_special_tokens=False,
        max_length=512,
        pad_to_max_length=True,
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

    plot_embs(tsne, None, 1000)


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

# imports 
# System stuff
import os
import sys
import json
import pickle
from shutil import rmtree

# Numerical / data imports
import numpy as np
import pandas as pd

# Torch-y stuff
import torch

# Transformers
from transformers import AutoTokenizer, AutoModelWithLMHead#, AutoModelForMaskedLM
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, models

# Own functions
from load_data import load_from_parses, load_from_json, load_dataframe

def generate_embeddings(model, tokenizer, df, use_covidbert=False):
    """Function that generates (CovidBERT) embeddings
    Args: 
      model: The (transformer) model to be used, e.g. CovidBERT
      tokenizer: Tokenizer corresponding to the model used
      df: DataFrame containing parsed data from the CORD-19 document parses
      use_covidbert: (bool) To set whether we use covidbert or regular BERT
    Returns:
      embeddings: Contextualized embeddings from the specified model
    """

    # Path structure
    if not os.path.exists("data"):
        os.makedirs("data")
    embs_path = os.path.join("data","embs")
    if not os.path.exists(embs_path):
        os.makedirs(embs_path)

    # Only use ones without missing abstracts
    # (Effectively circumvented using titles instead while building abstracts)
    miss_abs = df[df['Abstract'].isnull()]
    no_miss_abs = df.drop(miss_abs.index)

    for i, abstract in enumerate(no_miss_abs['Abstract']):

        # Only do it for first 2000 for testing purposes
        if i > 1999:
            break

        # Get cord uid and title for article
        cord_uid = no_miss_abs.iloc[i,0]
        title = no_miss_abs.iloc[i,1]

        if i % 10 == 0:
            print(f"Abstract: {i:7d}, cord_uid {cord_uid}")

        # In case we want to use CovidBERT
        if use_covidbert:

            """"Add preprocessing for tokens instead of split"""
            abstract = abstract.split(" ")
            outputs = model.encode(abstract)

        # Use Regular BERT instead
        else:

            # Use (BERT Tokenizer and get outputs tuple
            tokenized = tokenizer.encode(abstract, return_tensors="pt")
            outputs = model(tokenized)

            # Retrieve last hidden states and CLS token
            #last_hidden_states = outputs[0]
            cls_token = outputs[1]

            # Write single CLS token to file to prevent RAM build-up
            # Cast to np if true
            to_numpy = True 
            if to_numpy:
                cls_token = cls_token.detach().numpy()
            embs_file = os.path.join("data","embs", str(cord_uid)+".pickle")
            with open(embs_file, "wb+") as file:
                pickle.dump(cls_token, file)

    print("Did I encode all abstracts and save pickle?")

def load_model(use_covidbert=False):
    """Function that loads and returns the CovidBERT model"""

    # # Load CovidBERT
    # if use_covidbert:
    #     print("Loading model...")
    #     model = AutoModelForMaskedLM.from_pretrained("deepset/covid_bert_base")
    #     print("Loading tokenizer...")
    #     tokenizer = AutoTokenizer.from_pretrained("deepset/covid_bert_base")
    #     print("Finished loading the model successfully!")

        #model = SentenceTransformer(model_path)

    # #Load CovidBERT
    # if use_covidbert:
    #     print("Loading model...")
    #     model = AutoModelWithLMHead.from_pretrained("manueltonneau/clinicalcovid-bert-nli")
    #     print("Loading tokenizer...")
    #     print("\n")
    #     tokenizer = AutoTokenizer.from_pretrained("manueltonneau/clinicalcovid-bert-nli")
    #     print("\n")
    #     print("Finished loading the model successfully!")

    #     # Save the model to model path
    #     model_path = os.path.join("models","clinicalcovid")
    #     if not os.path.exists(model_path):
    #         os.makedirs(model_path)
    #     model.save_pretrained(model_path)
    #     tokenizer.save_pretrained(model_path)

    #     model = SentenceTransformer(model_path)

    # Load CovidBERT 
    if use_covidbert:
        print("Loading model...")
        model = AutoModelWithLMHead.from_pretrained("gsarti/covidbert-nli")
        print("Loading tokenizer...")
        print("\n")
        tokenizer = AutoTokenizer.from_pretrained("gsarti/covidbert-nli")
        print("\n")
        print("Finished loading the model successfully!")

        # Save the model to model path
        model_path = os.path.join("models","gsarticovid")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"Successfully saved model to {model_path}")

        print("Loading Sentence Transformer now!")
        word_embedding_model = models.BERT(
            model_path,
            # max_seq_length=args.max_seq_length,
            # do_lower_case=args.do_lower_case
        )
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        rmtree(model_path)
        model.save(model_path)
        print("Finished building Sentence Transformer!")

    # Load regular BERT
    else:
        print("Loading BERT")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        print("Finished loading BERT")

    return model, tokenizer

if __name__ == "__main__":

    # First check if we have the right folder structure
    if not os.path.exists("data"):
        os.makedirs("data")

    # Whether we use CovidBERT or normal BERT
    use_covidbert = False

    # Load model and tokenizer
    model, tokenizer = load_model(use_covidbert=use_covidbert)

    # Use bulky loader if we don't have the cord19.json yet
    cord19_json_path = os.path.join("data", "cord19.json")
    if not os.path.exists(cord19_json_path):
        load_from_parses()

    # Load the file from the created json
    data = load_from_json(cord19_json_path)

    # Load dataframe if csv exists, otherwise create it
    df = load_dataframe(data)

    # If embeddings don't exist, create them
    embs_path = os.path.join("data","embs")
    if not os.path.exists(embs_path):
        os.makedirs(embs_path)

    if len(os.listdir(embs_path)) == 0:
        generate_embeddings(model, tokenizer, df, use_covidbert=use_covidbert)
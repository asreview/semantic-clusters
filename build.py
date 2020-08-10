# imports 
# System stuff
import os
import sys
import json
import pickle

# Numerical / data imports
import numpy as np
import pandas as pd

# Torch-y stuff
import torch

# Transformers
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForMaskedLM
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, models

# Own functions
from load_data import load_from_parses, load_from_json, load_dataframe

def generate_embeddings(model, tokenizer, df):
    """Function that generates (CovidBERT) embeddings
    Args: 
      model: The (transformer) model to be used, e.g. CovidBERT
      tokenizer: Tokenizer corresponding to the model used
      df: DataFrame containing parsed data from the CORD-19 document parses
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

        # Only do it for first 10001
        if i > 10000:
            break

        # Get cord uid and title for article
        cord_uid = no_miss_abs.iloc[i,0]
        title = no_miss_abs.iloc[i,1]

        if i % 10 == 0:
            print(f"Abstract: {i:7d}, cord_uid {cord_uid}")

        # Use (Covid)BERT Tokenizer and get outputs tuple
        tokenized = tokenizer.encode(abstract, return_tensors="pt")
        outputs = model(tokenized)

        # Retrieve last hidden states and CLS token
        #last_hidden_states = outputs[0]
        cls_token = outputs[1]
        #print(cls_token.size())

        # print("LHS size: ", last_hidden_states.size())
        # print("CLS size: ", cls_token.size())

        # Build dict
        #embs_dict[cord_uid] = cls_token

        # Write single CLS token to file to prevent RAM build-up
        # Cast to np if true
        to_numpy = True 
        if to_numpy:
            cls_token = cls_token.detach().numpy()
        embs_file = os.path.join("data","embs_np", str(cord_uid)+".pickle")
        with open(embs_file, "wb+") as file:
            pickle.dump(cls_token, file)

    print("Did I encode all abstracts and save pickle?")

def load_model():
    """Function that loads and returns the CovidBERT model"""

    # print("Loading model...")
    # model = AutoModelWithLMHead.from_pretrained("gsarti/covidbert-nli")
    # print("Loading tokenizer...")
    # print("\n\n")
    # tokenizer = AutoTokenizer.from_pretrained("gsarti/covidbert-nli")
    # print("Finished loading the model successfully!")

    # print("Loading model...")
    # model = AutoModelForMaskedLM.from_pretrained("deepset/covid_bert_base")
    # print("Loading tokenizer...")
    # tokenizer = AutoTokenizer.from_pretrained("deepset/covid_bert_base")
    # print("Finished loading the model successfully!")

    # Load regular BERT for testing purposes
    print("Loading BERT")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    print("Finished loading BERT")

    return model, tokenizer

if __name__ == "__main__":

    # First check if we have the right folder structure
    if not os.path.exists("data"):
        os.makedirs("data")

    # Load model and tokenizer
    model, tokenizer = load_model()
    #model = SentenceTransformer("./src/models/covidbert/")

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
    if len(os.listdir(embs_path)) == 0 or not os.path.exists(embs_path):
        generate_embeddings(model, tokenizer, df)
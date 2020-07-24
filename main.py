# imports 
# System stuff
import os
import sys
import json

# Numerical / data imports
import numpy as np
import pandas as pd

# Torch-y stuff
import torch

# Transformers
from transformers import AutoTokenizer, AutoModelWithLMHead

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
    miss_abs = df[df['Abstract'].isnull()]
    no_miss_abs = df.drop(miss_abs.index)
    for abstract in no_miss_abs['Abstract']:
        tokenized = tokenizer.encode(abstract)
        tokens_tensor = torch.tensor(tokenized)
        print("Size: ",tokens_tensor.size())
        outputs = model(tokens_tensor.unsqueeze(1)) 
        print(outputs)
        print(type(outputs))
        break

def load_model():
    """Function that loads and returns the CovidBERT model"""

    print("Loading model...")
    model = AutoModelWithLMHead.from_pretrained("gsarti/covidbert-nli")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gsarti/covidbert-nli")
    print("Finished loading the model successfully!")

    return model, tokenizer

if __name__ == "__main__":

    # First check if we have the right folder structure
    if not os.path.exists("data"):
        os.makedirs("data")

    # Load model and tokenizer
    model, tokenizer = load_model()

    # Use bulky loader if we don't have the cord19.json yet
    cord19_json_path = os.path.join("data", "cord19.json")
    if not os.path.exists(cord19_json_path):
        load_from_parses()

    # Load the file from the created json
    data = load_from_json(cord19_json_path)

    # Load dataframe if csv exists, otherwise create it
    df = load_dataframe(data)
    print(df.head(5))

    generate_embeddings(model, tokenizer, df)



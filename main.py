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
from load_data import load_from_parses, load_from_json



def load_model():
    """Function that loads and returns the CovidBERT model"""

    print("Loading model...")
    model = AutoModelWithLMHead.from_pretrained("gsarti/covidbert-nli")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gsarti/covidbert-nli")
    print("Finished loading the model successfully!")

    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_model()

    # Use bulky loader if we don't have the cord19.json yet
    cord19_path = os.path.join("data", "cord19.json")
    if not os.path.exists(cord19_path):
        load_from_parses()

    # Load the file from the created json
    data = load_from_json(cord19_path)



# imports 
# Numerical / data imports
import numpy as np
import pandas as pd

# Torch-y stuff
import torch

# Transformers
from transformers import AutoTokenizer, AutoModelWithLMHead

def load_model():
    """Function that loads and returns the CovidBERT model"""

    print("Loading model...")
    model = AutoModelWithLMHead.from_pretrained("gsarti/covidbert-nli")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gsarti/covidbert-nli")
    print("Finished loading the model successfully!")

    return model, tokenizer

if __name__ == "__main__":
    load_model()

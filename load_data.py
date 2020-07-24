# System / os stuff
import csv
import os
import json

# Collections
from collections import defaultdict
import pandas as pd

def load_from_json(cord19_path):
    """Function that loads from a saved cord19.json file"""
    if os.path.exists(cord19_path):
        with open(cord19_path) as json_file:
            data = json.load(json_file)
    else:
        raise ValueError("The provided path does not exist! Please use load_from_parses() to create the json file.")
    return data


def load_from_parses():
    """Function to load the CORD-19 dataset from the provided JSONS"""

    cord_uid_to_text = defaultdict(list)

    # open the file
    if not os.path.exists("data"):
        os.makedirs("data")
    metadata = os.path.join("data", "metadata.csv")

    with open(metadata) as f_in:
        reader = csv.DictReader(f_in)
        for i, row in enumerate(reader):
        
            # access some metadata
            cord_uid = row['cord_uid']
            title = row['title']
            abstract = row['abstract']
            authors = row['authors'].split('; ')

            # access the full text (if available) for Intro
            introduction = []
            if row['pdf_json_files']:
                for json_path in row['pdf_json_files'].split('; '):

                    # Data is saved in "data" folder, so navigate there instead
                    json_path = os.path.join("data", json_path)
                    with open(json_path) as f_json:
                        full_text_dict = json.load(f_json)
                        
                        # grab introduction section from *some* version of the full text
                        for paragraph_dict in full_text_dict['body_text']:
                            paragraph_text = paragraph_dict['text']
                            section_name = paragraph_dict['section']
                            if 'intro' in section_name.lower():
                                introduction.append(paragraph_text)

                        # stop searching other copies of full text if already got introduction
                        if introduction:
                            break
            if i % 100 == 0:
                print(f"At row {i} now!")

            # save for later usage
            cord_uid_to_text[cord_uid].append({
                'title': title,
                'abstract': abstract,
                'introduction': introduction
            })
    
    print(type(cord_uid_to_text))

    # Save the full CORD dataset as one json file
    data_path = os.path.join("data","cord19.json")
    with open(data_path, 'w') as fp:
        json.dump(cord_uid_to_text, fp)

# System / os stuff
import csv
import os
import json

# Collections
from collections import defaultdict
import pandas as pd

def load_dataframe(data):
    """Function that creates a DataFrame from the json if there is no csv containing it"""

    # Either csv already exists and we can simply load it
    df_path = os.path.join("data","cord19_df.csv")
    if os.path.exists(df_path):
        print("Loading DataFrame...")
        df = pd.read_csv(df_path)

    # Or..
    else:
        print("Creating DataFrame...")
        # Create a DataFrame instead
        d = []

        # JSON data contains dictionaries in a list for each entry
        for _, (_,val) in enumerate(data.items()):
            val_dict = val[0]
            cord_uid = val_dict['cord_uid']
            title = val_dict['title']
            abstract = val_dict['abstract']
            intro = val_dict['introduction']

            d.append((cord_uid,title,abstract,intro))

        # Turn list of tuples into DataFrame and write it to a CSV
        df = pd.DataFrame(d, columns=('cord_uid','Title', 'Abstract', 'Introduction'))
        df.to_csv(df_path, index=False)

    return df

def load_from_json(cord19_json_path):
    """Function that loads the data from a saved cord19.json file"""
    if os.path.exists(cord19_json_path):
        with open(cord19_json_path) as json_file:
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
            #authors = row['authors'].split('; ')

            # Abstracts are quite big, so cut them
            abstract = abstract.split(" ")
            if len(abstract) > 200:
                abstract = abstract[:200]
            abstract = " ".join(abstract)

            # # If we don't have an abstract, use title
            # if len(abstract) < 5:
            #     abstract = title

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
                'cord_uid': cord_uid,
                'title': title,
                'abstract': abstract,
                'introduction': introduction
            })
    
    print(type(cord_uid_to_text))

    # Save the full CORD dataset as one json file
    data_path = os.path.join("data","cord19.json")
    with open(data_path, 'w') as fp:
        json.dump(cord_uid_to_text, fp)

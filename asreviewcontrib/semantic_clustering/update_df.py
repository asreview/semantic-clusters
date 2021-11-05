# imports

# System
import os

# data
import numpy as np 
import pandas as pd  

# Self
from load_data import load_from_json, load_dataframe, load_from_parses

############################# READ ME #############################
##### This is a file with several functions that was used to ######
##### recombine a saved dataFrame with titles and abstracts. ######
###################################################################

def update_titles():
    """Function to add titles to the kmeans df"""

    # Use bulky loader if we don't have the cord19.json yet
    cord19_json_path = os.path.join("data", "cord19.json")
    if not os.path.exists(cord19_json_path):
        load_from_parses()

    # Load the file from the created json
    data = load_from_json(cord19_json_path)

    # Load dataframe if csv exists, otherwise create it
    df = load_dataframe(data)
    df = df.iloc[:2000,:-1]

    # Load other df
    kmeans_df_path = os.path.join("data","dataframes","kmeans_df.csv")
    kmeans_df = pd.read_csv(kmeans_df_path)

    # Retrieve titles
    titles = []
    for i, cord_uid in enumerate(kmeans_df['cord_uid']):
        for df_uid in df.cord_uid.values:
            if cord_uid == df_uid:
                title = df[df.cord_uid == df_uid].Title.values.tolist()
                titles.append(title)

    flatten = [title[0] for title in titles]
    kmeans_df['Title'] = np.array(flatten)
    print(kmeans_df.head())
    print(kmeans_df.columns)

    kmeans_df.to_csv(kmeans_df_path,index=None)
    

def update_abstracts():
    """Function to add abstracts to the kmeans df"""

    # Use bulky loader if we don't have the cord19.json yet
    cord19_json_path = os.path.join("data", "cord19.json")
    if not os.path.exists(cord19_json_path):
        load_from_parses()

    # Load the file from the created json
    data = load_from_json(cord19_json_path)

    # Load dataframe if csv exists, otherwise create it
    df = load_dataframe(data)
    df = df.iloc[:2000,:-1]

    print(df.head())
    print(df.columns)
    print(df.shape)
    print("\n\n")

    exit()

    # Load other df
    kmeans_df_path = os.path.join("data","dataframes","kmeans_df.csv")
    kmeans_df = pd.read_csv(kmeans_df_path)

    print(kmeans_df.head())
    print(kmeans_df.columns)
    print(kmeans_df.shape)

    # print()
    # print(kmeans_df[kmeans_df['cord_uid'] == "6c6cw80p"])
    # print(kmeans_df[kmeans_df['cord_uid'] == "ug7v899j"])

    abstracts = []
    for i,cord_uid in enumerate(kmeans_df['cord_uid']):
        for df_uid in df.cord_uid.values:
            if cord_uid == df_uid:
                abstract = df[df.cord_uid == df_uid].Abstract.values.tolist()
                abstracts.append(abstract)
    #print(abstracts[:10])

    flatten = [absy[0] for absy in abstracts]
    kmeans_df['Abstract'] = np.array(flatten)
    print(kmeans_df.head())
    print(kmeans_df.columns)
    print(kmeans_df[kmeans_df.cord_uid == "ug7v899j"].Abstract)

    kmeans_df.to_csv(kmeans_df_path,index=None)

if __name__ == "__main__":
    #update_abstracts()
    #update_titles()
    pass

import pandas as pd
import numpy as np

# Function to collect all csvs for a subreddit and return a dataframe
def subreddit_to_df(subreddit):
    import os
    # Get file names in data folder for subreddit
    subreddit_files = []
    for file in os.listdir("../data"):
        split_filename = file.split("-")
        if split_filename[0] == subreddit and split_filename[1] != "full":
            subreddit_files.append(file)
    
    # Read csvs into dataframes then concatenate each of them 
    df = pd.concat([pd.read_csv(f'./data/{file}') for file in subreddit_files]).drop(columns="Unnamed: 0")
    
    # Convert UTC time stamp into a datetime column
    df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    
    # Create column with all text
    df['alltext'] = df['title'].str.cat(df['selftext'], sep = " ")
    
    return df


# Function to remove columns, deal with missing text

def process_df(df):
    # Create new copy of dataframe to be processed
    new_df = df.copy()
    
    # Drop all columns except for selftext and subreddit
    new_df.drop(columns=["id", "created_utc", "subreddit_id", "url", "datetime"], inplace=True)
    
    # Replace all "[removed]" values from selftext with np.nan
    new_df.replace("[removed]", np.nan, inplace=True)
    
    # Drop np.nans
    new_df.dropna(inplace=True)
    
    return new_df